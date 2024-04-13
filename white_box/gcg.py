from dataclasses import dataclass
import gc
from tqdm import tqdm
from typing import List, Union

import torch
import transformers
from torch import Tensor

from accelerate.utils import find_executable_batch_size

from white_box.model_wrapper import slice_acts
from white_box.monitor import Monitor, ActMonitor, TextMonitor

@dataclass
class GCGConfig:
    num_steps: int = 75
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 1
    add_space_before_target: bool = True
    device: str = "cuda"

class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = [] # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if len(self.buffer) < self.size:
            self.buffer.append((float('inf'), optim_ids))
            return

        self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])

def sample_ids_from_grad(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    topk: int = 256,
    n_replace: int = 1,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized 
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace: int
            the number of token positions to update per sequence
    
    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.stack([torch.randperm(n_optim_tokens, device=grad.device) for _ in range(search_width)])[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids

def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids) 
            token ids 
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer
    
    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
           filtered_ids.append(ids[i]) 
    
    return torch.stack(filtered_ids)

def compute_candidates_loss(
    search_batch_size: int, 
    model: transformers.PreTrainedModel,
    kv_cache: tuple,
    input_embeds: Tensor, 
    target_ids: Tensor,
    monitor : Union[ActMonitor, TextMonitor] = None,
):
    """Computes the GCG loss on all candidate token id sequences.

    Args:
        search_batch_size : int
            the number of candidate sequences to evaluate in a given batch
        model : ~transformers.PreTrainedModel
            the model used to compute the loss
        kv_cache : tuple
            the KV cache generated from all tokens before the start of the optimized string
        input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
            the embeddings of the `search_width` candidate sequences to evaluate
        target_ids : Tensor, shape = (1, n_target_ids)
            the token ids of the target sequence 
    """
    all_loss = []
    for i in range(0, input_embeds.shape[0], search_batch_size):
        with torch.no_grad():
            input_embeds_batch = input_embeds[i:i+search_batch_size]
            current_batch_size = input_embeds_batch.shape[0]

            kv_cache_batch = []
            for i in range(len(kv_cache)):
                kv_cache_batch.append([])
                for j in range(len(kv_cache[i])):
                    kv_cache_batch[i].append(kv_cache[i][j].expand(current_batch_size, -1, -1, -1))

            outputs = model(inputs_embeds=input_embeds_batch, past_key_values=kv_cache_batch)
            logits = outputs.logits

            tmp = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(current_batch_size, 1)

            if isinstance(monitor, ActMonitor):
                monitor_input = slice_acts(outputs, 
                        N_TOKS = 1, 
                        layers = monitor.layer,
                        tok_idxs = -1,
                        return_prompt_acts = False)
            elif isinstance(monitor, TextMonitor):
                monitor_input =  tokenizer.batch_decode(before_ids ) + tokenizer.batch_decode(optim_ids) + tokenizer.batch_decode(after_ids) #! fix
        
            monitor_loss = monitor.get_loss(monitor_input)
        
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none") + monitor_loss
            loss = loss.view(current_batch_size, -1).mean(dim=-1)
            all_loss.append(loss)

            del outputs
            torch.cuda.empty_cache()
            gc.collect()

    return torch.cat(all_loss, dim=0)

def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    monitor : Union[ActMonitor, TextMonitor] = None,
    config: GCGConfig = None, 
) -> dict:
    """Generates a single optimized string using GCG. 

    Args:
        model: The model to optimize on.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.
        monitor : A monitor that GCG is trying to jailbreak as well
    
    Returns:
        A dict that stores losses and the final optim_str
    """
    if config == None:
        config = GCGConfig()
    
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    if not any(["{optim_str}" in d["content"] for d in messages]):
        messages[-1]["content"] = messages[-1]["content"] + " {optim_str}"

    model.to(config.device)

    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
    # Remove the BOS token -- this will get added when tokenizing, if necessary
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "")
    before_str, after_str = template.split("{optim_str}")

    target = " " + target if config.add_space_before_target else target

    # Tokenize everything that doesn't get optimized
    before_ids = tokenizer([before_str], padding=False)["input_ids"]
    after_ids = tokenizer([after_str], add_special_tokens=False)["input_ids"]
    target_ids = tokenizer([target], add_special_tokens=False)["input_ids"]

    before_ids, after_ids, target_ids = [torch.tensor(ids, device=config.device) for ids in (before_ids, after_ids, target_ids)]

    # Embed everything that doesn't get optimized
    embedding_layer = model.get_input_embeddings()
    before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

    # Compute the KV Cache for tokens that appear before the optimized tokens
    with torch.no_grad():
        output = model(inputs_embeds=before_embeds, use_cache=True)
        kv_cache = output.past_key_values
    
    optim_ids = tokenizer(config.optim_str_init, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)
    buffer = AttackBuffer(config.buffer_size)
    buffer.add(loss=float('inf'), optim_ids=optim_ids)
    for _ in range(config.buffer_size - 1):
        buffer.add(loss=float('inf'), optim_ids=torch.randint_like(optim_ids, 0, tokenizer.vocab_size, dtype=torch.long, device=config.device))
    
    losses = []
    optim_strings = []
    for i in tqdm(range(config.num_steps)):
        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(dtype=model.dtype, device=config.device)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        
        input_embeds = torch.cat([optim_embeds, after_embeds, target_embeds], dim=1)
        output = model(inputs_embeds=input_embeds, past_key_values=kv_cache, output_hidden_states = True if monitor is not None else False)
        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[..., shift-1:-1, :].contiguous() # (1, num_target_ids, vocab_size)
        shift_labels = target_ids
        
        if isinstance(monitor, ActMonitor):
            monitor_input = slice_acts(output, 
                        N_TOKS = 1, 
                        layers = monitor.layer,
                        tok_idxs = -1,
                        return_prompt_acts = False)
        elif isinstance(monitor, TextMonitor):
            monitor_input =  tokenizer.batch_decode(before_ids ) + tokenizer.batch_decode(optim_ids) + tokenizer.batch_decode(after_ids) #! fix
        
        monitor_loss = monitor.get_loss(monitor_input)
        
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) + monitor_loss
        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        # Sample candidate token sequences
        sampled_ids = sample_ids_from_grad(
            optim_ids.squeeze(0),
            optim_ids_onehot_grad.squeeze(0),
            config.search_width,
            config.topk,
            config.n_replace
        )
        sampled_ids = filter_ids(sampled_ids, tokenizer)
        new_search_width = sampled_ids.shape[0]

        input_embeds = torch.cat([
            embedding_layer(sampled_ids),
            after_embeds.repeat(new_search_width, 1, 1),
            target_embeds.repeat(new_search_width, 1, 1)
        ], dim=1)

        # Compute loss on all candidate sequences 
        loss = find_executable_batch_size(compute_candidates_loss, new_search_width)(
            model,
            kv_cache,
            input_embeds,
            target_ids
        )

        current_loss = loss.min().item()
        losses.append(current_loss)

        optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

        if current_loss < buffer.get_highest_loss():
            buffer.add(current_loss, optim_ids)

        optim_str = tokenizer.batch_decode(buffer.get_best_ids())
        optim_strings.append(optim_str)

        print(f"step: {i} | optim_str: {optim_str} | loss: {buffer.get_lowest_loss()}")
    
    return {
        "losses": losses,
        "optim_strings": optim_strings,
    }