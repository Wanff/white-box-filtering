from dataclasses import dataclass
import gc
from tqdm import tqdm
from typing import List, Union
import heapq
import numpy as np

import torch
import transformers
from torch import Tensor

from accelerate.utils import find_executable_batch_size

from ..model_wrapper import slice_acts, ModelWrapper
from ..monitor import Monitor, ActMonitor, TextMonitor

@dataclass
class GCGConfig:
    num_steps: int = 75
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! "
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 1
    add_space_before_target: bool = True
    device: str = "cuda"
    gcg_loss_weight : float = 1.0
    monitor_loss_weight : float = 1.0
    use_search_width_sched : bool = False

class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = [] # elements are (loss: float, optim_ids: Tensor)
        self.size = size
        self.n_repeat = 0 #number of times the same best_ids is returned 
        self.prev_best_ids = None

    def get_best_ids(self) -> Tensor:
        if self.prev_best_ids is not None and torch.equal(self.prev_best_ids, self.buffer[0][1]):
            self.n_repeat += 1
        else:
            self.n_repeat = 0
            self.prev_best_ids = self.buffer[0][1]
            
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
            return

        self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])
    
    def sample(self) -> Tensor:
        idx = torch.randint(high = self.size, size = (1,)).item()
        return self.buffer[idx][1]


def clear_gpus():
    gc.collect()
    torch.cuda.empty_cache()
    
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
    
    if len(filtered_ids) == 0:
        return []
    else:
        return torch.stack(filtered_ids)

def compute_candidates_loss(
    search_batch_size: int, 
    model: transformers.PreTrainedModel,
    kv_cache: tuple,
    input_embeds: Tensor, 
    target_ids: Tensor,
    gcg_weight: float,
    monitor_weight: float,
    monitor : Union[ActMonitor, TextMonitor] = None,
    sampled_ids: Tensor = None,
    config : GCGConfig = None,
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
            sampled_ids_batch = sampled_ids[i:i+search_batch_size]
            current_batch_size = input_embeds_batch.shape[0]

            kv_cache_batch = []
            for i in range(len(kv_cache)):
                kv_cache_batch.append([])
                for j in range(len(kv_cache[i])):
                    kv_cache_batch[i].append(kv_cache[i][j].expand(current_batch_size, -1, -1, -1))

            outputs = model(inputs_embeds=input_embeds_batch, past_key_values=kv_cache_batch, output_hidden_states = True if monitor is not None else False)
            logits = outputs.logits

            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., shift-1:-1, :].contiguous()
            shift_labels = target_ids.repeat(current_batch_size, 1)

            if monitor is not None:
                if isinstance(monitor, ActMonitor):
                    monitor_input = slice_acts(outputs, 
                            N_TOKS = 0, 
                            layers = monitor.layer,
                            tok_idxs = torch.tensor(monitor.tok_idxs) - target_ids.shape[1],
                            return_prompt_acts = False,
                            device = model.device)
                elif isinstance(monitor, TextMonitor):
                    monitor_input = torch.cat([sampled_ids_batch, monitor.after_ids.repeat(search_batch_size, 1)], dim=1)
            
                monitor_loss = monitor.get_loss_no_grad(monitor_input)
                del monitor_input
                clear_gpus()
            else:
                monitor_loss = torch.tensor(0)
                
            gcg_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none") 
            gcg_loss = gcg_loss.view(current_batch_size, -1).mean(dim=-1)
            monitor_loss =  monitor_loss
            
            loss = gcg_weight * gcg_loss + monitor_weight * monitor_loss
            all_loss.append(loss)

            del outputs
            clear_gpus()

    return torch.cat(all_loss, dim=0)

def gen_init_buffer_ids(mw : ModelWrapper, num_toks: int, num_seqs: int):
    if num_seqs == 0:
        return []
    punc_tok_ids = torch.tensor(mw.tokenizer([".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}"])['input_ids']).to(mw.model.device, dtype = torch.float32)[:, 1]
    init_buffer_ids = [punc_tok_ids[torch.multinomial(punc_tok_ids, num_toks, replacement = True)].unsqueeze(0).long() for _ in range(num_seqs)]
    
    return init_buffer_ids

def step_loss_weights(gcg_weight, monitor_weight, total_steps):
    size = 1/total_steps
    if np.abs(gcg_weight - monitor_weight) < size:
        gcg_weight, monitor_weight = 0.5, 0.5
    if gcg_weight > monitor_weight:
        gcg_weight -= size
        monitor_weight += size
    elif gcg_weight < monitor_weight:
        gcg_weight += size
        monitor_weight -= size
    pre_softmax_gcg_weight, pre_softmax_monitor_weight = gcg_weight, monitor_weight
    comb = torch.tensor([gcg_weight, monitor_weight])
    comb = torch.nn.functional.softmax(comb / 0.1, dim=0)
    gcg_weight = comb[0].item()
    monitor_weight = comb[1].item()
    return gcg_weight, monitor_weight, pre_softmax_gcg_weight, pre_softmax_monitor_weight

def step_loss_weights(gcg_weight, monitor_weight, total_steps):
    return gcg_weight, monitor_weight, gcg_weight, monitor_weight
    
def run(
    mw : ModelWrapper,
    messages: Union[str, List[dict]],
    target: str,
    monitor : Union[ActMonitor, TextMonitor] = None,
    config: GCGConfig = None, 
) -> dict:
    """Generates a single optimized string using GCG. 

    Args:
        mw: ModelWrapper
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.
        monitor : A monitor that GCG is trying to jailbreak as well
    
    Returns:
        A dict that stores losses and the final optim_str
    """
    def n_replace_sched(step: int) -> int:
        n_replace = int((1 - (step / 300)) * config.n_replace)
        if n_replace < 1:
            return 1
        return n_replace
    def search_width_sched(n_repeat : int) -> int:
        """as n_repeat goes from 1 to 10, search_width goes from search_width to 4 * search_width"""
        if config.use_search_width_sched:
            if n_repeat > 10:
                mult = 10
            else:
                mult = n_repeat // 2 + 1
            return config.search_width * mult
        else:
            return config.search_width
        
    if config == None:
        config = GCGConfig()
    
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    if not any(["{optim_str}" in d["content"] for d in messages]):
        messages[-1]["content"] = messages[-1]["content"] + " {optim_str}"

    template = mw.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
    # Remove the BOS token -- this will get added when tokenizing, if necessary
    if mw.tokenizer.bos_token and template.startswith(mw.tokenizer.bos_token):
        template = template.replace(mw.tokenizer.bos_token, "")
    before_str, after_str = template.split("{optim_str}")
    target = " " + target if config.add_space_before_target else target
    print(f"before, after, target : {before_str} | {after_str} | {target}")
        
    # Tokenize everything that doesn't get optimized
    before_ids = mw.tokenizer([before_str], padding=False)["input_ids"]
    after_ids = mw.tokenizer([after_str], add_special_tokens=False)["input_ids"]
    target_ids = mw.tokenizer([target], add_special_tokens=False)["input_ids"]
    before_ids, after_ids, target_ids = [torch.tensor(ids, device=mw.model.device) for ids in (before_ids, after_ids, target_ids)]

    # Embed everything that doesn't get optimized
    embedding_layer = mw.model.get_input_embeddings()
    before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]
    
    # Compute the KV Cache for tokens that appear before the optimized tokens
    with torch.no_grad():
        output = mw.model(inputs_embeds=before_embeds, use_cache=True)
        kv_cache = output.past_key_values

        if monitor is not None and isinstance(monitor, TextMonitor): 
            monitor.set_kv_cache(messages[0]["content"])
            
            monitor.after_embeds = monitor.model.get_input_embeddings()(monitor.after_ids)

    optim_ids = mw.tokenizer(config.optim_str_init, return_tensors="pt", add_special_tokens=False)["input_ids"].to(mw.model.device)
    
    buffer = AttackBuffer(config.buffer_size)
    buffer_ids = torch.stack([optim_ids] + gen_init_buffer_ids(mw, optim_ids.shape[1], config.buffer_size - 1)).squeeze(1)
    input_embeds = torch.cat([
            embedding_layer(buffer_ids),
            after_embeds.repeat(buffer_ids.shape[0], 1, 1),
            target_embeds.repeat(buffer_ids.shape[0], 1, 1)
        ], dim=1)    
        
    gcg_weight, monitor_weight = config.gcg_loss_weight, config.monitor_loss_weight
    pre_softmax_gcg_weight, pre_softmax_monitor_weight = gcg_weight, monitor_weight
    
    loss = find_executable_batch_size(compute_candidates_loss, buffer_ids.shape[0])(
            mw.model,
            kv_cache,
            input_embeds,
            target_ids,
            gcg_weight, 
            monitor_weight,
            monitor, 
            sampled_ids = buffer_ids,
            config = config
        )
    for i in range(config.buffer_size):
        buffer.add(loss=loss[i].item(), optim_ids=buffer_ids[i].unsqueeze(0))
    
    losses = []
    monitor_losses = []
    gcg_losses = []
    early_stopping_condition = []
    optim_strings = []
    for i in range(config.num_steps):
        optim_ids = buffer.get_best_ids()
        
        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(dtype=mw.model.dtype, device=mw.model.device)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight
        input_embeds = torch.cat([optim_embeds, after_embeds, target_embeds], dim=1) #*
        output = mw.model(inputs_embeds=input_embeds, past_key_values=kv_cache, output_hidden_states = True if monitor is not None else False)
        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[..., shift-1:-1, :].contiguous() # (1, num_target_ids, vocab_size)
        shift_labels = target_ids
        
        if monitor is not None:
            if isinstance(monitor, ActMonitor):
                monitor_input = slice_acts(output, 
                            N_TOKS = 0, 
                            layers = monitor.layer,
                            tok_idxs = torch.tensor(monitor.tok_idxs) - target_ids.shape[1],
                            return_prompt_acts = False,
                            device = mw.model.device)
                monitor_loss = monitor.get_loss(monitor_input)
                del monitor_input
            elif isinstance(monitor, TextMonitor):
                monitor_input_embeds = torch.cat([optim_embeds, monitor.after_embeds], dim=1)
                monitor_loss = monitor.get_loss(monitor_input_embeds)
            
            print(monitor_loss)
            del output
            clear_gpus()
        else:
            monitor_loss = torch.tensor(0)
            del output 
            clear_gpus()
        
        gcg_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        model_preds = torch.argmax(shift_logits, dim=-1)
        gcg_weight, monitor_weight, pre_softmax_gcg_weight, pre_softmax_monitor_weight = step_loss_weights(pre_softmax_gcg_weight, pre_softmax_monitor_weight, config.num_steps)

        print(f'Pre softmax GCG weight: {pre_softmax_gcg_weight}, Pre softmax Monitor weight: {pre_softmax_monitor_weight}')
        print(f'GCG weight: {gcg_weight}, Monitor weight: {monitor_weight}')
        loss = gcg_weight * gcg_loss + monitor_weight * monitor_loss
        
        print(f"model preds : {mw.tokenizer.decode(model_preds.squeeze(0))}")
        print(f"monitor_loss : {monitor_loss.item()} | gcg_loss : {gcg_loss} |  loss : {loss.item()} | search_width : {search_width_sched(buffer.n_repeat)} | n_replace : {n_replace_sched(i)}")
        
        losses.append(loss.item())
        monitor_losses.append(monitor_loss.item())
        gcg_losses.append(gcg_loss.item())
        
        gcg_condition = model_preds.eq(shift_labels).all() 
        monitor_condition = (monitor_loss < 0.5) 
        
        if gcg_condition and monitor_condition:
            print("Early Stopping Condition Met")
            early_stopping_condition.append(1)
        else:
            early_stopping_condition.append(0)
  
        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        # Sample candidate token sequences
        sampled_ids = sample_ids_from_grad(
            optim_ids.squeeze(0),
            optim_ids_onehot_grad.squeeze(0),
            search_width_sched(buffer.n_repeat),
            config.topk,
            n_replace_sched(i)
        )
        
        sampled_ids = filter_ids(sampled_ids, mw.tokenizer)
        if len(sampled_ids) == 0:
            print("No good sampled ids")
            if len(optim_strings) == 0:
                print(f"step: {i} | optim_str: None found yet | loss: {buffer.get_lowest_loss()}")
            else:
                print(f"step: {i} | optim_str: {optim_strings[-1]} | loss: {buffer.get_lowest_loss()}")

            continue 
        
        new_search_width = sampled_ids.shape[0]

        input_embeds = torch.cat([
            embedding_layer(sampled_ids),
            after_embeds.repeat(new_search_width, 1, 1),
            target_embeds.repeat(new_search_width, 1, 1)
        ], dim=1)

        # Compute loss on all candidate sequences 
        loss = find_executable_batch_size(compute_candidates_loss, new_search_width)(
            mw.model,
            kv_cache,
            input_embeds,
            target_ids,
            gcg_weight, 
            monitor_weight, 
            monitor,
            config = config,
            sampled_ids = sampled_ids,
        )

        current_loss = loss.min().item()
                
        # # get the buffer_size best optim_ids, sorted from lowest loss to highest loss
        # buffer_idxs = loss.argsort()[:config.buffer_size]
        # # for each optim_ids, add it to buffer if its loss is less than highest loss, if not, then break from the loop
        # for idx_buffer in buffer_idxs:
        #     if loss[idx_buffer] < buffer.get_highest_loss():                
        #         buffer.add(loss[idx_buffer].item(), sampled_ids[idx_buffer].unsqueeze(0))
        #     else:
                # break
        
        optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
        if current_loss < buffer.get_highest_loss():
            buffer.add(current_loss, optim_ids)

        optim_str = mw.tokenizer.batch_decode(buffer.get_best_ids())        
        optim_strings.append(optim_str)

        print(f"step: {i} | optim_str: {optim_str} | lowest loss after sample_id: {buffer.get_lowest_loss()}")
    
    return {
        "losses": losses,
        "monitor_losses": monitor_losses,
        "gcg_losses" : gcg_losses, 
        "optim_strings": optim_strings,
        "early_stopping" : early_stopping_condition,
    }
            