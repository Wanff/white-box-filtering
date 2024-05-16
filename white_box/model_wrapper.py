#from: https://github.com/andyzoujm/representation-engineering/blob/main/repe/rep_control_reading_vec.py
# wrapping classes
import numpy as np
from typing import List, Union, Optional
import functools
import time 
from tqdm import tqdm 
from collections import defaultdict

import torch
from einops import rearrange
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
# from accelerate import find_executable_batch_size

from white_box.utils import untuple

import functools
import gc
import inspect
import torch

# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                # return function(batch_size + 1, *args, **kwargs)
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                else:
                    raise

    return decorator

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.normalize = False

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output
            
        if self.controller is not None:
        
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)

            if self.mask is not None:
                mask = self.mask

            # we should ignore the padding tokens when doing the activation addition
            # mask has ones for non padding tokens and zeros at padding tokens.
            # only tested this on left padding
            elif "position_ids" in kwargs:
                pos = kwargs["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                target_shape = modified.shape
                mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)
                mask = mask.to(modified.dtype)
            else:
                # print(f"Warning: block {self.block_name} does not contain information 'position_ids' about token types. When using batches this can lead to unexpected results.")
                mask = 1.0

            if len(self.controller.shape) == 1:
                self.controller = self.controller.reshape(1, 1, -1)
            
            self.controller = self.controller.to(modified.device)
            if type(mask) == torch.Tensor:
                mask = mask.to(modified.device)
            if isinstance(self.token_pos, int):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple) or isinstance(self.token_pos, np.ndarray):
                if self.controller.shape[0] > 1 and modified.shape[1] == 1:
                    #if controller is multiple tokens and modified is one, meaning we are in autoregressive generation mode, we skip out of this loop
                    pass
                else:
                    modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, str):
                if self.token_pos == "end":
                    len_token = self.controller.shape[1]
                    modified[:, -len_token:] = self.operator(modified[:, -len_token:], self.controller * mask)
                elif self.token_pos == "start":
                    len_token = self.controller.shape[1]
                    modified[:, :len_token] = self.operator(modified[:, :len_token], self.controller * mask)
                else:
                    assert False, f"Unknown token position {self.token_pos}."
            else:

                assert len(self.controller.shape) == len(modified.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."
                modified = self.operator(modified, self.controller * mask)

            if self.normalize:
                norm_post = torch.norm(modified, dim=-1, keepdim=True)
                modified = modified / norm_post * norm_pre
            
        if isinstance(output, tuple):
            output = (modified,) + output[1:] 
        else:
            output = modified
        
        return output

    def set_controller(self, activations, token_pos=None, masks=None, normalize=False, operator='linear_comb'):
        self.normalize = normalize
        self.controller = activations.squeeze()
        self.mask = masks
        self.token_pos = token_pos
        if operator == 'linear_comb':
            def op(current, controller):
                return current + controller
        elif operator == 'piecewise_linear':
            def op(current, controller):
                sign = torch.sign((current * controller).sum(-1, keepdim=True))
                return current + controller * sign
        elif operator == 'projection':
            def op(current, controller):
                # print(current.shape, controller.shape)
                projection = torch.sum(current.float() * controller.float(), dim = 2).unsqueeze(2) * controller.float()
                if current.dtype == torch.float16:
                    projection = projection.half()
                return current - projection
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")
        self.operator = op
        
    def reset(self):
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.operator = None

    def set_masks(self, masks):
        self.mask = masks


#! attention always comes first, mlp comes second bc that's just how it is
LLAMA_BLOCK_NAMES = [
    "self_attn",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj"
    ]

PYTHIA_BLOCK_NAMES = [
    "attention",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm",
    "attention.query_key_value",
]

GPT_BLOCK_NAMES = [
    'attn',
    'mlp',
    'ln_1',
    'ln_2'
]

def detokenize_to_list(tokenizer, input_ids):
    return [[tokenizer.decode(input_ids[i][j]) for j in range(len(input_ids[0]))] for i in range(len(input_ids))]

def slice_acts(out, N_TOKS: int, return_prompt_acts: bool, layers: List, tok_idxs: List = None, device: str = 'cpu'):
    """slices acts out of huggingface modeloutput object, have to + 1 layers

    Args:
        out (_type_): _description_
        N_TOKS (int): how many tokens generated
        return_prompt_toks (bool): _description_
        layers (List): _description_

    Returns:
     shape: b l t d
    """
    #! expects hf layer idxs (ie 1-index)
    #out.hidden_states is shape  max_new_tokens x n_layers + 1 x batch x activations

    if N_TOKS == 0:
        acts = torch.stack(out.hidden_states, dim = 1) #this is when you just call model(), not generate
    elif N_TOKS == 1:
        # acts = torch.stack([torch.cat(out.hidden_states[0], dim = 1)], dim = 1)  #1, N_TOKS bc the first index is all previous tokens
        acts = torch.stack(out.hidden_states[0], dim = 0) #shape: n_layers + 1 x batch_size x seq_len x d_M
        acts = rearrange(acts, 'l b t d -> b l t d')
        return_prompt_acts = False
    else:
        #first loop goes through the tokens, second loop goes through the layers or something
        acts = torch.stack([torch.cat(out.hidden_states[i], dim = 1) for i in range(1, N_TOKS)], dim = 1)  #1, N_TOKS bc the first index is all previous tokens
        print(acts.shape)
        acts = rearrange(acts, 'b t l d -> b l t d')

    #shape: batch_size x N_TOKS - 1 x n_layers + 1 x d_M
    #n_layers + 1 bc of embedding, N_TOKS - 1 bc of how max_new_tokens works

    if return_prompt_acts:
        prompt_acts = torch.stack(out.hidden_states[0], dim = 0) #shape: n_layers + 1 x batch_size x seq_len x d_M
        prompt_acts = rearrange(prompt_acts, 'l b t d -> b l t d')
        print(prompt_acts.shape)
        acts = torch.cat([prompt_acts, acts], dim = 2)
    
    if device == 'cpu':
        acts = acts.cpu()
        
    if tok_idxs is not None:
        acts = acts[:, :, tok_idxs]
    acts = acts[:, torch.tensor(layers) + 1]
    return acts

def rename_attribute(object_, old_attribute_name, new_attribute_name):
    setattr(object_, new_attribute_name, getattr(object_, old_attribute_name))
    delattr(object_, old_attribute_name)

def ceildiv(a, b):
    #https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)

def blockify_block_name(block_name):
    if "." in block_name:
        sepped_blocks = block_name.split(".")
        for i in range(len(sepped_blocks) - 1):
            sepped_blocks.insert(2*i + 1, 'block')
        block_name = ".".join(sepped_blocks)
    
    return block_name
            
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer, template = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.template = template
        
        self.model_base = self.model #I do this for wrapping purposes
        if hasattr(self.model, 'model'):
            self.block_names = [blockify_block_name(name) for name in LLAMA_BLOCK_NAMES]
            
            self.model_base = self.model.model
            self.num_layers = self.model.config.num_hidden_layers
            self.universal_b_name_map = {
                "attn": self.block_names[0],
                "mlp": self.block_names[1],
                "ln_1": self.block_names[2],
                "ln_2": self.block_names[3],
                "attn_q": self.block_names[4],
                "attn_k": self.block_names[5],
                "attn_v": self.block_names[6],
            }   
            
        elif hasattr(self.model, 'gpt_neox'):
            self.block_names = [blockify_block_name(name) for name in PYTHIA_BLOCK_NAMES]
            
            self.model_base = self.model.gpt_neox
            self.num_layers = self.model.config.num_hidden_layers
            
            self.universal_b_name_map = {
                "attn": self.block_names[0],
                "mlp": self.block_names[1],
                "ln_1": self.block_names[2],
                "ln_2": self.block_names[3],
                "attn_q": self.block_names[4],
                "attn_k": self.block_names[4],
                "attn_v": self.block_names[4],
            }   
        
        elif hasattr(self.model, 'transformer'):
            self.block_names = GPT_BLOCK_NAMES
            self.model_base = self.model.transformer
            self.model_base.layers = self.model.transformer.h
        
    #Generation Functions
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def generate(self, prompts, **kwargs):
        input_ids, attention_mask = self.process_prompts(prompts)
        return self.model.generate(input_ids = input_ids, attention_mask = attention_mask, **kwargs)

    def process_prompts(self, prompts : Union[List[str], List[int]], use_chat_template : bool = True):
        if isinstance(prompts[0], str):
            if self.template is not None and use_chat_template:
                prompts = [self.template.format(instruction=s) for s in prompts]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, max_length=2048, truncation=True)
        else:
            inputs = self.tokenizer.pad({'input_ids': prompts}, padding = True, return_attention_mask=True)
        
        if isinstance(inputs.input_ids, list): 
            input_ids = torch.tensor(inputs.input_ids).to(self.model.device)
            attention_mask = torch.tensor(inputs.attention_mask).to(self.model.device)
        else:
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
        
        return input_ids, attention_mask
                
    def batch_hiddens(self, 
                      prompts: Union[List[str], List[int]], 
                      layers: List[int], 
                      tok_idxs: Union[List[int], int] = -1, 
                      return_types = ['resid'], 
                      logging : bool = False, 
                      use_chat_template : bool = True,
                      **kwargs):
        """
        Takes a list of strings or tokens and returns the hidden states of the specified layers and token indices.
        
        does NOT support autoregressive generation
        """
        if return_types != ['resid']:
            self.wrap_all()
            self.reset()
        
        if isinstance(layers, int):
            layers = [layers]

        def _inner_loop(batch_size):
            nonlocal prompts, layers, tok_idxs, return_types, logging, kwargs
            hidden_states_dict = defaultdict(list)
            tot_prompts = 0
            
            for i in tqdm(range(0, len(prompts), batch_size)):
                start_time = time.time()
                tot_prompts += batch_size
                batched_prompts = prompts[i:i+batch_size]
            
                input_ids, attention_mask = self.process_prompts(batched_prompts, use_chat_template = use_chat_template)
                outputs = self.model(input_ids = input_ids, attention_mask=attention_mask, output_hidden_states = True, **kwargs)
                
                #if padding_side==left, tok_idxs is neg, one dimensional, and relative to the end of the sequence
                #if padding_side==right, tok_idxs is pos, two dimensional (batch x tok_pos), and relative to the start of the sequence
                if self.tokenizer.padding_side == "right":
                    #* we're assuming that tok_idxs is negative
                    batch_last_tok_idxs = (torch.sum(attention_mask, dim = 1, keepdim = True) - 1).to("cpu")
                    batch_tok_idxs = batch_last_tok_idxs.repeat(1, len(tok_idxs)) + torch.tensor(tok_idxs).repeat(len(batched_prompts), 1) + 1
                else:
                    batch_tok_idxs = tok_idxs
                    
                for act_type in return_types:
                    if act_type == 'resid':
                        hidden_states_dict['resid'].append(self._get_hidden_states(outputs, batch_tok_idxs, layers, 'residual'))
                    else:
                        assert act_type in self.universal_b_name_map.keys(), f"Unknown activation type {act_type}." 
                        acts_dict = self.get_activations(layers, self.universal_b_name_map[act_type])
                        for layer in layers:
                            if self.tokenizer.padding_side == "left":
                                acts_dict[layer] = acts_dict[layer][:, batch_tok_idxs, :]
                            else:
                                acts_dict[layer] = acts_dict[layer][torch.arange(acts_dict[layer].shape[0]), batch_tok_idxs, :] #untested
                        
                        hidden_states_dict[act_type].append(torch.stack([acts_dict[layer] for layer in acts_dict.keys()]))
                
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
                
            for act_type in hidden_states_dict:
                hidden_states_dict[act_type] = torch.cat(hidden_states_dict[act_type], dim = 1).transpose(0, 1)
            return hidden_states_dict
        
        @find_executable_batch_size(starting_batch_size=len(prompts))
        def inner_loop(batch_size):
            return _inner_loop(batch_size)
        try:
            return inner_loop()
        except:
            return _inner_loop(1)
                               
    def batch_generate_autoreg(self, prompts, 
                                   max_new_tokens: int = 32,
                                   output_hidden_states = False, 
                                   output_tokens = False, 
                                   layers = None,
                                   tok_idxs = None,
                                   return_prompt_acts = False,
                                   **kwargs):
        if isinstance(prompts[0], str):
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, max_length=512, truncation=True)
        else:
            inputs = self.tokenizer.pad({'input_ids': prompts}, padding = True, return_attention_mask=True)
            
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id, # TODO: BAD this is hardcoded, different for eg dutch model
                    max_new_tokens = max_new_tokens,
                    min_new_tokens = max_new_tokens,
                    output_hidden_states = output_hidden_states,
                    return_dict_in_generate = output_hidden_states or output_tokens,
                    **kwargs
                    )
        
        #! this is not good code:
        return_dict = {}
        if output_hidden_states:
            assert layers is not None and tok_idxs is not None, "Must specify layers and token indices to slice."
            return_dict['hidden_states']  = slice_acts(out, 
                                                N_TOKS = max_new_tokens, 
                                                layers = layers,
                                                tok_idxs = tok_idxs,
                                                return_prompt_acts = return_prompt_acts)
        if output_tokens:
            return_dict['tokens'] = out['sequences']
        
        if output_hidden_states or output_tokens: 
            return_dict['generations'] = self.tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)

            return return_dict
        else:
            return self.tokenizer.batch_decode(out, skip_special_tokens=True)            
        
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.model.device)).logits
            return logits
        
    def run_prompt(self, prompt, **kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            return output
        
    def _get_hidden_states(
            self, 
            outputs,
            tok_idxs: Union[List[int], int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']
    
        hidden_states_layers = {}
        for layer in hidden_layers:
            layer = layer + 1 #we do this because usually we 0-index layers, but hf does not bc it includes the embedding
            hidden_states = outputs['hidden_states'][layer] #tuple where layer is tuple
            
            if self.tokenizer.padding_side == "left":
                hidden_states =  hidden_states[:, tok_idxs, :]
            elif self.tokenizer.padding_side == "right":
                # print(torch.arange(hidden_states.shape[0]))
                # print(tok_idxs)
                hidden_states =  hidden_states[torch.arange(hidden_states.shape[0]), tok_idxs, :]
                
            # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
            hidden_states_layers[layer - 1] = hidden_states.detach().cpu().to(dtype = torch.float32)
        
        return torch.stack([hidden_states_layers[layer] for layer in hidden_states_layers.keys()])
    
    def query_tok_dist(self, prompt, TOP_K = 10):
        """
        Gets top 10 predictions after last token in a prompt
        """
        tokens = self.tokenizer.encode_plus(prompt, return_tensors = 'pt').to(self.model.device)
        
        output = self.model(**tokens)

        logits = output['logits']
        
        trg_tok_idx = tokens['input_ids'].shape[1] - 1
        
        #gets probs after last tok in seq
        probs = F.softmax(untuple(logits)[0][trg_tok_idx], dim=-1) #the [0] is to index out of the batch idx
 
        probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

        #assert probs add to 1
        assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs)-1)) 

        probs_ = []
        for index, prob in enumerate(probs):
            probs_.append((index, prob))

        top_k = sorted(probs_, key = lambda x: x[1], reverse = True)[:TOP_K]
        top_k = [(t[1].item(), self.tokenizer.decode(t[0])) for t in top_k]
        
        return top_k
        
    def rej_sampl_generate(self, prompts, 
                                probe: Union[torch.nn.Module, LogisticRegression],
                                probe_layer: int,
                                max_new_tokens: int = 32, 
                                rej_sample_length: int = 5,
                                log_rej_samples = False,
                                max_tries = 10,
                                **generation_kwargs
    ):
        """
        Generates a sequence and then rejects or accepts it based on the probe.
        
        probe_layer: the layer to probe, if None, probes the last layer
        probe: a probe to use, if None, uses the default probe
        """
        input_ids, attention_mask = self.process_prompts(prompts)
        
        for i in range(ceildiv(max_new_tokens, rej_sample_length)):
            if log_rej_samples:
                print(f"part {i} of generation")
                
            out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens = rej_sample_length,
                    return_dict_in_generate = True,
                    output_hidden_states = True,
                    **generation_kwargs,
                    )
            
            # print(out.hidden_states[1][0].shape)
            #shape batch_size x d_m
            hidden_states =  slice_acts(out, 
                                        N_TOKS = rej_sample_length, 
                                        layers = probe_layer+1, #bc slice_acts expects 1-indexed layers
                                        tok_idxs = -1,
                                        return_prompt_acts = False,
                                        device = "cuda").float()
            gens = out.sequences
            preds = probe.predict(hidden_states).detach().cpu().numpy()
            
            flagged_gen_idxs = untuple(np.where(preds == 1))
            good_gen_idxs = untuple(np.where(preds == 0))
            pot_banned_words = out.sequences[:, input_ids.shape[1]:input_ids.shape[1] + rej_sample_length]
            
            if log_rej_samples:
                print(f"Genned (Banned) words {pot_banned_words}")
                print(f"Genned (Banned) words {detokenize_to_list(self.tokenizer, pot_banned_words)}")
                print(f"Preds {preds}")
            
            num_tries = 0
            while 1 in preds and num_tries < max_tries:
                preds = []
                for i in flagged_gen_idxs:
                    temp_input_ids = input_ids[i]
                    temp_attention_mask = attention_mask[i]

                    for banned_tok_idx in range(rej_sample_length):
                        # print(self.tokenizer.decode(temp_input_ids))
                        # print(pot_banned_words[i, banned_tok_idx])
                        out = self.model.generate(
                                input_ids=temp_input_ids.unsqueeze(dim = 0),
                                attention_mask=temp_attention_mask.unsqueeze(dim = 0),
                                pad_token_id=self.tokenizer.eos_token_id,
                                max_new_tokens = 1,
                                return_dict_in_generate = True,
                                output_hidden_states = True,
                        
                                bad_words_ids = [[pot_banned_words[i, banned_tok_idx].item()]],
                                **generation_kwargs,
                                )
                        temp_input_ids = out.sequences[0]
                        temp_attention_mask = torch.cat([temp_attention_mask, torch.tensor([1], device = attention_mask.device)], dim = 0)
                                        
                    hidden_states = slice_acts(out, 
                                            N_TOKS = 1, 
                                            layers = probe_layer+1, #bc slice_acts expects 1-indexed layers
                                            tok_idxs = -1,
                                            return_prompt_acts = False, 
                                            device = "cuda").float()
                    gens[i] = out.sequences[0]
                    preds.append(probe.predict(hidden_states).detach().cpu().numpy().item())
                
                num_tries += 1
                if log_rej_samples:
                    print(f"New preds {preds}")
                    for i in flagged_gen_idxs:
                        print(self.tokenizer.decode(gens[i], skip_special_tokens=True))
                        print("--------------")
                    # print(self.tokenizer.batch_decode(gens, skip_special_tokens=True))
                    print()
            
            gen_len = input_ids.shape[1]
            input_ids = gens
            attention_mask = torch.cat([attention_mask, torch.ones(input_ids[:, gen_len:].shape, device = attention_mask.device)], dim = 1)

            if log_rej_samples:
                print(f"took {num_tries} times to get final generations:")
                # print(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True))
                print()

        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    
    #Wrapping Logic
    def wrap(self, layer_id, block_name):
        assert block_name in self.block_names
        
        if self.is_wrapped(self.model_base.layers[layer_id]):
            block = rgetattr(self.model_base.layers[layer_id].block, block_name)
            if not self.is_wrapped(block):
                rsetattr(self.model_base.layers[layer_id].block, block_name, WrappedBlock(block))
        else:
            block = rgetattr(self.model_base.layers[layer_id], block_name)
            if not self.is_wrapped(block):
                rsetattr(self.model_base.layers[layer_id], block_name, WrappedBlock(block))

    def wrap_decoder_block(self, layer_id):
        block = self.model_base.layers[layer_id]
        if not self.is_wrapped(block):
            self.model_base.layers[layer_id] = WrappedBlock(block)

    def wrap_all(self):
        for layer_id, layer in enumerate(self.model_base.layers):
            for block_name in self.block_names:
                self.wrap(layer_id, block_name)
            self.wrap_decoder_block(layer_id)
    
    def wrap_block(self, layer_ids, block_name):
        def _wrap_block(layer_id, block_name):
            if block_name in self.block_names:
                self.wrap(layer_id, block_name)
            elif block_name == 'decoder_block':
                self.wrap_decoder_block(layer_id)
            else:
                assert False, f"No block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _wrap_block(layer_id, block_name)
        else:
            _wrap_block(layer_ids, block_name)
        
    def reset(self):
        for layer in self.model_base.layers:
            if self.is_wrapped(layer):
                layer.reset()
                for block_name in self.block_names:
                    if self.is_wrapped(rgetattr(layer.block, block_name)):
                        rgetattr(layer.block, block_name).reset()
            else:
                for block_name in self.block_names:
                    if self.is_wrapped(rgetattr(layer, block_name)):
                        rgetattr(layer, block_name).reset()

    def set_masks(self, masks):
        for layer in self.model_base.layers:
            if self.is_wrapped(layer):
                layer.set_masks(masks)
                for block_name in self.block_names:
                    if self.is_wrapped(rgetattr(layer.block, block_name)):
                        rgetattr(layer.block, block_name).set_masks(masks)
            else:
                for block_name in self.block_names:
                    if self.is_wrapped(rgetattr(layer, block_name)):
                        rgetattr(layer, block_name).set_masks(masks)

    def is_wrapped(self, block):
        if hasattr(block, 'block'):
            return True
        return False
    
    def unwrap(self):
        for l, layer in enumerate(self.model_base.layers):
            if self.is_wrapped(layer):
                self.model_base.layers[l] = layer.block
            for block_name in self.block_names:
                if self.is_wrapped(rgetattr(self.model_base.layers[l], block_name)):
                    rsetattr(self.model_base.layers[l],
                            block_name,
                            rgetattr(self.model_base.layers[l], block_name).block)

    #Activation Storing and Interventions
    def get_activations(self, layer_ids, block_name='decoder_block'):

        def _get_activations(layer_id, block_name):
            current_layer = self.model_base.layers[layer_id]

            if self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name == 'decoder_block':
                    return current_layer.output.detach().cpu()
                elif block_name in self.block_names and self.is_wrapped(rgetattr(current_block, block_name)):
                    return rgetattr(current_block, block_name).output.detach().cpu()
                else:
                    assert False, f"No wrapped block named {block_name}."

            else:
                if block_name in self.block_names and self.is_wrapped(rgetattr(current_layer, block_name)):
                    return rgetattr(current_layer, block_name).output.detach().cpu()
                else:
                    assert False, f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            activations = {}
            for layer_id in layer_ids:
                activations[layer_id] = _get_activations(layer_id, block_name)
            return activations
        else:
            return _get_activations(layer_ids, block_name)

    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=None, masks=None, normalize=False, operator='linear_comb'):

        def _set_controller(layer_id, activations, block_name, masks, normalize, operator):
            current_layer = self.model_base.layers[layer_id]

            if block_name == 'decoder_block':
                current_layer.set_controller(activations, token_pos, masks, normalize, operator)
            elif self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name in self.block_names and self.is_wrapped(rgetattr(current_block, block_name)):
                    rgetattr(current_block, block_name).set_controller(activations, token_pos, masks, normalize, operator)
                else:
                    return f"No wrapped block named {block_name}."

            else:
                if block_name in self.block_names and self.is_wrapped(rgetattr(current_layer, block_name)):
                    rgetattr(current_layer, block_name).set_controller(activations, token_pos, masks, normalize, operator)
                else:
                    return f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            assert isinstance(activations, dict), "activations should be a dictionary"
            for layer_id in layer_ids:
                _set_controller(layer_id, activations[layer_id], block_name, masks, normalize, operator)
        else:
            _set_controller(layer_ids, activations, block_name, masks, normalize, operator)
      