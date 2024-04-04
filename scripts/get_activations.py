import transformers
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Union, Callable
from collections import defaultdict

from tqdm import tqdm
import argparse
import gc
import pickle

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_from_disk, load_dataset

import sys
sys.path.append('../')  # Add the parent directory to the path

from white_box.utils import char_by_char_similarity, tok_by_tok_similarity, levenshtein_distance, ceildiv, gen_pile_data
from white_box.model_wrapper import ModelWrapper

def toks_to_string(tokenizer, toks):
    return "".join(tokenizer.batch_decode(toks))

def get_memmed_activations_from_pregenned(mw: ModelWrapper, prompts: Union[List[str],List[int]],
                                          save_path: str,
                                          act_types: List[str] = ['resid'],
                                          save_every: int = 100,
                                          layers: List[int] = None,
                                           tok_idxs: List[int] = None,
                                           logging: bool = False,
                                           file_spec: str = "",
                                           **generation_kwargs,
                                          ):
    """gets activations from pregenerated prompts, which enables access to attention/mlp

    Args:
        mw (ModelWrapper): _description_
        prompts (Union[List[str],List[int]]): _description_
        save_path (str): _description_
        act_types (List[str], optional): _description_. Defaults to ['resid'].
        save_every (int, optional): _description_. Defaults to 100.
        layers (List[int], optional): _description_. Defaults to None.
        tok_idxs (List[int], optional): _description_. Defaults to None.
        logging (bool, optional): _description_. Defaults to False.
        file_spec (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if layers is None:
        layers = list(range(mw.model.config.num_hidden_layers))
        
    acts_dict = {}
    for act_type in act_types:
        acts_dict[act_type] = dict([(layer, []) for layer in layers])
            
    for batch_idx in range(ceildiv(len(prompts), save_every) ):
        if logging:
            start_time = time.time()
            print(f"Batch {batch_idx + 1} of {len(prompts) // save_every}")
            
        batch = prompts[batch_idx * save_every : (batch_idx + 1) * save_every]
                
        batch_act_dict = mw.batch_hiddens(
                    batch,
                    layers = layers,
                    tok_idxs = tok_idxs,
                    return_types = act_types,
                    **generation_kwargs,
                    )

        for act_type in act_types:
            for layer in layers:
                acts_dict[act_type][layer].append(batch_act_dict[act_type][layer])
            
        if logging:
            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time
            print(f"Time elapsed for this batch: {elapsed_time:.2f} seconds")
            print(f"Num Hidden States Genned: {len(batch)}")
            for act_type in act_types:
                print(f"Shape of {act_type} one batch Hidden States: {acts_dict[act_type][layers[0]][-1].shape}")
            print()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Save all_hidden_states
        if batch_idx == int(.25 * ceildiv(len(prompts), save_every)) or batch_idx == int(.5 * ceildiv(len(prompts), save_every)) or batch_idx == int(.75 * ceildiv(len(prompts), save_every)):
            print("Saving batch")
            torch.save(acts_dict, save_path + f"/{file_spec}acts_dict.pt")
    
    for act_type in act_types:
        for layer in layers:
            acts_dict[act_type][layer] = torch.cat(acts_dict[act_type][layer], dim = 0)
            
    torch.save(acts_dict, save_path + f"/{file_spec}acts_dict.pt")
    
    gc.collect()
    torch.cuda.empty_cache()
    return acts_dict

def get_memmed_activations(mw: ModelWrapper, prompts: Union[List[str],List[int]],
                    save_path: str,
                    save_every: int = 100, 
                    N_TOKS: int = 32,
                    layers: List[int] = None,
                    tok_idxs: List[int] = None,
                    return_prompt_acts: bool = False,
                    logging: bool = False,
                    file_spec: str = "",
                    **generation_kwargs):
    """
    gets residual activations for autoregressive generation

    Args:
        mw (ModelWrapper): _description_
        prompts (Union[List[str],List[int]]): _description_
        save_path (str): _description_
        act_types (List[str], optional): _description_. Defaults to ['resid'].
        save_every (int, optional): _description_. Defaults to 100.
        N_TOKS (int, optional): _description_. Defaults to 32.
        layers (List[int], optional): _description_. Defaults to None.
        tok_idxs (List[int], optional): _description_. Defaults to None.
        return_prompt_acts (bool, optional): _description_. Defaults to False.
        logging (bool, optional): _description_. Defaults to False.
        file_spec (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if layers is None:
        layers = range(1, mw.model.config.num_hidden_layers + 1)
    
    if isinstance(prompts[0], str):
        prompts_str = prompts
        prompts_toks = mw.tokenizer(prompts, return_tensors = 'pt', padding = True)['input_ids']
    else:
        prompts_str = mw.tokenizer.batch_decode(prompts)
        prompts_toks = prompts

    all_generations = []
    all_tokens = []
    all_hidden_states = []
    all_mem_status = {
        'tok_by_tok_sim' : [],
        'char_by_char_sim' : [],
        'lev_distance' : []
    }
    
    for batch_idx in tqdm(range(ceildiv(len(prompts), save_every) )):
        if logging:
            start_time = time.time()
            print(f"Batch {batch_idx + 1} of {ceildiv(len(prompts), save_every)}")
            
        batch_toks = prompts_toks[batch_idx * save_every : (batch_idx + 1) * save_every]
        
        out = mw.batch_generate_autoreg(prompts=batch_toks[:, :N_TOKS],
                    max_new_tokens=N_TOKS,
                    output_hidden_states=True,
                    output_tokens=True,
                    layers = layers,
                    tok_idxs = tok_idxs,
                    return_prompt_acts=return_prompt_acts,
                    do_sample = False,
                    **generation_kwargs,
                    )
        
        all_generations.extend(out['generations'])
        all_tokens.extend(out['tokens'].cpu().numpy().tolist())
        
        all_mem_status['tok_by_tok_sim'].extend(tok_by_tok_similarity(all_tokens[-len(batch_toks):], batch_toks))
        all_mem_status['char_by_char_sim'].extend(char_by_char_similarity(mw.tokenizer.batch_decode(batch_toks ), out['generations']))
        all_mem_status['lev_distance'].extend(levenshtein_distance(mw.tokenizer.batch_decode(batch_toks), out['generations']))

        all_hidden_states.extend(out['hidden_states'].cpu())
        
        if logging:
            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time
            print(f"Time elapsed for this batch: {elapsed_time:.2f} seconds")
            print(f"Num Hidden States Genned: {len(batch_toks)}")
            print(f"Shape of Hidden States: {all_hidden_states[-1].shape}")
            print(f"Generations:")
            for i, g in enumerate(all_generations[-len(batch_toks):]):
                print(g)
                print(f"Lev {all_mem_status['lev_distance'][batch_idx*save_every + i]}")
                print(f"Char by Char {all_mem_status['char_by_char_sim'][batch_idx*save_every + i]}")
                print(f"Tok by Tok {all_mem_status['tok_by_tok_sim'][batch_idx*save_every + i]}")
                print()
            print()
        
        gc.collect()
        torch.cuda.empty_cache()

        #save hidden_states
        torch.save(all_hidden_states[-len(batch_toks):], save_path + f"/{file_spec}check{batch_idx}_all_hidden_states.pt")

    df = pd.DataFrame(list(zip(prompts_str, prompts_toks.tolist(), all_generations, all_tokens, all_mem_status['tok_by_tok_sim'], all_mem_status['char_by_char_sim'], all_mem_status['lev_distance'], [file_spec] * len(prompts_str))), 
                columns =['prompt_str', 'prompt_toks', 'gen_str', 'gen_toks', 'tok_by_tok_sim', 'char_by_char_sim', 'lev_distance', 'source'])

    df.to_csv(save_path + f'/{file_spec}metadata.csv', index=False, escapechar='\\')

    all_hidden_states = torch.stack(all_hidden_states, dim = 0)
    torch.save(all_hidden_states, save_path + f"/{file_spec}all_hidden_states.pt")
    
    #* delete the checkpoints
    print("deleting checkpoints")
    for batch_idx in range(ceildiv(len(prompts), save_every) ):
        os.remove(save_path + f"/{file_spec}check{batch_idx}_all_hidden_states.pt")
        
    return all_hidden_states, all_generations, all_tokens, all_mem_status

def get_activations_autoreg_generic(mw: ModelWrapper, prompts: Union[List[str], List[int]],
                            save_path : str,
                            save_every: int = 100, 
                            max_new_tokens: int = 32,
                            layers: List[int] = None,
                            tok_idxs: List[int] = None,
                            return_prompt_acts: bool = False, 
                            logging : bool = False, 
                            file_spec : str = "",
                            **generation_kwargs):
    """gets residual activations autoregressively without memorization stuff

    Args:
        mw (ModelWrapper): _description_
        prompts (Union[List[str], List[int]]): _description_
        save_path (str): _description_
        save_every (int, optional): _description_. Defaults to 100.
        max_new_tokens (int, optional): _description_. Defaults to 32.
        layers (List[int], optional): _description_. Defaults to None.
        tok_idxs (List[int], optional): _description_. Defaults to None.
        return_prompt_acts (bool, optional): _description_. Defaults to False.
        logging (bool, optional): _description_. Defaults to False.
        file_spec (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    if not os.path.exists(save_path):
        os.makedirs(args.save_path)

    if layers is None:
        layers = range(1, mw.model.config.num_hidden_layers + 1)
        
    all_generations = []
    all_tokens = []
    all_hidden_states = []
    
    for batch_idx in range(ceildiv(len(prompts), save_every) ):
        if logging:
            start_time = time.time()
            print(f"Batch {batch_idx + 1} of {ceildiv(len(prompts), save_every)}")
            
        batch = prompts[batch_idx * save_every : (batch_idx + 1) * save_every]
        if isinstance(prompts[0], str):
            batch = mw.tokenizer(batch, return_tensors = 'pt', padding = True)['input_ids']
        
        out = mw.batch_generate_autoreg(prompts=batch,
                    max_new_tokens=max_new_tokens,
                    output_hidden_states=True,
                    output_tokens=True,
                    layers = layers,
                    tok_idxs = tok_idxs,
                    return_prompt_acts=return_prompt_acts,
                    **generation_kwargs,
                    )
        
        all_hidden_states.extend(out['hidden_states'].cpu())
        all_generations.extend(out['generations'])
        all_tokens.extend(out['tokens'].cpu().numpy().tolist())
        
        if logging:
            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time
            print(f"Time elapsed for this batch: {elapsed_time:.2f} seconds")
            print(f"Num Hidden States Genned: {len(batch)}")
            print(f"Shape of Hidden States: {all_hidden_states[-1].shape}")
            print(f"Generations:")
            for i, g in enumerate(all_generations[-len(batch):]):
                print(g)
                print("__________________________")
            print()
        
        gc.collect()
        torch.cuda.empty_cache()

        #Saving
        torch.save(all_hidden_states[-len(batch):], save_path + f"/{file_spec}check{batch_idx}_all_hidden_states.pt")

        # Save all_generations
        with open(save_path + f"/{file_spec}all_generations.pkl", "wb") as f:
            pickle.dump(all_generations, f)
            
        with open(save_path + f"/{file_spec}all_tokens.pkl", "wb") as f:
            pickle.dump(all_tokens, f)
        
    all_hidden_states = torch.stack(all_hidden_states, dim = 0)
    torch.save(all_hidden_states, save_path + f"/{file_spec}all_hidden_states.pt")
    
    #* delete the checkpoints
    print("deleting checkpoints")
    for batch_idx in range(ceildiv(len(prompts), save_every) ):
        os.remove(save_path + f"/{file_spec}check{batch_idx}_all_hidden_states.pt")
        
    return all_hidden_states, all_generations, all_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="path to dataset or dataset name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--act_types",
        nargs="+",
        default = None,
    )
    parser.add_argument(
        "--N_PROMPTS",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--N_TOKS",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--return_prompt_acts",
        action="store_true",
    )
    parser.add_argument(
        "--logging",
        action="store_true",
    )
    parser.add_argument(
        '--seed', 
        type=int,
        default=0,
    )
    parser.add_argument(
        "--mem",
        action="store_true",
    )
    parser.add_argument(
        "--tok_idxs",
        nargs="+",
        default=None,
    )


    args = parser.parse_args()
    
    print(args)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    mw = ModelWrapper(model, tokenizer)
    
    if 'pythia-evals' in args.dataset: 
        if "deduped" in args.model_name:
            #model_name looks like: EleutherAI/pythia-12B-deduped
            dataset_name = "deduped." + args.model_name.split("-")[-2]
        else:
            #model_name looks like: EleutherAI/pythia-12B
            dataset_name = "duped." + args.model_name.split("-")[-1]
        
        mem_data = load_dataset('EleutherAI/pythia-memorized-evals')[dataset_name]

        toks = [seq for seq in mem_data[:args.N_PROMPTS]['tokens']]
        
        #? this seems silly
        for i in range(len(toks)): 
            left = 64 - len(toks[i])
            assert left == 0, "Need to pad left"
            toks[i] = [tokenizer.pad_token_id] * left + toks[i] # pad left as suggested above
        
        toks = torch.tensor(toks)
        prompts = [toks_to_string(tokenizer, seq) for seq in toks]
        file_spec = 'mem_'
    elif args.dataset == 'pile': 
        prompts = gen_pile_data(args.N_PROMPTS, tokenizer, min_n_toks = 64)
        toks = tokenizer(prompts, return_tensors = 'pt', padding = True, max_length = 64, truncation = True)['input_ids']
        file_spec = 'pile_'        
    elif args.dataset == 'pile-test': 
        prompts = gen_pile_data(args.N_PROMPTS, tokenizer, min_n_toks = 64, split='validation')
        toks = tokenizer(prompts, return_tensors = 'pt', padding = True, max_length = 64, truncation = True)['input_ids']
        file_spec = 'pile_test_'
    else:
        dataset = pickle.load(open(args.dataset, 'rb'))
        prompts = dataset[:args.N_PROMPTS]
        if isinstance(prompts[0], str):
            toks = tokenizer(prompts, return_tensors='pt', padding=True, max_length=64, truncation=True)['input_ids']
        else:
            toks = torch.tensor(prompts)
        file_spec = (args.dataset.split("/")[-1]).split(".")[0] + "_"
        
    if args.mem:
        tok_idxs =  (7 * np.arange(10)).tolist() #every 5th token
        tok_idxs[-1]= tok_idxs[-1] - 1 #goes from 63 to 62
        print(tok_idxs)

        hidden_states, generations, gen_tokens, mem_status = get_memmed_activations(mw,
                                                                                toks, 
                                                                                args.save_path,
                                                                                save_every = args.save_every,
                                                                                N_TOKS = args.N_TOKS,
                                                                                layers = args.layers,
                                                                                tok_idxs = tok_idxs,
                                                                                return_prompt_acts = args.return_prompt_acts,
                                                                                logging = args.logging,
                                                                                file_spec = file_spec)
        
        if args.act_types: 
            acts_dict = get_memmed_activations_from_pregenned(mw,
                                                            gen_tokens,
                                                            args.save_path,
                                                            act_types = args.act_types,
                                                            save_every = args.save_every,
                                                            layers = args.layers,
                                                            tok_idxs = tok_idxs,
                                                            logging = args.logging,
                                                            file_spec = file_spec + "attn_mlp_")
    else:
        
        hidden_states, generations, gen_tokens = get_activations_autoreg_generic(mw,
                                                                                toks, 
                                                                                args.save_path,
                                                                                save_every = args.save_every,
                                                                                max_new_tokens = args.N_TOKS,
                                                                                layers = args.layers,
                                                                                return_prompt_acts = args.return_prompt_acts,
                                                                                logging = args.logging,
                                                                                tok_idxs = args.tok_idxs,
                                                                                file_spec = file_spec)

    print("Done")