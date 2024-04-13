import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse
import gc

from typing import List, Optional, Tuple, Dict, Union, Callable
from collections import defaultdict

import torch
from datasets import Dataset, load_from_disk, load_dataset
from accelerate.utils import find_executable_batch_size

from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS
from white_box.model_wrapper import ModelWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Generating activations for chat HF models")
    parser.add_argument("--model_name", type=str,
                        help="The name of the model")
    parser.add_argument("--dataset_path", type=str,
        help="path to HF dataset or a csv file with prompts",
    )
    parser.add_argument("--save_path", type=str,
                        help="The path for saving activations")
    
    # Activation saving arguments
    parser.add_argument("--act_types", nargs="+", default = None,
                        help="The types of activations to save: ['resid', 'mlp', 'attn']")
    parser.add_argument("--layers", nargs="+", default = None)
    parser.add_argument("--tok_idxs", nargs="+", default=None)
    parser.add_argument("--tok_mult", type = int, default = None, help = "Every tok_mult-th token position will be saved")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    
    model_config = MODEL_CONFIGS[args.model_name]
    model, tokenizer = load_model_and_tokenizer(**model_config)
    template = get_template(args.model_name, chat_template=model_config.get('chat_template', None))
    
    mw = ModelWrapper(model, tokenizer, template = template)

    if args.dataset_path.endswith(".csv"):
        prompts = pd.read_csv(args.dataset_path).prompt.tolist()
    else:
        #! not fleshed out
        dataset = load_dataset("csv", data_files=args.dataset_path)
        prompts = dataset['train']['prompt']
    
    if args.tok_mult is not None:
        tok_idxs = list(range(0, len(prompts), args.tok_mult))
    else:
        tok_idxs = None
        
    acts = get_activations(mw, prompts, args.save_path,
                            act_types = args.act_types,
                            layers = args.layers,
                            tok_idxs = tok_idxs,
                            logging = True,
                            file_spec = f"{args.model_name}_",
                            )
    
    
def get_activations(mw: ModelWrapper, prompts: Union[List[str],List[int]],
                        save_path: str,
                        act_types: List[str] = ['resid'],
                        layers: List[int] = None,
                        tok_idxs: List[int] = None,
                        logging: bool = False,
                        file_spec: str = "",
                        **generation_kwargs,
                        ):
    """gets activations from pregenerated prompts, which enables access to attention/mlp
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if layers is None:
        layers = list(range(mw.model.config.num_hidden_layers))

    acts_dict = mw.batch_hiddens(prompts,
                                    layers = layers,
                                    tok_idxs = tok_idxs,
                                    return_types = act_types,
                                    logging = logging,
                                    **generation_kwargs,
                                    )

    torch.save(acts_dict, save_path + f"/{file_spec}acts_dict.pt")
    
    gc.collect()
    torch.cuda.empty_cache()
    return acts_dict