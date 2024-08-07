import numpy as np 
import torch 
from tqdm import tqdm 
import pickle 
import pandas as pd
from typing import List, Dict, Any, Tuple, Union, Optional, Callable
import requests 
import time 
import os

import datasets
from datasets import load_dataset
from dataclasses import dataclass
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import argparse 
import json
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import sys
sys.path.append('../') 

from white_box.model_wrapper import ModelWrapper
from white_box.utils import gen_pile_data, get_batched_preds
from white_box.dataset import clean_data 
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS

from white_box.dataset import PromptDist, ActDataset, create_prompt_dist_from_metadata_path, ProbeDataset
from white_box.probes import LRProbe
from white_box.monitor import ActMonitor, TextMonitor

def main(args): 
    
    set_seed(args.seed)
    
    model_config = MODEL_CONFIGS['gemma-2b']
    model, tokenizer = load_model_and_tokenizer(**model_config, padding_side='right', model_override = f'../data/llama2_7b/gemma-2b_head_harmbench_alpaca_metadata_model_0_{args.seed}')
    template = get_template('gemma', chat_template=model_config.get('chat_template', None))['prompt']

    alpaca_negatives = pd.read_csv('../data/llama2_7b/alpaca_negatives_metadata.csv')['prompt'].tolist()
    preds = get_batched_preds(alpaca_negatives, model, tokenizer, template, 'cuda', batch_size=64, head=True)
    np.save(f'../data/llama2_7b/gemma_alpaca_negatives_preds_{args.seed}.npy', preds)
    
if __name__ == '__main__':
    
    # get first arg 
    parser = argparse.ArgumentParser(description='Get FPR preds')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    main(args)