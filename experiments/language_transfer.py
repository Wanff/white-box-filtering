import numpy as np 
import torch 
from tqdm import tqdm 
import pickle 
import pandas as pd
from typing import List, Dict, Any, Tuple, Union, Optional, Callable
import requests 
import time 

import datasets
from datasets import load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('../') 

from white_box.model_wrapper import ModelWrapper
from white_box.utils import gen_pile_data 
from white_box.dataset import clean_data 
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS

from white_box.dataset import PromptDist, ActDataset, create_prompt_dist_from_metadata_path, ProbeDataset
from white_box.probes import LRProbe
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk, DatasetDict
from sklearn.metrics import accuracy_score, roc_auc_score

import argparse 

def main(args): 

    if args.probe_dataset == 'jb_': 
        neg =  create_prompt_dist_from_metadata_path(args.orig_path + f'{args.probe_dataset}metadata.csv', col_filter = "(metadata['jb_name'] == 'harmless')")
        pos = create_prompt_dist_from_metadata_path(args.orig_path + f'{args.probe_dataset}metadata.csv', col_filter = "(metadata['jb_name'] == 'DirectRequest')")
    else: 
        raise NotImplementedError
    print(len(pos.idxs), len(neg.idxs))
    dataset = ActDataset([pos], [neg])
    dataset.instantiate()
    probe_dataset = ProbeDataset(dataset)

    neg =  create_prompt_dist_from_metadata_path(args.data_path + f'/{args.probe_dataset}metadata.csv', col_filter = "(metadata['jb_name'] == 'harmless')")
    pos = create_prompt_dist_from_metadata_path(args.data_path + f'/{args.probe_dataset}metadata.csv', col_filter = "(metadata['jb_name'] == 'DirectRequest')")
    print(len(pos.idxs), len(neg.idxs))
    dataset = ActDataset([pos], [neg])
    dataset.instantiate()
    eq_nm_probe_dataset = ProbeDataset(dataset)

    neg =  create_prompt_dist_from_metadata_path(args.data_path + f'/harmless_behaviors_custom_metadata.csv', col_filter='metadata["prompt"] != ""')
    pos = create_prompt_dist_from_metadata_path(args.data_path + f'/harmful_behaviors_custom_metadata.csv', col_filter='metadata["prompt"] != ""')
    print(len(pos.idxs), len(neg.idxs))
    dataset = ActDataset([pos], [neg])
    dataset.instantiate()
    nq_nm_probe_dataset = ProbeDataset(dataset)

    # english model, english language
    acc, auc, probe = probe_dataset.train_sk_probe(args.layer, tok_idxs = list(range(5)), test_size = 0.1, C = 1e-3, max_iter = 2000)
    print(f'{args.probe_dataset} {args.probe_type} {args.layer} trained on english questions & english model {acc} {auc}')

    # english, new model
    acc, auc, eq_nm_probe = eq_nm_probe_dataset.train_sk_probe(args.layer, tok_idxs = list(range(5)), test_size = 0.1, C = 1e-3, max_iter = 2000)
    print(f'{args.probe_dataset} {args.probe_type} {args.layer} trained on english questions & new model {acc} {auc}')

    # new language, new model
    acc, auc, nq_nm_probe = nq_nm_probe_dataset.train_sk_probe(args.layer, tok_idxs = list(range(5)), test_size = 0.1, C = 1e-3, max_iter = 2000)
    print(f'{args.probe_dataset} {args.probe_type} {args.layer} trained on new questions & new model {acc} {auc}')

    # ee transfer to english, new model
    eq_nm_acc, eq_nm_auc = eq_nm_probe_dataset.get_probe_accuracy(probe, args.layer, list(range(5)))
    print(f'{args.probe_dataset} {args.probe_type} {args.layer} trained on eq em, tested on eq nm {eq_nm_acc} {eq_nm_auc}')

    # ee transer to new language, new model
    nq_nm_acc, nq_nm_auc = nq_nm_probe_dataset.get_probe_accuracy(probe, args.layer, list(range(5)))
    print(f'{args.probe_dataset} {args.probe_type} {args.layer} trained on eq em, tested on nq nm {nq_nm_acc} {nq_nm_auc}')

    # eq nm transfer to new language new model
    nq_nm_acc, nq_nm_auc = nq_nm_probe_dataset.get_probe_accuracy(eq_nm_probe, args.layer, list(range(5)))
    print(f'{args.probe_dataset} {args.probe_type} {args.layer} trained on eq nm, tested on nq nm {nq_nm_acc} {nq_nm_auc}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='White-box language transfer')
    parser.add_argument('--orig_path', type=str, default='../data/llama2_7b/', help='Path to original data')
    parser.add_argument('--data_path', type=str, default='../data/turkish/', help='Path to data')
    parser.add_argument('--model_name', type=str, default='akdeniz27/llama-2-7b-hf-qlora-dolly15k-turkish', help='Model name')
    parser.add_argument('--probe_dataset', type=str, default='jb_', help='Probe dataset')
    parser.add_argument('--probe_type', type=str, default='sk', help='Probe type')
    parser.add_argument('--layer', type=int, default=24, help='Layer')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)
