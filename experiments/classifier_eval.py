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

def tpr_at_fpr_target(y_true: np.ndarray, y_pred: np.ndarray, fpr_target: float) -> (float, float): 
    """
    Get the TPR at a given FPR target. Lower bounds the TPR. Returns a tuple of the TPR and the FPR.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    tpr_at_fpr = np.interp(fpr_target, fpr, tpr)
    actual_fpr = fpr[np.argmin(np.abs(tpr - tpr_at_fpr))]
    return tpr_at_fpr, actual_fpr

def results_given_preds(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]: 
    return {
        'Accuracy': accuracy_score(labels, preds > 0.5),
        'AUC': roc_auc_score(labels, preds),
        'TPR': ((preds > 0.5) & (labels == 1)).sum() / (labels == 1).sum(),
        'FPR': ((preds > 0.5) & (labels == 0)).sum() / (labels == 0).sum(),
        'TNR': ((preds <= 0.5) & (labels == 0)).sum() / (labels == 0).sum(),
        'FNR': ((preds <= 0.5) & (labels == 1)).sum() / (labels == 1).sum(),
        'TPR@FPR0.05': tpr_at_fpr_target(labels, preds, 0.05)[0],
        'TPR@FPR0.01': tpr_at_fpr_target(labels, preds, 0.01)[0],
        'TPR@FPR0.001': tpr_at_fpr_target(labels, preds, 0.001)[0],
    }


def print_results(results: Dict[str, np.ndarray], prefix: str = '') -> None:
    print(prefix)
    for key, value in results.items():
        print(f'{key}: {value}')

def main(args): 
    
    generated_df = pd.read_csv(os.path.join(args.data_path, 'llama2_7b/generated_test_metadata.csv'))
    hb_alpaca_df = pd.read_csv(os.path.join(args.data_path, 'llama2_7b/harmbench_alpaca_test_metadata.csv'))
    file_spec = "jb_"
    jb_metadata = pd.read_csv(f"{args.data_path}/llama2_7b/{file_spec}metadata.csv", sep = "t")
    jbs =  create_prompt_dist_from_metadata_path(args.data_path + f'/llama2_7b/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1) & (metadata['jb_name'] != 'DirectRequest')")
    failed_jbs = create_prompt_dist_from_metadata_path(args.data_path + f'/llama2_7b/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0) & (metadata['jb_name'] != 'DirectRequest') & (metadata['jb_name'] != 'harmless')")
    print(len(jbs.idxs), len(failed_jbs.idxs))
    dataset = ActDataset([jbs], [failed_jbs])
    dataset.instantiate()
    jb_labeled_by_success_probe_dataset = ProbeDataset(dataset)
    jb_labeled_by_success_metadata = jb_labeled_by_success_probe_dataset.metadata
    print(len(jb_labeled_by_success_metadata))

    final = defaultdict(defaultdict(list))
    for seed in range(5): 
        
        model_config = MODEL_CONFIGS['llamaguard']
        model, tokenizer = load_model_and_tokenizer(**model_config, padding_side='right', model_override = f'{args.model_path}_{args.seed}')
        template = get_template('llamaguard', chat_template=model_config.get('chat_template', None))['prompt']

        set_seed(seed)
        
        # advbench + custom gpt
        pos_prompts = pd.read_csv(os.path.join(args.data_path, 'harmful_behaviors_custom_metadata.csv'))['prompt'].tolist()
        neg_prompts = pd.read_csv(os.path.join(args.data_path, 'harmless_behaviors_custom_metadata.csv'))['prompt'].tolist()
        preds = get_batched_preds(pos_prompts + neg_prompts, model, tokenizer, template, args.device, args.batch_size)
        labels = np.concatenate([np.ones(len(pos_prompts)), np.zeros(len(neg_prompts))])
        results = results_given_preds(preds, labels)
        for key, value in results.items():
            final['advbench+customgpt'][key].append(value)
            
        # harmbench_alpaca
        preds = get_batched_preds(hb_alpaca_df['prompt'].tolist(), model, tokenizer, template, args.device, args.batch_size)
        labels = hb_alpaca_df['label'].values
        results = results_given_preds(preds, labels)
        for key, value in results.items():
            final['harmbench_alpaca'][key].append(value)
        
        # generated
        preds = get_batched_preds(generated_df['prompt'].tolist(), model, tokenizer, template, args.device, args.batch_size)
        labels = generated_df['label'].values
        results = results_given_preds(preds, labels)
        for key, value in results.items():
            final['generated'][key].append(value)
            
        # jailbreaks
    
    for key, value in final.items(): 
        print(f'{key}: {np.mean(value)} +/- {np.std(value)}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data', help='data path')
    parser.add_argument('--model_name', type=str, default='llamaguard', help='model name')
    parser.add_argument('--model_path', type=str, default='../data/llama2_7b/llamaguard_harmbench_alpaca__model_0')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    args = parser.parse_args()

    main(args)