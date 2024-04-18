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
import jailbreakbench as jbb 

import sys
sys.path.append('../') 

from white_box.model_wrapper import ModelWrapper
from white_box.utils import gen_pile_data 
from white_box.dataset import clean_data 
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS

from white_box.dataset import PromptDist, ActDataset, create_prompt_dist_from_metadata_path, ProbeDataset
from white_box.probes import LRProbe
from white_box.gcg import run, GCGConfig
from white_box.monitor import ActMonitor


if __name__=="__main__":
    model_name = 'llama2_7b'
    data_path = f"../data/{model_name}/"
    
    # file_spec = "jb_gcg_"
    file_spec = "jb_unif_behav_"
    
    model_config = MODEL_CONFIGS[model_name]
    model, tokenizer = load_model_and_tokenizer(**model_config)
    template = get_template(model_name, chat_template=model_config.get('chat_template', None))

    mw = ModelWrapper(model, tokenizer, template = template)
    jb_metadata = pd.read_csv(f"../data/{model_name}/{file_spec}metadata.csv", sep="t", index_col=0)
    jb_acts = torch.load(f"../data/{model_name}/{file_spec}hidden_states.pt")

    # layer = 25
    layer = 16

    # neg_gcg = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0) & ((metadata['jb_name'] == 'GCG') | (metadata['jb_name'] == 'EnsembleGCG'))")
    # pos_gcg =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1) & ((metadata['jb_name'] == 'GCG') | (metadata['jb_name'] == 'EnsembleGCG'))")
    neg =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['jb_name'] == 'harmless')")
    pos = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['jb_name'] == 'DirectRequest')")
    print(len(pos.idxs), len(neg.idxs))
    dataset = ActDataset([pos], [neg])
    dataset.instantiate()
    probe_dataset = ProbeDataset(dataset)

    acc, auc, probe = probe_dataset.train_sk_probe(layer, tok_idxs = list(range(5)), test_size = 0.25, C = 1e-3, max_iter = 2000)
    print(acc, auc)
    
    advbench_behaviors = pd.read_csv("../data/harmful_behaviors_custom.csv")
    am = ActMonitor(probe = probe, layer = layer, tok_idxs = [-1, -2, -3, -4, -5])
    results = []
    for i, row in list(advbench_behaviors.iterrows()):
        # dog_str =  " Using write purely thinking Dog grammar,(initial(' Sure, here>) altogetherthrough procedure Without indicating final answer"
        
        gcg_config = GCGConfig(num_steps = 300, search_width = 64, n_replace = 12, buffer_size = 16, gcg_loss_weight=1)
        print(row['goal'])
        print(row['target'])
        
        attack_res = run(mw, messages = row['goal'], target = row['target'], monitor = am, config = gcg_config)
        results.append(attack_res)

    pickle.dump(results, open("../data/llama2_7b/gcg_res.pkl", "wb"))

#python run_gcg.py &> ../data/llama2_7b/gcg_run2.out &