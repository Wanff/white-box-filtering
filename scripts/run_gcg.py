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
import argparse 
import json

import sys
sys.path.append('../') 

from white_box.model_wrapper import ModelWrapper
from white_box.utils import gen_pile_data 
from white_box.dataset import clean_data 
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS

from white_box.dataset import PromptDist, ActDataset, create_prompt_dist_from_metadata_path, ProbeDataset
from white_box.probes import LRProbe
from white_box.gcg import run, GCGConfig
from white_box.monitor import ActMonitor, TextMonitor

def parse_args():
    parser = argparse.ArgumentParser()
    
    #file saving args
    parser.add_argument("--model_name", type=str, required=True,
                        help="The name of the model")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving activations")
    parser.add_argument("--file_spec", type =str, 
                        help="string appended to saved data")
    
    # GCG Args
    parser.add_argument("--num_steps", type = int, default = 300, 
                        help = "num steps for GCG")
    parser.add_argument("--search_width", type = int, default = 48, 
                        help = "batch size for GCG")
    parser.add_argument("--n_replace", type = int, default = 16, 
                    help = "number of tokens to replace per search in GCG")
    parser.add_argument("--buffer_size", type = int, default = 16, 
                    help = "buffer size in GCG")
    parser.add_argument("--gcg_loss_weight", type = float, default = 1, 
                    help = "")
    parser.add_argument("--monitor_loss_weight", type = float, default = 1, 
                    help = "")
    parser.add_argument("--use_search_width_sched", action="store_true",
                    help = "")
    parser.add_argument("--seed", type = int, default = None)
    
    # Monitor Args
    parser.add_argument('--monitor_type', type = str, help = "can be act, act_rand, text, or none")
    parser.add_argument("--probe_layer", type = int, default = 24,
                        help="string appended to saved acts")
    parser.add_argument("--probe_data_path", type = str, default = "jb_",
                        help="file spec for the acts/metadata used to train the probe")
    parser.add_argument("--probe_reg", type = float, default = 1e-2,
                        help="regularization for the probe")
    parser.add_argument("--max_iter", type = int, default = 2000,
                        help="num iterations for the probe")
    
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    print(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    model_config = MODEL_CONFIGS[args.model_name]
    model, tokenizer = load_model_and_tokenizer(**model_config)
    template = get_template(args.model_name, chat_template=model_config.get('chat_template', None))

    mw = ModelWrapper(model, tokenizer, template = template)
    
    if "act" in args.monitor_type:
        layer = args.probe_layer
        neg =  create_prompt_dist_from_metadata_path(f'{args.probe_data_path}metadata.csv', col_filter = "(metadata['jb_name'] == 'harmless')")
        pos = create_prompt_dist_from_metadata_path(f'{args.probe_data_path}metadata.csv', col_filter = "(metadata['jb_name'] == 'DirectRequest')")
        print(len(pos.idxs), len(neg.idxs))
        dataset = ActDataset([pos], [neg])
        dataset.instantiate()
        probe_dataset = ProbeDataset(dataset)

        acc, auc, probe = probe_dataset.train_sk_probe(layer, tok_idxs = list(range(5)), test_size = 0.25, C = args.probe_reg, max_iter = args.max_iter, device = mw.model.device)
        print(acc, auc)
        
        if args.monitor_type == "act_rand":
            trained_probe = probe
            
            probe  = LRProbe.from_weights(torch.rand_like(probe.net[0].weight.data), torch.rand_like(probe.net[0].bias.data), device = mw.model.device) #random probe
            print(probe.net[0].weight.data)
            print("Acc of Random Probe")
            print(probe_dataset.idxs_probe_gets_wrong(probe, layer, tok_idxs = list(range(5))))
            print(f"Cosine Similarity of Random Probe to Learned Probe: {torch.nn.functional.cosine_similarity(probe.net[0].weight.data, trained_probe.net[0].weight.data)}")
            

        monitor = ActMonitor(probe = probe, layer = layer, tok_idxs = [-1, -2, -3, -4, -5], device = mw.model.device)
    elif args.monitor_type == "text":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/LlamaGuard-7b", 
                torch_dtype=torch.float16, 
                device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/LlamaGuard-7b",
            use_fast=False,
            padding_side="left")
        
        monitor = TextMonitor(model, tokenizer, score_id = 25110)
    else:
        monitor = None
    
    advbench_behaviors = pd.read_csv("../data/harmful_behaviors_custom.csv")
    results = []
    for i, row in list(advbench_behaviors.iterrows())[18:]:
        
        gcg_config = GCGConfig(num_steps = args.num_steps, 
                               search_width = args.search_width, 
                               n_replace = args.n_replace, 
                               buffer_size = args.buffer_size, 
                               gcg_loss_weight = args.gcg_loss_weight,
                               monitor_loss_weight = args.monitor_loss_weight,
                               use_search_width_sched = args.use_search_width_sched)
        print(row['goal'])
        print(row['target'])
        

        attack_res = run(mw, messages = row['goal'], target = row['target'], monitor = monitor, config = gcg_config)
        results.append(attack_res)

    with open(args.save_path + f"/{args.file_spec}.json", 'a') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

#python run_gcg.py &> ../data/llama2_7b/gcg_run2.out &