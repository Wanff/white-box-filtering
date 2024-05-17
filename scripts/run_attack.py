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
from peft import AutoPeftModelForSequenceClassification, AutoPeftModelForCausalLM

import sys
sys.path.append('../') 

from white_box.model_wrapper import ModelWrapper
from white_box.utils import gen_pile_data 
from white_box.dataset import clean_data 
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS

from white_box.dataset import PromptDist, ActDataset, create_prompt_dist_from_metadata_path, ProbeDataset
from white_box.probes import LRProbe
from white_box.monitor import ActMonitor, TextMonitor

from white_box.attacks.gcg import GCGConfig

from white_box.attacks.gcg import run as run_gcg
# from white_box.attacks.pair import run as run_pair
# from white_box.attacks.log_prob_attack import run as run_log_prob

def parse_args():
    parser = argparse.ArgumentParser()
    
    #file saving args
    parser.add_argument("--model_name", type=str, required=True,
                        help="The name of the model")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving activations")
    parser.add_argument("--file_spec", type =str, 
                        help="string appended to saved data")
    parser.add_argument("--attack_type", type = str,
                        help="can be pair, gcg, or log_prob")
    parser.add_argument("--attack_args_path", type = str,
                        help = "path to json config file")
    parser.add_argument("--seed", type = int, default = 0)
    
    # Monitor Args
    parser.add_argument('--monitor_type', type = str, help = "can be act, act_rand, text, or none")
    parser.add_argument('--monitor_path', type = str, help = "path to monitor model")
    parser.add_argument('--text_monitor_config', type = str, default='llamaguard+', help = "config for text monitor")
    parser.add_argument("--probe_layer", type = int, default = 24,
                        help="string appended to saved acts")
    parser.add_argument("--probe_type", type = str, default = "sk",
                        help="type of probe")
    parser.add_argument("--tok_idxs", nargs="+", default = [-1, -2, -3, -4, -5],
                        help="tok_idxs")
    
    parser.add_argument("--probe_data_path", type = str, default = "../data/llama2_7b/jb_",
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
    
    if args.monitor_type == 'act': 
        layer = args.probe_layer
        # if last 3 are 'jb_'
        if args.probe_data_path[-3:] == 'jb_':
            neg =  create_prompt_dist_from_metadata_path(f'{args.probe_data_path}metadata.csv', col_filter = "(metadata['jb_name'] == 'harmless')")
            pos = create_prompt_dist_from_metadata_path(f'{args.probe_data_path}metadata.csv', col_filter = "(metadata['jb_name'] == 'DirectRequest')")
        else: 
            pos = create_prompt_dist_from_metadata_path(f'{args.probe_data_path}metadata.csv', col_filter = "(metadata['label'] == 1)")
            neg =  create_prompt_dist_from_metadata_path(f'{args.probe_data_path}metadata.csv', col_filter = "(metadata['label'] == 0)")
        print(len(pos.idxs), len(neg.idxs))
        dataset = ActDataset([pos], [neg])
        dataset.instantiate()
        probe_dataset = ProbeDataset(dataset)

        if args.probe_type == "sk":
            acc, auc, probe = probe_dataset.train_sk_probe(layer, tok_idxs = [int(i) for i in args.tok_idxs], test_size = None, C = args.probe_reg, max_iter = args.max_iter, use_train_test_split=False, device='cuda')
        elif args.probe_type == "mm":
            acc, auc, probe = probe_dataset.train_mm_probe(layer, tok_idxs= [int(i) for i in args.tok_idxs], test_size=None)
        elif args.probe_type == "mlp":
            acc, auc, probe = probe_dataset.train_mlp_probe(layer, tok_idxs= [int(i) for i in args.tok_idxs], test_size=None,
                                                            weight_decay = 1, lr = 0.0001, epochs = 5000)
        print(acc, auc)
        
        if args.monitor_type == "act_rand":
            trained_probe = probe
            
            probe  = LRProbe.from_weights(torch.rand_like(probe.net[0].weight.data), torch.rand_like(probe.net[0].bias.data)) #random probe
            print(probe.net[0].weight.data)
            print("Acc of Random Probe")
            print(probe_dataset.idxs_probe_gets_wrong(probe, layer, tok_idxs = list(range(5))))
            print(f"Cosine Similarity of Random Probe to Learned Probe: {torch.nn.functional.cosine_similarity(probe.net[0].weight.data, trained_probe.net[0].weight.data)}")
            
        monitor = ActMonitor(probe = probe, layer = layer, tok_idxs = [int(i) for i in args.tok_idxs])
    elif args.monitor_type == "text":
        if args.monitor_path is not None: 
            if "peft" in args.monitor_path:
                model = AutoPeftModelForCausalLM.from_pretrained(args.monitor_path, 
                    torch_dtype=torch.float16, 
                    device_map="auto")
                model = model.merge_and_unload()
            else:
                model = AutoModelForCausalLM.from_pretrained(args.monitor_path, 
                    torch_dtype=torch.bfloat16, 
                    device_map="auto")
        else: 
            model = AutoModelForCausalLM.from_pretrained("meta-llama/LlamaGuard-7b", 
                torch_dtype=torch.float16, 
                device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/LlamaGuard-7b",
            use_fast = False,
            padding_side = "right")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        model.train()
        monitor = TextMonitor(model, tokenizer, args.text_monitor_config)
    else:
        monitor = None
    
    if args.attack_type == "gcg":
        model_config = MODEL_CONFIGS[args.model_name]
        model, tokenizer = load_model_and_tokenizer(**model_config)
        template = get_template(args.model_name, chat_template=model_config.get('chat_template', None))

        mw = ModelWrapper(model, tokenizer, template = template)
        
    advbench_behaviors = pd.read_csv("../data/harmful_behaviors_custom_metadata.csv")
    results = []
    
    config = json.load(open(args.attack_args_path, 'r'))
    if args.attack_type == "gcg":
        gcg_config = GCGConfig(**config)
        print(gcg_config.__dict__)
    elif args.attack_type == "pair":
        pair_config = config
        pair_config['monitor'] = monitor
        print(pair_config)
    elif args.attack_type == "log_prob":
        log_prob_config = config
        log_prob_config['monitor'] = monitor
        print(log_prob_config)
    else:
        raise ValueError(f"attack_type {args.attack_type} not recognized")
    
    advbench_behaviors = advbench_behaviors.sample(frac=1, random_state = args.seed)

    for i, row in list(advbench_behaviors.iterrows()):
        print(row['goal'])
        print(row['target'])
        
        if args.attack_type == "gcg":
            attack_res = run_gcg(mw, 
                             messages = row['goal'], target = row['target'], 
                             monitor = monitor, config = gcg_config)
        elif args.attack_type == "pair":
            pair_config['goal'] = row['goal']
            pair_config['target'] = row['target']
            attack_res = run_pair(**pair_config)
            
        elif args.attack_type == "log_prob":
            log_prob_config['goal'] = row['goal']
            log_prob_config['target'] = row['target']
            attack_res = run_log_prob(**log_prob_config)  
        
        results.append(attack_res)

    with open(args.save_path + f"/{args.file_spec}.json", 'a') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

#python run_gcg.py &> ../data/llama2_7b/gcg_run2.out &