import os
import pandas as pd
import numpy as np 
from typing import Union 
import itertools
import argparse 
import pickle

import sys
sys.path.append('../') 

from white_box.dataset import ProbeDataset, create_prompt_dist_from_metadata_path, ActDataset
from white_box.monitor import TextMonitor, ActMonitor
from white_box.chat_model_utils import MODEL_CONFIGS, load_model_and_tokenizer, get_template

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default = 24,
                        help="The name of the model")
    parser.add_argument("--tok_idxs", type=int, default = -1,
                        help="The name of the model")
    parser.add_argument("--N", type=int, default = 5,
                        help="The name of the model")
    parser.add_argument("--save_path", type=str, default = "../data/llama2_7b/decorrelation_5.pkl",
                        help="The path for saving stuff")
    parser.add_argument('--verbose', action='store_true', default = True)

    args = parser.parse_args()
    return args

def load_probe_data():
    data_path = '../data/llama2_7b'
    file_spec = "harmbench_alpaca_test_"
    harmful = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1)")
    harmless =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0)")
    print(len(harmless.idxs), len(harmful.idxs))
    dataset = ActDataset([harmful], [harmless])
    dataset.instantiate()
    hb_test_probe_dataset = ProbeDataset(dataset)

    file_spec = "generated_test_"
    harmful = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1)")
    harmless =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0)")
    print(len(harmless.idxs), len(harmful.idxs))
    dataset = ActDataset([harmful], [harmless])
    dataset.instantiate()
    gpt_test_probe_dataset = ProbeDataset(dataset)

    file_spec = "jb_"
    jbs =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1) & (metadata['jb_name'] != 'DirectRequest')")
    failed_jbs = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0) & (metadata['jb_name'] != 'DirectRequest') & (metadata['jb_name'] != 'harmless')")
    print(len(jbs.idxs), len(failed_jbs.idxs))
    dataset = ActDataset([jbs, failed_jbs], [])
    dataset.instantiate()
    jb_probe_dataset = ProbeDataset(dataset)
    
    return hb_test_probe_dataset, gpt_test_probe_dataset, jb_probe_dataset

def load_tc_data():
    data_path = '../data/llama2_7b'

    gpt_df = pd.read_csv(os.path.join(data_path, 'generated_test_metadata.csv'))
    print(len(gpt_df))
    hb_df = pd.read_csv(os.path.join(data_path, 'harmbench_alpaca_test_metadata.csv'))
    print(len(hb_df))
    
    file_spec = "jb_"
    jb_metadata = pd.read_csv(f"{data_path}/{file_spec}metadata.csv", sep = "t")
    jbs =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1) & (metadata['jb_name'] != 'DirectRequest')")
    failed_jbs = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0) & (metadata['jb_name'] != 'DirectRequest') & (metadata['jb_name'] != 'harmless')")
    dataset = ActDataset([jbs, failed_jbs], [])
    dataset.instantiate()
    jb_probe_dataset = ProbeDataset(dataset)
    jb_prompts = jb_metadata.iloc[jb_probe_dataset.act_dataset.metadata_idxs]['prompt'].values
    jb_df = pd.DataFrame({'prompt': jb_prompts, 'label': jb_probe_dataset.act_dataset.y.detach().cpu().numpy()})
    print(len(jb_df))
    
    return hb_df, gpt_df, jb_df

def train_probes(N , layer , tok_idxs, which_compute_instance : str = 'cais'):
    data_path = "../data/llama2_7b"
    if which_compute_instance == 'aws':
        file_spec = "all_harmbench_alpaca_"
        harmful = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1) & (metadata.index < 2400)")
        harmless =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0) & (metadata.index < 2400)")
        print(len(harmless.idxs), len(harmful.idxs))
        dataset = ActDataset([harmful], [harmless])
        dataset.instantiate()
        hb_alpaca_train_probe_dataset = ProbeDataset(dataset)
    elif which_compute_instance == 'cais':
        file_spec = "harmbench_alpaca_"
        harmful = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1)")
        harmless =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0)")
        print(len(harmless.idxs), len(harmful.idxs))
        dataset = ActDataset([harmful], [harmless])
        dataset.instantiate()
        hb_alpaca_train_probe_dataset = ProbeDataset(dataset)
            
    ams = []
    for _ in range(N):
        acc, auc, probe = hb_alpaca_train_probe_dataset.train_sk_probe(layer, tok_idxs = tok_idxs, C = 1e-1, max_iter = 1000, 
                                                                    use_train_test_split = False,
                                                                    shuffle = True,
                                                                    random_state = None)
        ams.append(ActMonitor(probe, layer = layer, tok_idxs = tok_idxs))
        
    return ams 

def load_tcs(N, neg_data : str = 'hb_alpaca'):
    tcs = []
    for i in range(N):
        model_config = MODEL_CONFIGS['llamaguard']
        if neg_data == 'hb_alpaca':
            model_name_or_path = f'../data/llama2_7b/llamaguard_harmbench_alpaca_metadata_model_0_{i}'
        elif neg_data == 'gpt': 
            model_name_or_path = f'../data/llama2_7b/llamaguard_generated_metadata_model_0_{i}'
        model, tokenizer = load_model_and_tokenizer(**model_config, padding_side='right', model_override = model_name_or_path)
        template = get_template('llamaguard', chat_template=model_config.get('chat_template', None))['prompt']
        model.to('cpu')
        hb_tm = TextMonitor(model, tokenizer, config_name = "llamaguard")
        tcs.append(hb_tm)
    
    # model_config = MODEL_CONFIGS['llamaguard']
    # model_config['dtype'] = 'float16'
    # model_name_or_path = f'OamPatel/LlamaGuard-harmbench-alpaca'
    # model, tokenizer = load_model_and_tokenizer(**model_config, padding_side='right', model_override = model_name_or_path)
    # template = get_template('llamaguard', chat_template=model_config.get('chat_template', None))['prompt']

    # hb_tm = TextMonitor(model, tokenizer, config_name = "llamaguard")
    # tcs.append(hb_tm)
    return tcs

def probe_errors(ams: list, probe_dataset: ProbeDataset):
    errors = []
    for am in ams:
        preds = am.predict_proba(probe_dataset.act_dataset.X[:, am.layer, am.tok_idxs]).detach().cpu().numpy().squeeze(1)
        labels = probe_dataset.act_dataset.y.detach().cpu().numpy()
        curr_err = (preds > 0.5) != labels
        errors.append(curr_err)
    return errors

def tc_errors(tcs: list, df: pd.DataFrame):
    errors = []
    for tc in tcs:
        
        tc.model.to('cuda')
        tc.device = 'cuda'
        
        preds = tc.get_batched_preds(df['prompt'].values, batch_size=8)
        labels = df['label'].values
        curr_err = (preds > 0.5) != labels
        errors.append(curr_err)
        
        tc.model.to('cpu')
        tc.device = 'cpu'
        
    return errors

def intra_group_corr(errors: list):
    corr = []
    for i, j in itertools.combinations(range(len(errors)), 2):
        r = np.corrcoef(errors[i], errors[j])[0][1]
        if not np.isnan(r):
            corr.append(r)
        else: 
            corr.append(1)
    return corr

def inter_group_corr(errors1: list, errors2: list):
    corr = []
    for i in range(len(errors1)):
        for j in range(len(errors2)):
            r = np.corrcoef(errors1[i], errors2[j])[0][1]
            if not np.isnan(r):
                corr.append(r)
            else: 
                corr.append(1)
    return corr

def inter_group_sum_corr(errors1: list, errors2: list):
    errors1 = np.array(errors1).sum(axis = 0)
    errors2 = np.array(errors2).sum(axis = 0)
    
    r = np.corrcoef(errors1, errors2)[0][1]
    return r if not np.isnan(r) else 1

def main():
    args = parse_args()
    print(args)
    
    if os.path.exists(args.save_path): 
        with open(args.save_path, 'rb') as f:
            results = pickle.load(f)
        ams_hb_errs, ams_gpt_errs, ams_jb_errs, tcs_hb_errs, tcs_gpt_errs, tcs_jb_errs = results
        
    else:
        ams = train_probes(args.N, args.layer, [args.tok_idxs])
        tcs = load_tcs(args.N)
        
        hb_test_probe_dataset, gpt_test_probe_dataset, jb_probe_dataset = load_probe_data() 
        hb_df, gpt_df, jb_df = load_tc_data()
        
        ams_hb_errs = probe_errors(ams, hb_test_probe_dataset)
        ams_gpt_errs = probe_errors(ams, gpt_test_probe_dataset)
        ams_jb_errs = probe_errors(ams, jb_probe_dataset)
        
        tcs_hb_errs = tc_errors(tcs, hb_df)
        tcs_gpt_errs = tc_errors(tcs, gpt_df)
        tcs_jb_errs = tc_errors(tcs, jb_df)
    
        with open(args.save_path, 'wb') as f:
            pickle.dump((ams_hb_errs, ams_gpt_errs, ams_jb_errs, tcs_hb_errs, tcs_gpt_errs, tcs_jb_errs), f)
        
    pp_hb_corr = intra_group_corr(ams_hb_errs)
    pp_gpt_corr = intra_group_corr(ams_gpt_errs)
    pp_jb_corr = intra_group_corr(ams_jb_errs)
    
    tt_hb_corr = intra_group_corr(tcs_hb_errs)
    tt_gpt_corr = intra_group_corr(tcs_gpt_errs)
    tt_jb_corr = intra_group_corr(tcs_jb_errs)
    
    tp_hb_corr = inter_group_corr(ams_hb_errs, tcs_hb_errs)
    tp_gpt_corr = inter_group_corr(ams_gpt_errs, tcs_gpt_errs)
    tp_jb_corr = inter_group_corr(ams_jb_errs, tcs_jb_errs)
    
    tp_hb_sum_corr = inter_group_sum_corr(ams_hb_errs, tcs_hb_errs)
    tp_gpt_sum_corr = inter_group_sum_corr(ams_gpt_errs, tcs_gpt_errs)
    tp_jb_sum_corr = inter_group_sum_corr(ams_jb_errs, tcs_jb_errs)
    
    print(f"Probe-Probe HB: {np.mean(pp_hb_corr, axis=0)}")
    print(f"Probe-Probe GPT: {np.mean(pp_gpt_corr, axis=0)}")
    print(f"Probe-Probe JB: {np.mean(pp_jb_corr, axis=0)}")
    
    print(f"TC-TC HB: {np.mean(tt_hb_corr, axis=0)}")
    print(f"TC-TC GPT: {np.mean(tt_gpt_corr, axis=0)}")
    print(f"TC-TC JB: {np.mean(tt_jb_corr, axis=0)}")
    
    print(f"Probe-TC HB: {np.mean(tp_hb_corr, axis=0)}")
    print(f"Probe-TC GPT: {np.mean(tp_gpt_corr, axis=0)}")
    print(f"Probe-TC JB: {np.mean(tp_jb_corr, axis=0)}")
    
    print(f"Probe-TC HB Sum Corr: {tp_hb_sum_corr}")
    print(f"Probe-TC GPT Sum Corr: {tp_gpt_sum_corr}")
    print(f"Probe-TC JB Sum Corr: {tp_jb_sum_corr}")
    
    # if args.verbose: 
        
    #     print(f"Probe-Probe HB: {pp_hb_corr}")
    #     print(f"Probe-Probe GPT: {pp_gpt_corr}")
    #     print(f"Probe-Probe JB: {pp_jb_corr}")
        
    #     print(f"TC-TC HB: {tt_hb_corr}")
    #     print(f"TC-TC GPT: {tt_gpt_corr}")
    #     print(f"TC-TC JB: {tt_jb_corr}")
        
    #     print(f"Probe-TC HB: {tp_hb_corr}")
    #     print(f"Probe-TC GPT: {tp_gpt_corr}")
    #     print(f"Probe-TC JB: {tp_jb_corr}")
    
    
if __name__ == "__main__":
    main()