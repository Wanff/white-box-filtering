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
    parser.add_argument("--save_path", type=str, default = "../data/llama2_7b/decorrelation.pkl",
                        help="The path for saving stuff")

    args = parser.parse_args()
    return args

def get_error_corr(monitor1 : Union[TextMonitor, ActMonitor], monitor2: Union[TextMonitor, ActMonitor], 
                   df: pd.DataFrame, 
                   probe_dataset : ProbeDataset,
                   batch_size: int = 8):
    if isinstance(monitor1, TextMonitor):
        preds1 = monitor1.get_batched_preds(df['prompt'].values, batch_size=batch_size)
        labels = df['label'].values
    else:
        preds1 = monitor1.predict_proba(probe_dataset.act_dataset.X[:, monitor1.layer, monitor1.tok_idxs]).detach().cpu().numpy().squeeze(1)
        labels = probe_dataset.act_dataset.y.detach().cpu().numpy()

    errors1 = (preds1 > 0.5) != labels
    num_errors1 = errors1.sum()
    if isinstance(monitor2, TextMonitor):
        preds2 = monitor2.get_batched_preds(df['prompt'].values, batch_size=batch_size)
        labels = df['label'].values
    else:
        preds2 = monitor2.predict_proba(probe_dataset.act_dataset.X[:, monitor2.layer, monitor2.tok_idxs]).detach().cpu().numpy().squeeze(1)
        labels = probe_dataset.act_dataset.y.detach().cpu().numpy()
        
    errors2 = (preds2 > 0.5) != labels
    num_errors2 = errors2.sum()
    correlation = np.corrcoef(errors1, errors2)[0][1]
    return 0 if np.isnan(correlation) else correlation, num_errors1, num_errors2


def load_probe_data():
    data_path = '../data/llama2_7b'
    file_spec = "all_harmbench_alpaca_"
    harmful = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1) & (metadata.index >= 2400)")
    harmless =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0) & (metadata.index >= 2400)")
    print(len(harmless.idxs), len(harmful.idxs))
    dataset = ActDataset([harmful], [harmless])
    dataset.instantiate()
    hb_test_probe_dataset = ProbeDataset(dataset)

    file_spec = "all_gpt_gen_"
    harmful = create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 1) & (metadata.index >= 2400)")
    harmless =  create_prompt_dist_from_metadata_path(data_path + f'/{file_spec}metadata.csv', col_filter = "(metadata['label'] == 0) & (metadata.index >= 2400)")
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

    gpt_df = pd.read_csv(os.path.join(data_path, 'gpt_gen_test_metadata.csv')).iloc[:-4]
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

def train_probes(N , layer , tok_idxs, which_compute_instance : str = 'aws'):
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

def inter_probe_corr(ams, hb_test_probe_dataset, gpt_test_probe_dataset, jb_probe_dataset):
    corr_hbs = []
    corr_gpts = []
    corr_jbs = []
    for (i, j) in itertools.combinations(range(len(ams)), 2):
        corr_hb, _, _ = get_error_corr(ams[i], ams[j], hb_test_probe_dataset.metadata, hb_test_probe_dataset)
        corr_gpt, _, _ = get_error_corr(ams[i], ams[j], gpt_test_probe_dataset.metadata, gpt_test_probe_dataset)
        corr_jb, _, _= get_error_corr(ams[i], ams[j], jb_probe_dataset.metadata, jb_probe_dataset)
        
        corr_hbs.append(corr_hb)
        corr_gpts.append(corr_gpt)
        corr_jbs.append(corr_jb)

    print(f"HB Alpaca: {np.mean(corr_hbs, axis=0)}")
    print(f"GPT: {np.mean(corr_gpts, axis=0)}")
    print(f"JB: {np.mean(corr_jbs, axis=0)}")
    return corr_hbs, corr_gpts, corr_jbs

def load_tcs(neg_data : str = 'hb_alpaca'):
    tcs = []
    # for i in range(5):
    #     model_config = MODEL_CONFIGS['llamaguard']
    #     if neg_data == 'hb_alpaca':
    #         model_name_or_path = f'OamPatel/LlamaGuard-harmbench-alpaca-{i}'
    #     elif neg_data == 'gpt': 
    #         model_name_or_path = f'OamPatel/LlamaGuard-gpt-{i}'
    #     model, tokenizer = load_model_and_tokenizer(**model_config, padding_side='right', model_override = model_name_or_path)
    #     template = get_template('llamaguard', chat_template=model_config.get('chat_template', None))['prompt']

    #     hb_tm = TextMonitor(model, tokenizer, config_name = "llamaguard")
    #     tcs.append(hb_tm)
    
    model_config = MODEL_CONFIGS['llamaguard']
    model_config['dtype'] = 'float16'
    model_name_or_path = f'OamPatel/LlamaGuard-harmbench-alpaca'
    model, tokenizer = load_model_and_tokenizer(**model_config, padding_side='right', model_override = model_name_or_path)
    template = get_template('llamaguard', chat_template=model_config.get('chat_template', None))['prompt']

    hb_tm = TextMonitor(model, tokenizer, config_name = "llamaguard")
    tcs.append(hb_tm)
    return tcs

def inter_tc_corr(tcs, hb_df, gpt_df, jb_df):
    corr_hbs = []
    corr_gpts = []
    corr_jbs = []
    for (i, j) in itertools.combinations(range(len(tcs)), 2):
        corr_hb, _, _ = get_error_corr(tcs[i], tcs[j], hb_df.metadata, None)
        corr_gpt, _, _ = get_error_corr(tcs[i], tcs[j], gpt_df.metadata, None)
        corr_jb, _, _= get_error_corr(tcs[i], tcs[j], jb_df.metadata, None)
        
        corr_hbs.append(corr_hb)
        corr_gpts.append(corr_gpt)
        corr_jbs.append(corr_jb)

    print(f"HB Alpaca: {np.mean(corr_hbs, axis=0)}")
    print(f"GPT: {np.mean(corr_gpts, axis=0)}")
    print(f"JB: {np.mean(corr_jbs, axis=0)}")
    return corr_hbs, corr_gpts, corr_jbs

def probe_tc_corr(ams, tcs, hb_df, gpt_df, jb_df, hb_test_probe_dataset, gpt_test_probe_dataset, jb_probe_dataset):
    corr_hbs = []
    corr_gpts = []
    corr_jbs = []
    for i in range(len(ams)):
        for j in range(len(tcs)):
            corr_hb, _, _ = get_error_corr(ams[i], tcs[j], hb_df, hb_test_probe_dataset)
            corr_gpt, _, _ = get_error_corr(ams[i], tcs[j], gpt_df, gpt_test_probe_dataset)
            corr_jb, _, _= get_error_corr(ams[i], tcs[j], jb_df, jb_probe_dataset)

            corr_hbs.append(corr_hb)
            corr_gpts.append(corr_gpt)
            corr_jbs.append(corr_jb)

    print(f"HB Alpaca: {np.mean(corr_hbs, axis=0)}")
    print(f"GPT: {np.mean(corr_gpts, axis=0)}")
    print(f"JB: {np.mean(corr_jbs, axis=0)}")
    return corr_hbs, corr_gpts, corr_jbs

def main():
    args = parse_args()
    print(args)
    
    ams = train_probes(5, args.layer, [args.tok_idxs])
    tcs = load_tcs()
    
    hb_test_probe_dataset, gpt_test_probe_dataset, jb_probe_dataset = load_probe_data() 
    hb_df, gpt_df, jb_df = load_tc_data()
    
    pp_corr_hbs, pp_corr_gpts, pp_corr_jbs = inter_probe_corr(ams, hb_test_probe_dataset, gpt_test_probe_dataset, jb_probe_dataset)
    
    tt_corr_hbs, tt_corr_gpts, tt_corr_jbs = inter_tc_corr(tcs, hb_df, gpt_df, jb_df)
    
    tp_corr_hbs, tp_corr_gpts, tp_corr_jbs = probe_tc_corr(ams, tcs, hb_df, gpt_df, jb_df, hb_test_probe_dataset, gpt_test_probe_dataset, jb_probe_dataset)
    
    res = {
        'pp_corr_hbs': pp_corr_hbs,
        'pp_corr_gpts': pp_corr_gpts,
        'pp_corr_jbs': pp_corr_jbs,
        'tt_corr_hbs': tt_corr_hbs,
        'tt_corr_gpts': tt_corr_gpts,
        'tt_corr_jbs': tt_corr_jbs,
        'tp_corr_hbs': tp_corr_hbs,
        'tp_corr_gpts': tp_corr_gpts,
        'tp_corr_jbs': tp_corr_jbs
    }
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(res, f)
    
    
if __name__ == "__main__":
    main()
    

    
    