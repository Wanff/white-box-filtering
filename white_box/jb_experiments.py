import numpy as np 
import pandas as pd
from tqdm import tqdm 
from collections import defaultdict 

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as sklearn_auc

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from white_box.dataset import PromptDist, ActDataset, create_prompt_dist_from_metadata_path, ProbeDataset

#* EXPERIMENTS
# Leave-one-out generalization
def leave_one_out_generalization_by_jb_success(jb_metadata : pd.DataFrame, split : str = "category"):
    if split == "category":
        probe_data_path = "../data/llama2_7b/jb_eq_by_cat_"
        all_categories = set(jb_metadata['category'].values.tolist()) - {'harmless'}
    elif split == "jb_name":
        probe_data_path = "../data/llama2_7b/jb_eq_by_jbname_"
        all_categories = set(jb_metadata['jb_name'].values.tolist()) - {'DirectRequest', 'harmless', 'AutoDAN'}

    res = {}

    for cat in tqdm(all_categories):
        pos =  create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = f"(metadata['{split}'] != '{cat}') & (metadata['label'] == 1)")
        neg = create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = f"(metadata['{split}'] != '{cat}') & (metadata['label'] == 0)")
        dataset = ActDataset([pos], [neg])
        dataset.instantiate()
        probe_dataset = ProbeDataset(dataset)
        
        pos_cat =  create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = f"(metadata['{split}'] == '{cat}') & (metadata['label'] == 1)")
        neg_cat = create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = f"(metadata['{split}'] == '{cat}') & (metadata['label'] == 0)")
        cat_dataset = ActDataset([pos_cat], [neg_cat])
        cat_dataset.instantiate()
        probe_cat_dataset = ProbeDataset(cat_dataset)
        in_dist_acc, in_dist_auc, gen_acc, gen_auc, in_dist_cat_acc, in_dist_cat_auc = [], [], [], [], [], []
        for layer in range(32):
            acc, auc, probe = probe_dataset.train_sk_probe(layer, tok_idxs = list(range(5)), C = 1e-2, max_iter = 2000, use_train_test_split = False)
            in_dist_acc.append(acc)
            in_dist_auc.append(auc)
            
            acc, auc, in_dist_probe = probe_cat_dataset.train_sk_probe(layer, tok_idxs = list(range(5)), C = 1e-2, max_iter = 2000, use_train_test_split = True)
            in_dist_cat_acc.append(acc)
            in_dist_cat_auc.append(auc)
            
            labels, states = cat_dataset.convert_states(cat_dataset.X, cat_dataset.y, tok_idxs = list(range(5)), layer = layer)
            acc = probe.get_probe_accuracy(states, labels)
            auc = probe.get_probe_auc(states, labels)
            
            gen_acc.append(acc)
            gen_auc.append(auc)
        
        res[cat] = {
            'in_dist_acc' : in_dist_acc,
            'in_dist_auc' : in_dist_auc,
            'gen_acc' : gen_acc,
            'gen_auc' : gen_auc,
            'in_dist_cat_acc' : in_dist_cat_acc,
            'in_dist_cat_auc' : in_dist_cat_auc
        }
    return res

#* PLOTTING
def plot_acc_auc(accs, aucs, title : str = ""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(accs))), y=accs, mode='lines+markers', name='Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(len(aucs))), y=aucs, mode='lines+markers', name='AUC'))

    # Set plot layout
    fig.update_layout(title=f'Accuracy and AUC over Layers : {title}',
                    xaxis_title='Layer',
                    yaxis_title='Value',
                    legend_title='Metrics')

    # Show plot
    fig.show()


def plot_metric_by_category(res, layer, metric):
    categories = []
    values = []
    
    for cat, data in res.items():
        categories.append(cat)
        values.append(np.round(data[metric][layer], 3))
    
    # Create a bar plot
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, text=values, textposition='auto')
    ])
    
    # Update layout
    fig.update_layout(
        title=f'{metric} at Layer {layer}',
        xaxis_title='Category',
        yaxis_title=metric,
        template='plotly_white'
    )
    
    # Show plot
    fig.show()

def plot_dual_metric_by_category(res, dim, primary_metric, secondary_metric, layer=24): 
    categories = []
    primary_values = []
    secondary_values = []
    
    for cat, data in res.items():
        if cat == 'misinformation_disinformation': 
            cat = 'misinfo_disinfo'
        if cat == 'chemical_biological': 
            cat = 'chem_bio'
        categories.append(cat)
        primary_values.append(np.round(data[primary_metric][dim], 3))
        secondary_values.append(np.round(data[secondary_metric][dim], 3))
    
    # Create bar plots for both metrics
    fig = go.Figure(data=[
        go.Bar(name=primary_metric, x=categories, y=primary_values, text=primary_values, textposition='auto', textfont=dict(color='white')),
        go.Bar(name=secondary_metric, x=categories, y=secondary_values, text=secondary_values, textposition='auto', textfont=dict(color='white'))
    ])
    
    # Update layout to have the bars grouped
    fig.update_layout(
        barmode='group',
        title=f'Comparison of {primary_metric} and {secondary_metric} at Layer {layer}',
        xaxis_title='Category',
        yaxis_title='Accuracy',
        template='plotly_white'
    )
    
    # Show plot
    fig.show()


def plot_probe_on_test_dataset(probes, test_dataset, title='test'): 
    
    metrics = defaultdict(list)

    for layer in range(32):
        pred_probas = probes[layer].predict_proba(test_dataset.act_dataset.X[:, layer])
        probas_mean = pred_probas.mean(dim=-1).detach().cpu().numpy()
        labels = test_dataset.act_dataset.y.detach().cpu().numpy()

        metrics['acc'].append(accuracy_score(labels, probas_mean > 0.5))
        metrics['auc'].append(roc_auc_score(labels, probas_mean))
        metrics['TPR'].append(((probas_mean > 0.5) & (labels == 1)).sum() / (labels == 1).sum())
        metrics['TNR'].append(((probas_mean < 0.5) & (labels == 0)).sum() / (labels == 0).sum())
        metrics['FPR'].append(((probas_mean > 0.5) & (labels == 0)).sum() / (labels == 0).sum())
        metrics['FNR'].append(((probas_mean < 0.5) & (labels == 1)).sum() / (labels == 1).sum())
    
    fig = go.Figure()
    x = list(range(32))
    fig.add_trace(go.Scatter(x=x, y=metrics['acc'], mode='lines', name='Accuracy'))
    fig.add_trace(go.Scatter(x=x, y=metrics['auc'], mode='lines', name='AUC'))
    fig.add_trace(go.Scatter(x=x, y=metrics['TPR'], mode='lines', name='TPR'))
    fig.add_trace(go.Scatter(x=x, y=metrics['TNR'], mode='lines', name='TNR'))
    fig.add_trace(go.Scatter(x=x, y=metrics['FPR'], mode='lines', name='FPR'))
    fig.add_trace(go.Scatter(x=x, y=metrics['FNR'], mode='lines', name='FNR'))
    fig.update_layout(
        title=f"Test on {title} dataset", 
        xaxis_title="Layers",
        yaxis_title="Value",
    )
    fig.show()
    
def results_given_probas(probas, labels): 
    print(f"Accuracy: {accuracy_score(labels, probas > 0.5)}")
    print(f"AUC: {roc_auc_score(labels, probas)}")
    print(f"TPR: {((probas > 0.5) & (labels == 1)).sum() / (labels == 1).sum()}")
    print(f"TNR: {((probas < 0.5) & (labels == 0)).sum() / (labels == 0).sum()}")
    print(f"FPR: {((probas > 0.5) & (labels == 0)).sum() / (labels == 0).sum()}")
    print(f"FNR: {((probas < 0.5) & (labels == 1)).sum() / (labels == 1).sum()}")
    print(f"num errors: {((probas > 0.5) != labels).sum()}")
    
def plot_roc_curves(preds, labels): 
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = sklearn_auc(fpr, tpr)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title(f'ROC Curve Layer 24')
    ax[0].legend()

    # plot 0 to 5% FPR range
    ax[1].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].set_xlim([0, 0.05])
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'ROC Curve Layer 24')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

#* Transfer Attacks
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional, Union

from white_box.monitor import TextMonitor 
from white_box.chat_model_utils import MODEL_CONFIGS, get_template, load_model_and_tokenizer
from white_box.model_wrapper import ModelWrapper

from white_box.dataset import create_prompt_dist_from_metadata_path, ActDataset, ProbeDataset
from white_box.monitor import ActMonitor
from white_box.attacks.prompts import get_universal_manual_prompt


def load_act_monitor(probe_data_path : str, probe_type : str, probe_layer : int = 24, tok_idxs : List[int] = [-1], probe_reg = 1e-1, max_iter = 1000):
    layer = probe_layer
    # if last 3 are 'jb_'
    if probe_data_path[-3:] == 'jb_':
        neg =  create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = "(metadata['jb_name'] == 'harmless')")
        pos = create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = "(metadata['jb_name'] == 'DirectRequest')")
    else: 
        if "all_harmbench_alpaca_" in probe_data_path:
            pos =  create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = "(metadata['label'] == 1) & (metadata.index < 2400)")
            neg =  create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = "(metadata['label'] == 0) & (metadata.index < 2400)")
        else:
            pos =  create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = "(metadata['label'] == 1)")
            neg =  create_prompt_dist_from_metadata_path(f'{probe_data_path}metadata.csv', col_filter = "(metadata['label'] == 0)")
    print(len(pos.idxs), len(neg.idxs))
    dataset = ActDataset([pos], [neg])
    dataset.instantiate()
    probe_dataset = ProbeDataset(dataset)

    if probe_type == "sk":
        acc, auc, probe = probe_dataset.train_sk_probe(layer, tok_idxs = [int(i) for i in tok_idxs], test_size = None, C = probe_reg, max_iter = max_iter, use_train_test_split=False, device='cuda')
    elif probe_type == "mm":
        acc, auc, probe = probe_dataset.train_mm_probe(layer, tok_idxs= [int(i) for i in tok_idxs], test_size=None)
    elif probe_type == "mlp":
        acc, auc, probe = probe_dataset.train_mlp_probe(layer, tok_idxs= [int(i) for i in tok_idxs], test_size=None,
                                                        weight_decay = 1, lr = 0.0001, epochs = 5000)
    print(acc, auc)
    
    monitor = ActMonitor(probe = probe, layer = layer, tok_idxs = [int(i) for i in tok_idxs])
    return monitor

def load_text_monitor():
    model = AutoModelForCausalLM.from_pretrained("data/llama2_7b/llamaguard_harmbench_alpaca__model_0", 
        torch_dtype=torch.float16, 
        device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/LlamaGuard-7b",
        use_fast = False,
        padding_side = "right")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    hb_tm = TextMonitor(model, tokenizer, config_name = "llamaguard")   
    return hb_tm

def load_mw(model_name):
    model_config = MODEL_CONFIGS[model_name]
    model, tokenizer = load_model_and_tokenizer(**model_config)
    template = get_template(model_name, chat_template=model_config.get('chat_template', None))

    mw = ModelWrapper(model, tokenizer, template = template)
    return mw

advbench_behaviors = pd.read_csv("datasets/harmful_behaviors_custom.csv")
advbench_behaviors = advbench_behaviors.sample(frac=1, random_state = 0)

class AttackTransferPortal:
    def __init__(self, am, tm, mw,
                am_log, tm_log):
        self.am = am
        self.tm = tm
        self.mw = mw
        
        self.am_log = am_log
        self.tm_log = tm_log
    
    def eval_on_str(self, prompt):
        state = self.mw.batch_hiddens([prompt], use_chat_template = True)
        am_loss = self.am.get_loss(state['resid'][:, self.am.layer]).item()
        
        self.tm.set_kv_cache(prompt + " {optim_str}", logging = False)
        tm_loss = self.tm.get_loss_no_grad(self.tm.after_ids).item()
        
        return am_loss, tm_loss
     
    def check_transfer_logprob(self, adv_idx, step, transfer_from_am = True, logging = True, prompt_template = 'best_llama2'):
        goal_modified = advbench_behaviors.iloc[adv_idx]['goal']
        target = advbench_behaviors.iloc[adv_idx]['target']

        orig_msg = get_universal_manual_prompt(prompt_template = 'best_llama2', target_str = target, goal = goal_modified.lower())

        if transfer_from_am:
            am_loss_from_log = self.am_log['monitor_losses'][adv_idx][step]
            if logging:
                print(f"AM Monitor Loss in Log {am_loss_from_log}")
            adv_str = self.am_log['optim_strs'][adv_idx][step]
            if isinstance(adv_str, List):
                adv_str = adv_str[0]
        else:
            tm_loss_from_log = self.tm_log['monitor_losses'][adv_idx][step]
            if logging:
                print(f"TM Monitor Loss in Log {tm_loss_from_log}")
            adv_str = self.tm_log['optim_strs'][adv_idx][step]

        prompt = orig_msg + adv_str
        state = self.mw.batch_hiddens([prompt], use_chat_template = True)
        am_loss = self.am.get_loss(state['resid'][:, self.am.layer]).item()
        
        self.tm.set_kv_cache(orig_msg + " {optim_str}", logging = False)
        adv_ids = self.tm.tokenizer(adv_str, return_tensors = 'pt')['input_ids'][:, 1:]
        adv_ids = adv_ids.to(self.tm.after_ids.device)
        ids = torch.cat([adv_ids, self.tm.after_ids.repeat(adv_ids.shape[0], 1)], dim=1)
        tm_loss = self.tm.get_loss_no_grad(ids).item()
        
        if logging:
            print(f"AM Monitor Loss {am_loss}")
            print(f"TM Monitor Loss {tm_loss}")
        
        return am_loss, tm_loss
    
    def check_transfer_gcg(self, adv_idx, step, transfer_from_am : bool = True, logging : bool = True):
        if transfer_from_am:
            # print("Transfer from AM")
            am_loss_from_log = self.am_log['monitor_losses'][adv_idx][step]
            if logging:
                print(f"AM Monitor Loss in Log {am_loss_from_log}")
            
            adv_str =  self.am_log['optim_strs'][adv_idx][step]
            
            if isinstance(adv_str, List):
                adv_str = adv_str[0]
                
            prompt = advbench_behaviors.iloc[adv_idx]['goal'] + " " + adv_str
            # print(prompt)
        else:
            # print("Transfer from TM")
            tm_loss_from_log = self.tm_log['monitor_losses'][adv_idx][step]
            if logging:
                print(f"TM Monitor Loss in Log {tm_loss_from_log}")
            
            adv_str = self.tm_log['optim_strs'][adv_idx][step]
            prompt = advbench_behaviors.iloc[adv_idx]['goal'] + " " + adv_str
            # print(prompt)
            
        state = self.mw.batch_hiddens([prompt])
        am_loss = self.am.get_loss(state['resid'][:, self.am.layer]).item()
        
        self.tm.set_kv_cache(advbench_behaviors.iloc[adv_idx]['goal'] + " {optim_str}", log = False)
        self.tm.after_embeds = self.tm.model.get_input_embeddings()(self.tm.after_ids)

        optim_ids = self.tm.tokenizer(adv_str, return_tensors = 'pt', add_special_tokens=False)['input_ids'].to(self.tm.device)
        monitor_input_ids = torch.cat([optim_ids, self.tm.after_ids.repeat(optim_ids.shape[0], 1)], dim=1)
        optim_embeds = self.tm.model.get_input_embeddings()(monitor_input_ids)
        monitor_input = torch.cat([optim_embeds, self.tm.after_embeds], dim=1)

        tm_loss = self.tm._predict_proba_from_embeds(monitor_input).item()
        
        if logging:
            print(f"AM Monitor Loss {am_loss}")
            print(f"TM Monitor Loss {tm_loss}")
        
        # if transfer_from_am:
        #     # assert np.abs(am_loss - am_loss_from_log) < 0.1
        #     if am_loss_from_log < 0.5:
        #         assert am_loss < 0.5
            
        #     if am_loss < 0.5:
        #         assert am_loss_from_log < 0.5
        # else:
        #     if tm_loss_from_log < 0.5:
        #         assert tm_loss < 0.5
        #     if tm_loss < 0.5:
        #         assert tm_loss_from_log < 0.5
        #     # assert np.abs(tm_loss - tm_loss_from_log) < 0.1
        
        return am_loss, tm_loss
