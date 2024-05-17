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