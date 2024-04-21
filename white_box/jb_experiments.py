import numpy as np 
import pandas as pd
from tqdm import tqdm 

import plotly.graph_objects as go
import plotly.express as px

from white_box.dataset import PromptDist, ActDataset, create_prompt_dist_from_metadata_path, ProbeDataset

#* EXPERIMENTS
# Leave-one-out generalization
def leave_one_out_generalization(jb_metadata : pd.DataFrame, split : str = "category"):
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

def plot_dual_metric_by_category(res, layer, primary_metric, secondary_metric):
    categories = []
    primary_values = []
    secondary_values = []
    
    for cat, data in res.items():
        categories.append(cat)
        primary_values.append(np.round(data[primary_metric][layer], 3))
        secondary_values.append(np.round(data[secondary_metric][layer], 3))
    
    # Create bar plots for both metrics
    fig = go.Figure(data=[
        go.Bar(name=primary_metric, x=categories, y=primary_values, text=primary_values, textposition='auto', marker_color='blue'),
        go.Bar(name=secondary_metric, x=categories, y=secondary_values, text=secondary_values, textposition='auto', marker_color='red')
    ])
    
    # Update layout to have the bars grouped
    fig.update_layout(
        barmode='group',
        title=f'Comparison of {primary_metric} and {secondary_metric} at Layer {layer}',
        xaxis_title='Category',
        yaxis_title='Metric Value',
        template='plotly_white'
    )
    
    # Show plot
    fig.show()
    