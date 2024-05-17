import re
from typing import List, Union, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import Levenshtein
import datasets
from datasets import load_dataset
from tqdm import tqdm


def untuple(x):
    return x[0] if isinstance(x, tuple) else x
    
def ceildiv(a, b):
    #https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)

def gen_pile_data(N : int, tokenizer : AutoTokenizer, min_n_toks : int = None, max_n_toks : int = None, split='train'): 
    """
    supports : min_n_toks and max_n_toks, truncates text if it is too long, and rejects if too short
    just max_n_toks, truncates text if it is too long
    just min_n_toks, rejects if too short
    if none, accept everything 
    
    Args:
        N (int): _description_
        tokenizer (AutoTokenizer): _description_
        min_n_toks (int, optional): _description_. Defaults to None.
        max_n_toks (int, optional): _description_. Defaults to None.
        split (str, optional): _description_. Defaults to 'train'.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if split == 'train': 
        pile = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
    elif split == 'validation':
        pile = load_dataset('monology/pile-uncopyrighted', split='validation', streaming=True)
    else: 
        raise Exception("Invalid split")

    sentences = []

    counter = 0
    for i, example in enumerate(pile):
        if counter == N:
            break
        
        toks = tokenizer(example['text'])['input_ids']
        
        if min_n_toks is not None:
            if len(toks) < min_n_toks:
                continue
            else:
                text = example['text']
            
        if max_n_toks is not None:
            if len(toks) > max_n_toks:
                toks = toks[:max_n_toks]     
                text = tokenizer.decode(toks)
            else:
               text = example['text']
            
        sentences.append(text)
        counter +=1
        
    return sentences

def tpr_at_fpr(probs, labels, target_fpr, left=0.5, right=1.0, max_steps=1000, thresh_tol=1e-6): 
    """
    Calculates the true positive rate at a given false positive rate. 
    Does up to max_steps steps of binary search, returns the best guess 
    that yields a false positive rate less than target_fpr
    
    probs: (n_examples, ) just prob on positive class
    labels: (n_examples, ) 0 or 1
    """
    assert len(probs) == len(labels)
    assert probs.shape == labels.shape
    assert probs.shape[0] > 0

    for _ in range(max_steps):
        mid = (left + right) / 2
        
        # calc fpr 
        preds = (probs > mid).astype(int)
        fp = np.logical_and(preds == 1, labels == 0).sum()
        tn = (labels == 0).sum()
        fpr = fp / tn if tn > 0 else 0

        if abs(fpr - target_fpr) < thresh_tol: 
            right = mid
            break
        elif fpr > target_fpr:
            left = mid
        else:
            right = mid
    
    # use right as threshold to ensure fpr <= target_fpr
    preds = (probs > right).astype(int)
    tp = np.logical_and(preds == 1, labels == 1).sum()
    fn = (labels == 1).sum()
    return tp / fn if fn > 0 else 0


# sim_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# def sim_scores(outputs, targets):
#     semantic_scores_gen = []
#     for target, output in zip(targets, outputs):
#         embedding1 = sim_model.encode(target, convert_to_tensor=True)
#         embedding2 = sim_model.encode(output, convert_to_tensor=True)
#         cosine_sim_gen = util.pytorch_cos_sim(embedding1, embedding2)
#         similarity_value_gen = cosine_sim_gen.item()
#         semantic_scores_gen.append(similarity_value_gen)
    
#     return semantic_scores_gen 

def char_by_char_similarity(outputs, targets):
    similarities = []
    for o, t in zip(outputs, targets):
        o = re.sub(r'\s', '', o)
        t = re.sub(r'\s', '', t)

        o = o.lower()
        t = t.lower()

        # remove '<|endoftext|>'
        o = o.replace('<|endoftext|>', '')
        t = t.replace('<|endoftext|>', '')

        max_len = max(len(o), len(t))
        matches = [c1 == c2 for c1, c2 in zip(o, t)]
        
        similarities.append(sum(matches)/max_len if max_len > 0 else 0)
    return similarities

def compare_token_lists(genned_toks, ground_toks):
    if len(ground_toks) < len(genned_toks):
        
        num_same_tokens = sum(1 for token1, token2 in zip(ground_toks, genned_toks[:len(ground_toks)]) if token1 == token2)
        percent_same_tokens = (num_same_tokens / len(ground_toks)) 
        return percent_same_tokens
    elif len(ground_toks) > len(genned_toks):
        raise ValueError(f"Ground truth is longer than generated text. This should not happen. (ground_len, gen_len) : {len(ground_toks), len(genned_toks)}")        
    else:
        num_same_tokens = sum(1 for token1, token2 in zip(ground_toks, genned_toks) if token1 == token2)
        percent_same_tokens = (num_same_tokens / len(ground_toks)) 
        
        return percent_same_tokens

def tok_by_tok_similarity(outputs : Union[List[str], List[List[int]]], targets : Union[List[str], List[List[int]]], tokenizer = None):
    if isinstance(outputs[0], str):
        assert tokenizer is not None
        if tokenizer.padding_side == "left":
            print("Are you sure you want padding_side == left?")
            
        outputs = tokenizer(outputs, return_tensors = 'pt',padding = True, truncation = True)['input_ids']
        targets = tokenizer(targets, return_tensors = 'pt',padding = True)['input_ids']
        max_length = max(outputs.shape[1], targets.shape[1])

        if outputs.shape[1] < max_length:
            outputs = torch.cat([outputs, torch.zeros(outputs.shape[0], max_length - outputs.shape[1], dtype = torch.long)], dim = 1)
        if targets.shape[1] < max_length:
            targets = torch.cat([targets, torch.zeros(targets.shape[0], max_length - targets.shape[1], dtype = torch.long)], dim = 1)
            
    return [compare_token_lists(t, o) for t, o in zip(outputs, targets)]

def levenshtein_distance(outputs, targets):
    diss = []
    for o, t in zip(outputs, targets):
        max_len = max(len(o), len(t))
        diss.append((max_len - Levenshtein.distance(o, t)) / max_len)
    return diss

def extract_quote_completion(s):
    s = s.replace(";",",").split(".")[0].split("\n")[0]
    return s.strip().lower()


def eval_completions(outputs : Union[List[str], List[List[int]]], 
                     targets : Union[List[str], List[List[int]]], 
                     sim_types = ['char', 'tok', 'lev', 'sem'], tokenizer = None, return_mean = False):
    """Pass in tokenizer for tok_by_tok_similarity if outputs and targets are strings

    Args:
        outputs (_type_): _description_
        targets (_type_): _description_
        sim_types (list, optional): _description_. Defaults to ['char', 'tok', 'lev', 'sem'].
        tokenizer (_type_, optional): _description_. Defaults to None.
        return_mean (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    return_dict = {}
    if 'char' in sim_types:
        cbc_sims = char_by_char_similarity(outputs, targets)
        return_dict['char_by_char_sim'] = cbc_sims
    
    if 'tok' in sim_types:
        tbt_sims = tok_by_tok_similarity(outputs, targets, tokenizer = tokenizer)
        return_dict['tok_by_tok_sim'] = tbt_sims
    
    if 'lev' in sim_types:
        lev_diss = levenshtein_distance(outputs, targets)
        return_dict['lev_distance'] = lev_diss
    
    if 'sem' in sim_types:
        sem_sims = sim_scores(outputs, targets)
        return_dict['sem_sim'] = sem_sims
    
    if return_mean:
        return {k: np.mean(v) for k, v in return_dict.items()}
   
    return return_dict


def dataset_wide_deduplicate(dataset, col): 
    """
    Takes in a huggingface dataset, outputs it with all rows deduplicated
    depending on the column col
    """
    dataset = dataset.to_pandas()
    dataset = dataset.drop_duplicates(subset = col)
    return datasets.Dataset.from_pandas(dataset)

def rotation_cipher(text, shift): 
    """
    Rotates the text by shift amount, character-wise. Also 
    known as the Caesar cipher.
    """
    shifted_text = []
    for char in text: 
        if char.isalpha():
            shifted_text.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
        else:
            shifted_text.append(char)
    return ''.join(shifted_text)

@torch.no_grad()
def get_batched_preds(prompts: List[str], model: torch.nn.Module, tokenizer: AutoTokenizer, template: str, device: str, batch_size: int = 8) -> np.ndarray:
    
    preds = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        
        current_batch_prompts = prompts[i:i+batch_size]
        current_batch_prompts = [template.format(instruction=prompt) for prompt in current_batch_prompts]
        toks = tokenizer(current_batch_prompts, return_tensors='pt', padding=True, truncation=True)
        last_token_idxs = toks['attention_mask'].sum(1) - 1
        
        output = model(**toks.to(device))   
        preds.append(torch.stack([output.logits[torch.arange(len(current_batch_prompts)), last_token_idxs, 9109], output.logits[torch.arange(len(current_batch_prompts)), last_token_idxs, 25110]], dim=1).softmax(-1).cpu().detach().numpy()[:, 1])
        del toks
        del output
        torch.cuda.empty_cache()

    return np.concatenate(preds)

