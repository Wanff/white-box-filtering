from collections import Counter
import re 
from typing import List, Union, Tuple, Dict, Callable 
import time
import pandas as pd
import requests
from tqdm import tqdm 
import torch 
import numpy as np
import json 

from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
from dataclasses import dataclass

from white_box.utils import eval_completions, ceildiv, gen_pile_data
from white_box.model_wrapper import ModelWrapper

#* clean memorized data 
def clean_data(data : Union[List[str], List[int]], tokenizer : AutoTokenizer, 
               return_toks : bool = False,
               ) -> Tuple[Union[List[str], List[int]], Dict[str, List]]:
    
    if isinstance(data[0], str) and tokenizer:
        toks = tokenizer(data, return_tensors = 'pt', padding = True, truncation = True, max_length = 64)['input_ids'].tolist()
        strings = data
    elif isinstance(data[0][0], int):
        toks = data
        strings = tokenizer.batch_decode(data)
    else:
        raise ValueError("Data must be a list of strings or a list of integers")
    
    assert len(toks) == len(strings)
    
    filtered_data = []
    dirty_dict = {
        "increment" : [],
        "repeated_majority" : [],
        "is_repeated_string" : [],
        "repeats_subseq" : []
    }
    for tok, string in zip(toks, strings):
        if check_increment(string):
            dirty_dict["increment"].append(string)
            continue 
        if check_repeated_majority(tok):
            dirty_dict["repeated_majority"].append(string)
            continue
        if check_is_repeated_string(string):
            dirty_dict["is_repeated_string"].append(string)
            continue
        if check_repeats_subseq(tok):
            dirty_dict["repeats_subseq"].append(string)
            continue
    
        if return_toks:
            filtered_data.append(tok)
        else:
            filtered_data.append(string)
    
    filtered_data = remove_duplicates(filtered_data)
    
    return filtered_data, dirty_dict 

def remove_duplicates(data : Union[List[str], List[int]]) -> Union[List[str], List[int]]:
    if isinstance(data[0], str):
        return list(set(data))
    elif isinstance(data[0][0], int):
        return list(set([tuple(x) for x in data]))
    else:
        raise ValueError("Data must be a list of strings or a list of integers")
    
# remove common patterns
def check_repeated_majority(toks : List[int], frac : float = 1/2):
    """
    Check if any value in the array is repeated more than half the length of the array.

    Parameters:
    toks (list): The input list of numbers.

    Returns:
    bool: True if a majority element exists, False otherwise.
    """
    if not toks:
        return False

    count = Counter(toks)
    length = len(toks)

    for key, value in count.items():
        if value > length * frac:
            return True

    return False

def remove_punc(string):
    return re.sub(r'[\[\]._,\-]', '', string)

def remove_non_numeric(string):
    return re.sub(r'[^\d]', '', string)

def check_is_repeated_string(string : str):
    # checks if a string is repeated after removing periods, underscores, dashes, and other special characters
    string = remove_punc(string)
    words = string.split()
    if len(words) == 0:
        return False
    else:
        return all(words[i] == words[i+1] for i in range(len(words)-1))
    
def check_increment(string : str):
    # Remove periods, underscores, dashes, and other special characters
    numbers = [int(remove_punc(word.strip())) for word in string.split() if remove_punc(word.strip()).isdigit()]
    if len(numbers) < 4:
        return False
    else:
        all_inc = all(numbers[i] == numbers[i+1] - 1 for i in range(len(numbers)-1)) 
        all_dec = all(numbers[i] ==  numbers[i+1] + 1 for i in range(len(numbers)-1))
        return all_inc or all_dec 

def check_repeats_subseq(toks : List[int], n=10): 
    """
    Returns True if has a tok seq contains a subsequence of length n that is repeated more than once. 
    """
    for i in range(len(toks) - n):
        subseq = toks[i:i+n]
        for j in range(i+1, len(toks) - n):
            if toks[j:j+n] == subseq:
                return True
    return False

#* data classes
@dataclass
class PromptDist:
    idxs : List[int]
    path_to_states : str
    path_to_metadata : str

def create_prompt_dist_from_metadata_path(path_to_metadata : str,
                                     condition : Callable,
                                     col_name : str = 'tok_by_tok_sim',
                                     ) -> PromptDist:
    metadata = pd.read_csv(path_to_metadata)
    path_to_states = path_to_metadata.replace('metadata.csv', 'hidden_states.pt')
    idxs = metadata[condition(metadata[col_name])].index.tolist()
    
    return PromptDist(idxs = idxs, path_to_states = path_to_states, path_to_metadata = path_to_metadata)

def create_prompt_dist_from_metadata(metadata : pd.DataFrame, 
                                     path_to_states : str, 
                                     path_to_metadata : str,
                                     condition : Callable,
                                     col_name : str = 'tok_by_tok_sim',
                                     ) -> PromptDist:
    
    idxs = metadata[condition(metadata[col_name])].index.tolist()
    
    return PromptDist(idxs = idxs, path_to_states = path_to_states, path_to_metadata = path_to_metadata)

def less_than_60_percent(x):
    return x < 0.6

def equal_100_percent(x):
    return x == 1

class ActDataset:
    def __init__(self, pos : Union[PromptDist, List[PromptDist]], neg : Union[PromptDist, List[PromptDist]]):
        if not isinstance(pos, List):
            pos = [pos]
        if not isinstance(neg, List):
            neg = [neg]
        
        self.pos = pos
        self.neg = neg
        
        self.X = None
        self.y = None
    
    def instantiate(self):
        if self.pos:
            pos_states = torch.cat([torch.load(x.path_to_states)[x.idxs] for x in self.pos])
        else:
            pos_states = torch.tensor([])
        
        if self.neg:
            neg_states = torch.cat([torch.load(x.path_to_states)[x.idxs] for x in self.neg])
        else:
            neg_states = torch.tensor([])
        
        self.X = torch.cat([pos_states, neg_states]).cpu().float()
        self.y = torch.cat([torch.ones(len(pos_states)), torch.zeros(len(neg_states))])
        return self.X, self.y
    
    def train_test_split(self,
                        test_size : float = 0.2, 
                        layer : int = None,
                        tok_idxs : List[int] = None,
                        random_state : int = 0,
                        balanced : bool = True):
        
        assert self.X is not None, "You must instantiate the dataset first"
        
        if tok_idxs is not None:
            labels = self.y.view(-1, 1).expand(-1, len(tok_idxs)).flatten()
            if layer is not None:
                states = self.X[:, layer, tok_idxs].reshape(-1, self.X.shape[3])
            else:
                states = self.X[:, :, tok_idxs].reshape(-1, self.X.shape[3])
        else:
            labels = self.y
            states = self.X
        
        train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size = test_size, random_state = random_state, stratify = labels if balanced else None)
        
        train_states = states[train_indices]
        test_states = states[test_indices]

        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        
        return train_states, test_states, train_labels, test_labels


#* generalization datasets

def gen_fuzzy_pos_dataset(metadata_paths : Union[List[str], str]) -> ActDataset: 
    """
    Generate a dataset of fuzzily memorized text by filtering for rows that have tok_by_tok_similarity between [0.9, 1). 
    """
    def between_90_and_100_percent(x):
        return (0.9 <= x) & (x < 1)
    
    if isinstance(metadata_paths, str):
        metadata_paths = [metadata_paths]
        
    prompt_dists = [create_prompt_dist_from_metadata_path(metadata_path, between_90_and_100_percent) for metadata_path in metadata_paths]
    fuzzy_pos_dataset = ActDataset(prompt_dists, [])
    fuzzy_pos_dataset.instantiate()
    
    return fuzzy_pos_dataset

def get_quotes(save_path = '../data/all_quotes.json'):
    """generates a bunch of quotes for the quotes dataset, starting off with a quotes.json file taken from here:
    https://github.com/andyzoujm/representation-engineering/blob/main/data/memorization/quotes/popular_quotes.json

    Args:
        save_path (str, optional): Defaults to '../data/quotes.json'.
    """
    all_quotes = []
    for i in tqdm(range(500)):
        quotes = requests.get('https://zenquotes.io/api/quotes').json()
        all_quotes.extend([q['q'] for q in quotes])
        time.sleep(1)
        
        if i % 10 == 0:
            time.sleep(7)
            
    all_quotes = [quote for quote in all_quotes if quote != "Too many requests. Obtain an auth key for unlimited access."]
    
    prev_quotes = json.load(open(save_path, 'rb'))
    
    all_quotes = list(set(prev_quotes + all_quotes))
    
    with open(save_path, 'w') as file:
        json.dump(all_quotes, file, indent=4)        
        
    return all_quotes


def gen_quotes_metadata(mw : ModelWrapper,
                        path_to_quotes : str = "../data/quotes.json") -> pd.DataFrame:
    all_quotes = json.load(open(path_to_quotes, 'rb'))
    
    quote_tok_out = mw.tokenizer(all_quotes, return_tensors = 'pt', padding = True, truncation = True, max_length = 64)

    quote_lens = quote_tok_out['attention_mask'].sum(dim = 1)
    quote_lens_set = set(quote_lens.tolist())
    
    quote_lens_w_index = np.array([(i, len_) for i, len_ in enumerate(quote_lens)])
    
    df = {
        'quote_toks' : [],
        'quote_strs' : [],
        'gen_toks' : [],
        'gen_strs' : [],
        'tok_by_tok_sim' : [],
        'char_by_char_sim' : [],
        'lev_distance' : [],
        'quote_len' : []
    }
    
    for quote_len in tqdm(quote_lens_set):
        quote_indices = quote_lens_w_index[quote_lens_w_index[:,1] == quote_len][:,0]
        quote_toks = torch.stack([quote_tok_out['input_ids'][i] for i in quote_indices])
        quote_strs = [all_quotes[i] for i in quote_indices]
        
        quotes_first_half = quote_toks[:,: ceildiv(quote_len, 2)]
        
        out = mw.batch_generate_autoreg(quotes_first_half, max_new_tokens = quote_len // 2, output_tokens = True)
        gen_toks = out['tokens'].tolist()
        gen_strs = out['generations']
        evals = eval_completions(gen_strs, quote_strs, sim_types = ['char', 'tok', 'lev'], tokenizer = mw.tokenizer, return_mean = False)
        
        df['quote_toks'].extend(quote_toks.tolist())
        df['quote_strs'].extend(quote_strs)
        df['gen_toks'].extend(gen_toks)
        df['gen_strs'].extend(gen_strs)
        df['tok_by_tok_sim'].extend(evals['tok_by_tok_sim'])
        df['char_by_char_sim'].extend(evals['char_by_char_sim'])
        df['lev_distance'].extend(evals['lev_distance'])
        df['quote_len'].extend([quote_len]*len(quote_indices))
    
    return pd.DataFrame.from_dict(df)
    
def gen_prefix_dataset_strings(prompts : List[str], tokenizer : AutoTokenizer, prefix_len : int = 50) -> List[str]:
    """
    Generate a dataset of prefixed data by taking the normal dataset and adding prefixes from the pile to the data
    """
    pile_data = gen_pile_data(N = len(prompts), tokenizer = tokenizer, 
                              min_n_toks = prefix_len, 
                              max_n_toks = prefix_len, 
                              split = "validation")
    
    return [pile_prefix + prompt  for prompt, pile_prefix in zip(prompts, pile_data)]
