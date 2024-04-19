import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List 

from white_box.probes import Probe, LRProbe, MLP, MMProbe

class Monitor:
    def predict_proba(self, data):
        raise NotImplementedError
    
    def get_loss(self, data):
        raise NotImplementedError

class TextMonitor(Monitor):
    def __init__(self, model : AutoModelForCausalLM, tokenizer : AutoTokenizer, 
                 score_id : int,
                 use_kv_cache : bool = True):
        
        self.model = model
        self.tokenizer = tokenizer
        self.score_id = score_id
        
        if use_kv_cache:
            chat = [{"role": "user", "content": ""}]
            input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device) 
            with torch.no_grad():
                output = model(input_ids, use_cache=True)
                self.kv_cache = output.past_key_values

    def predict_proba(self, prompt : str):
        chat = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)

        output = self.model.generate(input_ids=input_ids, max_new_tokens=5, 
                                     output_scores = True, 
                                     past_key_values = self.kv_cache if self.kv_cache is not None else None,
                                     return_dict_in_generate = True, 
                                     pad_token_id=0)
        
        return torch.softmax(output['scores'][0], dim = 1)[:, self.score_id]
    
    def get_loss(self, prompt : str):
        return self.predict_proba(prompt)
    
class ActMonitor():
    def __init__(self, probe : Probe, layer : int, tok_idxs : List[int]):
        self.probe = probe
        self.layer = layer
        self.tok_idxs = tok_idxs

    def predict_proba(self, acts):
        return self.probe.predict_proba(acts)

    def get_loss(self, acts : torch.tensor):
        return self.predict_proba(acts)