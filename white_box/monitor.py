import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Union

from white_box.probes import Probe, LRProbe, MLP, MMProbe

class Monitor:
    def predict_proba(self, data):
        raise NotImplementedError
    
    def get_loss(self, data):
        raise NotImplementedError
    

#TODO: need to create generic textmonitor and actmonitor classes, and then the pca/mm/mlp stuff should subclass actmonitor and the llamaguard/oai mod/harmbench should subclass textmonitor

class TextMonitor(Monitor):
    def __init__(self, model : AutoModelForCausalLM | AutoModelForSequenceClassification, tokenizer : AutoTokenizer, 
                 score_id : int,
                 use_kv_cache : bool = True,
                 monitor_type : str = "input", 
                 model_type : str = 'causal_lm'):
        #LLAMA Guard
        self.model = model
        self.tokenizer = tokenizer
        self.score_id = score_id
        
        self.before_str = None
        self.after_str = None 
        self.monitor_type = monitor_type
        self.model_type = model_type
        
        if use_kv_cache:
            chat = [{"role": "user", "content": ""}]
            input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device) 
            pre_instruction_ids = input_ids[:, :826]
            with torch.no_grad():
                output = model(pre_instruction_ids, use_cache=True)
                self.kv_cache = output.past_key_values

    def _predict_proba(self, prompt : str):
        chat = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
        
        if self.model_type == 'causal_lm':
            output = self.model.generate(input_ids=input_ids, max_new_tokens=5,
                                     output_scores = True, 
                                     past_key_values = self.kv_cache if self.kv_cache is not None else None,
                                     return_dict_in_generate = True, 
                                     pad_token_id=0)
            proba = torch.softmax(output['scores'][0], dim = 1)[:, self.score_id].squeeze()
        elif self.model_type == 'sequence_classification':
            output = self.model(input_ids, past_key_values = self.kv_cache if self.kv_cache is not None else None)
            proba = output.logits.softmax(dim=-1)[0,1]
        
        return proba

    def _predict_proba_from_embeds(self, embeds: torch.Tensor): 
        if self.model_type == 'causal_lm': 
            output = self.model(inputs_embeds=embeds,
                                        past_key_values=self.kv_cache if self.kv_cache is not None else None)
            proba = output.logits.softmax(-1)[:,-1,self.score_id].squeeze()
        elif self.model_type == 'sequence_classification': 
            output = self.model(inputs_embeds=embeds, past_key_values = self.kv_cache if self.kv_cache is not None else None)
            proba = output.logits.softmax(dim=-1)[0,1]
        
        return proba
    
    def predict_proba(self, prompts : Union[str, List[str]]):
        if isinstance(prompts, str):
            return self._predict_proba(prompts)
        else:
            probas =  []
            for prompt in prompts:
                probas.append(self._predict_proba(prompt))

            return torch.tensor(probas, device = self.model.device)
    
    def get_loss(self, embeds : torch.Tensor): 
        return self._predict_proba_from_embeds(embeds)

    def get_loss_no_grad(self, prompt : str):
        return self.predict_proba(prompt)
    
    def set_before_after_str(self, before_str : str, after_str: str):
        self.before_str = before_str
        self.after_str = after_str
    
    def get_monitor_input(self, optim_strs : Union[str, torch.Tensor, List[str]]) -> List[str]:
        assert self.before_str is not None and self.after_str is not None

        if isinstance(optim_strs, str):
            optim_strs = [optim_strs]
        elif isinstance(optim_strs, torch.Tensor):
            optim_strs = self.tokenizer.batch_decode(optim_strs)
        
        return [self.before_str + optim_str + self.after_str for optim_str in optim_strs]

    
class ActMonitor():
    def __init__(self, probe : Probe, layer : int, tok_idxs : List[int], monitor_type : str = "input"):
        self.probe = probe
        self.layer = layer
        self.tok_idxs = tok_idxs
        self.monitor_type = monitor_type 

    def predict_proba(self, acts):
        return self.probe.predict_proba(acts.float())

    def get_loss(self, acts : torch.tensor):
        if len(self.tok_idxs) > 1:
            return self.predict_proba(acts.float()).mean(dim = -1)
        else:
            return self.predict_proba(acts.float())

    def get_loss_no_grad(self, acts : torch.Tensor): 
        with torch.no_grad(): 
            return self.get_loss(acts)