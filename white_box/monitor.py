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
                 monitor_type : str = "input"):
        #LLAMA Guard
        self.model = model
        self.tokenizer = tokenizer
        self.score_id = score_id
        
        self.before_str = None
        self.after_str = None 
        self.monitor_type = monitor_type
        
    def set_kv_cache(self, goal_str : str): 
        inst_idx = 831
        chat = [{"role": "user", "content": ""}]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device) 
        goal_ids = self.tokenizer(goal_str, return_tensors="pt").input_ids.to(self.model.device)[:, 1:] #the indexing is to get rid of the <s> token
        pre_instruction_ids = torch.cat([input_ids[:, :inst_idx], goal_ids], dim=1)

        self.after_ids = input_ids[:,inst_idx:]
        self.after_str = self.tokenizer.decode(input_ids[0,inst_idx:])

        print(self.tokenizer.decode(pre_instruction_ids[0]))
        print("BEGIN AFTER")
        print(self.after_str)

        with torch.no_grad():
            output = self.model(pre_instruction_ids, use_cache=True)
            self.kv_cache = output.past_key_values

    def _predict_proba(self, input_ids : torch.Tensor):
        """This function assumes you're already using a kvcache that has the llamaguard instruction string set
        """
        # chat = [{"role": "user", "content": prompt}]
        # input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
        
        batch_size = input_ids.shape[0]
        if self.kv_cache is not None:
            kv_cache_batch = []
            for i in range(len(self.kv_cache)):
                kv_cache_batch.append([])
                for j in range(len(self.kv_cache[i])):
                    kv_cache_batch[i].append(self.kv_cache[i][j].expand(batch_size, -1, -1, -1))
                    
            output = self.model(input_ids=input_ids,
                                past_key_values = kv_cache_batch)
            proba = torch.stack([output.logits[:,-1, 9109], output.logits[:,-1, 25110]], dim=1).softmax(-1)
            proba = proba[:,1]
    
        return proba

    def _predict_proba_from_embeds(self, embeds: torch.Tensor): 
        #this function does not take in batched embeds
        
        output = self.model(inputs_embeds=embeds,
                                    past_key_values=self.kv_cache if self.kv_cache is not None else None)
        proba = torch.stack([output.logits[:,-1, 9109], output.logits[:,-1, 25110]], dim=1).softmax(-1)
        proba = proba[0,1]
        
        return proba
    
    def get_loss(self, embeds : torch.Tensor): 
        return self._predict_proba_from_embeds(embeds.type(self.model.dtype))

    def get_loss_no_grad(self, input_ids : torch.Tensor):
        return self._predict_proba(input_ids)
    
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
        # with torch.no_grad(): 
        return self.get_loss(acts)