import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List

from ..monitor import Monitor, ActMonitor, TextMonitor
from ..model_wrapper import slice_acts

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float, monitor : Monitor = None):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
        
class HuggingFace(LanguageModel):
    def __init__(self,model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,
                        monitor : Monitor = None):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        output_hidden_states = True if isinstance(monitor, ActMonitor) else False
        # Batch generation
        if temperature > 0:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
                
                #actmonitor things
                output_hidden_states = output_hidden_states,
                return_dict_in_generate = output_hidden_states
                
            )
        else:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1, # To prevent warning messages
                
                output_hidden_states = output_hidden_states,
                return_dict_in_generate = output_hidden_states
            )
        
        if output_hidden_states:
            output_ids = out['sequences']
        else:
            output_ids = out
            
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        
        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()
            
        if isinstance(monitor, ActMonitor) :
            monitor_input = slice_acts(out, 
                        N_TOKS = 0, 
                        layers = monitor.layer,
                        tok_idxs = monitor.tok_idxs,
                        return_prompt_acts = False)
            monitor_loss = monitor.predict(monitor_input).mean().item()
            
            #TODO: outputs_list needs to be modified to take into account monitor_out
            for i, out in enumerate(outputs_list):
                if out[i] == 1:
                    outputs_list[i] += " [Monitor Output] : Detected Harmful Content"
                else:
                    outputs_list[i] += " [Monitor Output] : Did Not Detect Harmful Content"
            return outputs_list
        elif isinstance(monitor, TextMonitor):
            raise NotImplementedError
        else:
            return outputs_list

    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = openai.ChatCompletion.create(
                            model = self.model_name,
                            messages = conv,
                            max_tokens = max_n_tokens,
                            temperature = temperature,
                            top_p = top_p,
                            request_timeout = self.API_TIMEOUT,
                            )
                output = response["choices"][0]["message"]["content"]
                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,
                        monitor : Monitor = None):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Claude():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
   
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model= anthropic.Anthropic(
            api_key=self.API_KEY,
            )

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.completions.create(
                    model=self.model_name,
                    max_tokens_to_sample=max_n_tokens,
                    prompt=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.completion
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,
                        monitor : Monitor = None):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]