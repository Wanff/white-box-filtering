from openai import OpenAI
import openai
import anthropic
import os
from dotenv import load_dotenv
import copy
import time
import torch
import gc
from typing import Dict, List

load_dotenv()

from ..monitor import Monitor, ActMonitor, TextMonitor
# from white_box.monitor import Monitor, ActMonitor, TextMonitor
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
        self.pos_to_token_dict = {v: k.replace('▁', ' ') for k, v in self.tokenizer.get_vocab().items()}


    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,
                        monitor : Monitor = None,
                        adv_ids = None):
        if 'llama2' in self.model_name.lower():
            max_n_tokens += 1  # +1 to account for the first special token (id=29871) for llama2 models
        batch_size = len(full_prompts_list)
        vocab_size = len(self.tokenizer.get_vocab())
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        output_hidden_states = True if isinstance(monitor, ActMonitor) else False

        # Batch generation
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_n_tokens,  
            min_new_tokens = max_n_tokens,
            do_sample=False if temperature == 0 else True,
            temperature=None if temperature == 0 else temperature,
            eos_token_id=self.eos_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,  # added for Mistral
            top_p=top_p,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states = output_hidden_states,
        )
        output_ids = output.sequences
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, input_ids.shape[1]:]  
        if 'llama2' in self.model_name.lower():
            output_ids = output_ids[:, 1:]  # ignore the first special token (id=29871)

        generated_texts = self.tokenizer.batch_decode(output_ids)
        
        #* LOGPROB STUFF
        # output.scores: n_output_tokens x batch_size x vocab_size (can be counter-intuitive that batch_size doesn't go first)
        logprobs_tokens = [torch.nn.functional.log_softmax(output.scores[i_out_token], dim=-1).cpu().numpy() 
                           for i_out_token in range(len(output.scores))]
        if 'llama2' in self.model_name.lower():
            logprobs_tokens = logprobs_tokens[1:]  # ignore the first special token (id=29871)

        logprob_dicts = [[{self.pos_to_token_dict[i_vocab]: logprobs_tokens[i_out_token][i_batch][i_vocab]
                         for i_vocab in range(vocab_size)} 
                         for i_out_token in range(len(logprobs_tokens))
                        ] for i_batch in range(batch_size)]
        outputs = [{'text': generated_texts[i_batch],
                    'logprobs': logprob_dicts[i_batch],
                    'n_input_tokens': len(input_ids[i_batch][input_ids[i_batch] != 0]),  # don't count zero-token padding
                    'n_output_tokens': len(output_ids[i_batch]),
                   } for i_batch in range(batch_size)
        ]
        
        #* monitor stuff
        if monitor is not None:
            if isinstance(monitor, ActMonitor):
                monitor_input = slice_acts(output, 
                            N_TOKS = max_n_tokens, 
                            layers = monitor.layer,
                            tok_idxs = torch.tensor(monitor.tok_idxs) - max_n_tokens if monitor.monitor_type == "input" else monitor.tok_idxs,
                            return_prompt_acts = True if monitor.monitor_type == "input" else False)
                monitor_losses = monitor.get_loss(monitor_input)
            elif isinstance(monitor, TextMonitor):
                # print(adv_ids)
                adv_ids = adv_ids.to(monitor.after_ids.device)
                ids = torch.cat([adv_ids, monitor.after_ids.repeat(adv_ids.shape[0], 1)], dim=1)
                # print(self.tokenizer.batch_decode(ids))
                monitor_losses = monitor.get_loss_no_grad(ids)
            
            # print(monitor_losses)
            for i, out in enumerate(outputs):
                outputs[i]['monitor_loss'] = monitor_losses[i].item()
                
        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs

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
    
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

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
        response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = conv,
                    max_tokens = max_n_tokens,
                    temperature = temperature,
                    top_p = top_p,
                    
                    )
        output = response.choices[0].message.content

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