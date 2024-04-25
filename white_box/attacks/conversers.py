import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template

from .lms import GPT, Claude, HuggingFace
from .prompts import get_judge_system_prompt
from .utils import extract_json, conv_template

from ..monitor import Monitor, ActMonitor, TextMonitor

ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1

def load_attack_and_target_models(attack_model : str, attack_max_n_tokens : int, max_n_attack_attempts : int,
                                 target_model : str, target_max_n_tokens : int, 
                                 monitor : Monitor = None ):
    # Load attack model and tokenizer
    attackLM = AttackLM(model_name = attack_model, 
                        max_n_tokens = attack_max_n_tokens, 
                        max_n_attack_attempts = max_n_attack_attempts, 
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        )
    preloaded_model = None
    if attack_model ==target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attackLM.model
    targetLM = TargetLM(model_name = target_model, 
                        max_n_tokens = target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        monitor = monitor,
                        )
    return attackLM, targetLM

def load_target_model(args, preloaded_model = None, monitor = None):
    targetLM = TargetLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        preloaded_model = preloaded_model,
                        monitor = monitor,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        )
    return targetLM

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                temperature: float,
                top_p: float):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        
        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs 
            outputs_list = self.model.batched_generate(full_prompts_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )['text']
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = extract_json(full_output)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs

class TargetLM():
    """
        Base class for target language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            temperature: float,
            top_p: float,
            preloaded_model: object = None,
            monitor : Monitor = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)
        
        self.monitor = monitor
        
        self.n_input_tokens = 0
        self.n_output_tokens = 0
        self.n_input_chars = 0
        self.n_output_chars = 0

    def get_response(self, prompts_list, max_n_tokens : int = None, temperature : float = None):
        batchsize = len(prompts_list)
        convs_list = [conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())
        
        outputs = self.model.batched_generate(full_prompts, 
                                                        max_n_tokens = self.max_n_tokens if max_n_tokens is None else max_n_tokens,  
                                                        temperature = self.temperature if temperature is None else temperature,
                                                        top_p = self.top_p,
                                                        monitor = self.monitor
                                                    )
        self.n_input_tokens += sum(output['n_input_tokens'] for output in outputs)
        self.n_output_tokens += sum(output['n_output_tokens'] for output in outputs)
        self.n_input_chars += sum(len(full_prompt) for full_prompt in full_prompts)
        self.n_output_chars += len([len(output['text']) for output in outputs])
        return outputs



def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        lm = GPT(model_name)
    elif model_name in ["claude-2", "claude-instant-1"]:
        lm = Claude(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,device_map="auto").eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        ) 

        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":'lmsys/vicuna-13b-v1.5',
            "template":"vicuna_v1.1"
        },
        "vicuna-7b":{
            "path":'lmsys/vicuna-7b-v1.5',
            "template":"vicuna_v1.1"
        },
        "llama-2":{
            "path":'meta-llama/Llama-2-7b-chat-hf',
            "template":"llama-2"
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template


