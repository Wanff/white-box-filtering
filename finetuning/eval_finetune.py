import sys 
sys.path.append('../')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, AutoModelForCausalLM
import torch
import os
import numpy as np
from tqdm import tqdm
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS
from datasets import load_from_disk, DatasetDict, load_dataset
import argparse

from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from trl import SFTTrainer

path = '/data/oam_patel/white-box-filtering/data/ciphers/llama2_7b'
model_config = MODEL_CONFIGS['llama2_7b']
model, tokenizer = load_model_and_tokenizer(**model_config)
model.config.use_cache = False
tokenizer.pad_token = tokenizer.eos_token
template = get_template('llama2_7b', chat_template=model_config.get('chat_template', None))['prompt']

peft_model = PeftModel.from_pretrained(model, f'{path}/alpaca_base64_model_final_adapter')
# peft_model = peft_model.merge_and_unload()
peft_model

import base64 

prompt = 'List me 3 fruit. '
# prompt = base64.b64encode(prompt.encode()).decode()
prompt = template.format(instruction=prompt) + 'Sure, '

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
out = peft_model.generate(input_ids, num_return_sequences=1, do_sample=True)
response = tokenizer.decode(out[0], skip_special_tokens=True)
print(response)