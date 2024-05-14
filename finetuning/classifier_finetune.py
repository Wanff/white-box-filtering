import sys 
sys.path.append('../')

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import default_data_collator, get_linear_schedule_with_warmup, Adafactor
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS
from datasets import load_from_disk, DatasetDict, load_dataset, Dataset
import argparse
from white_box.utils import rotation_cipher
import base64
import pandas as pd

from peft import get_peft_model, LoraConfig, TaskType
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    # load model
    model_config = MODEL_CONFIGS['llama2_7b']
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    template = get_template(args.model_name, chat_template=model_config.get('chat_template', None))['prompt']

    # load dataset
    train_dataset = pd.read_csv(f'{args.path}/{args.file_spec}metadata.csv')
    test_dataset = pd.read_csv(f'{args.path}/{args.file_spec}test_metadata.csv')
    
    # hf dataset
    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)
    
    def custom_collator(examples):
        input_ids = []
        last_token_idxs = []
        for ex in examples: 
            prompt = ex['prompt']
            prompt = "<<SYS>>\nIs the following query requesting something harmful? Output your response. The first line must contain only 'safe' and 'unsafe'.\n<</SYS>>\n\n" + prompt
            prompt = template.format(instruction=prompt)
            input_ids.append(tokenizer(prompt)['input_ids'])
            last_token_idxs.append(len(input_ids[-1]) - 1)
        
        input_ids = tokenizer.pad({'input_ids': input_ids}, return_tensors='pt')['input_ids']
        labels = torch.tensor([example['label'] for example in examples])
        last_token_idxs = torch.tensor(last_token_idxs)
        return {'input_ids': input_ids, 'labels': labels, 'last_token_idxs': last_token_idxs}
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collator)

    # lora 
    # peft_config = LoraConfig(
    #     target_modules=["a_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "score"],
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type=TaskType.SEQ_CLS, 
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * 0.05, 
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    model = model.to(args.device)

    for epoch in range(args.num_epochs):
        total_loss = 0
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            model.train()
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if step == 0: 
                print(batch['input_ids'].shape, batch['labels'].shape)
                print(tokenizer.decode(batch['input_ids'][0]))
            outputs = model(batch['input_ids']) # b,s,v
            preds = torch.stack([outputs.logits[torch.arange(outputs.logits.shape[0]), batch['last_token_idxs'], 9109], outputs.logits[torch.arange(outputs.logits.shape[0]), batch['last_token_idxs'], 25110]], dim=1).softmax(-1)
            # preds = outputs.logits
            loss = F.cross_entropy(preds, batch['labels'])
            loss = loss / args.accumulation_steps
            loss.backward()
            total_loss += loss.detach().float()
            
            if step % args.accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            torch.cuda.empty_cache()
            del batch
            del outputs
    
            pbar.set_description(f"Step {step} | Loss: {loss}")

        print("RUNNING EVAL")
        acc = 0
        model.eval()
        for step, batch in enumerate(test_dataloader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(batch['input_ids'])
            preds = torch.stack([outputs.logits[torch.arange(outputs.logits.shape[0]), batch['last_token_idxs'], 9109], outputs.logits[torch.arange(outputs.logits.shape[0]), batch['last_token_idxs'], 25110]], dim=1).softmax(-1)
            # preds = outputs.logits
            loss = F.cross_entropy(preds, batch['labels'])
            preds = preds.argmax(-1)
            acc += torch.sum(preds == batch['labels'])
            total_loss += loss.detach().float()

            torch.cuda.empty_cache()
            del batch
            del outputs
    
            pbar.set_description(f"Step {step} | Loss: {loss}")
    
        test_epoch_loss = total_loss / (len(test_dataloader) * args.batch_size)
        test_acc = acc / (len(test_dataloader) * args.batch_size)
        print(f"{epoch} | Test Loss: {test_epoch_loss} | Test Acc: {test_acc}")

        train_epoch_loss = total_loss / len(train_dataloader)
        print(f"{epoch} | Train Loss: {train_epoch_loss}")
         
        model.save_pretrained(f'{args.path}/llamaguard_new_{args.file_spec}_model_{epoch}')
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/LlamaGuard-7b', help='model name')
    parser.add_argument('--path', type=str, default='../data/llama2_7b')
    parser.add_argument('--file_spec', type=str, default='generated_')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size per device')
    parser.add_argument('--accumulation_steps', type=int, default=16, help='accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=4, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()

    set_seed(args.seed)

    main(args)