import sys 
sys.path.append('../')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import default_data_collator, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS
from datasets import load_from_disk, DatasetDict, load_dataset
import argparse
from white_box.utils import rotation_cipher
import base64

from peft import get_peft_model, LoraConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def rotation_cipher(text, rotation):
    result = ""
    for char in text:
        if char.isalpha():
            ascii_offset = ord('a') if char.islower() else ord('A')
            rotated_char = chr((ord(char) - ascii_offset + rotation) % 26 + ascii_offset)
            result += rotated_char
        else:
            result += char
    return result

def encode_text(text, cipher):
    if cipher == 'base64':
        return base64.b64encode(text.encode()).decode()
    elif cipher.startswith('rot'):
        shift = int(args.cipher.split('_')[1])
        return rotation_cipher(text, shift)
    return text

def split_keep_delimiter(string, delimiter):
    parts = string.split(delimiter, 1)
    if len(parts) > 1:
        parts[0] += delimiter
    return parts

def main(args): 

    model_config = MODEL_CONFIGS[args.model_name]
    model_config['dtype'] = args.dtype
    model, tokenizer = load_model_and_tokenizer(**model_config)
        

    if args.use_peft:
        peft_config = LoraConfig(
            target_modules=["a_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "Im_head"],
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    if "llama3" not in args.model_name:
        template = get_template(args.model_name, chat_template=model_config.get('chat_template', None))['prompt']

    dataset = load_dataset(args.dataset_name, split='train')
    dataset = dataset.filter(lambda x: len(x['input']) == 0)
    dataset = dataset.remove_columns('input')
    dataset = dataset.shuffle(seed=args.seed)

    if args.subsample_size is not None:
        dataset = dataset.select(range(args.subsample_size))

    if 'alpaca' in args.dataset_name: 

        def instruction_format(examples): 
            instructions, outputs = examples['instruction'], examples['output']
            
            if "llama3" in args.model_name:
                cipher_name = "base64" if args.cipher == "base64" else f"Caesar cipher with rotation {args.cipher.split('_')[1]}"
                translate_prompt_msgs = [ [{"role" : "system", "content": f"Translate this message from {cipher_name} to English"},
                            {"role":"user", "content" : encode_text(output, cipher = args.cipher)},
                            {"role" : "assistant", "content" : output}] for output in outputs]
                
                translate_prompt_strs = [tokenizer.apply_chat_template(msg, tokenize = False) for msg in translate_prompt_msgs]

                encoded_prompt_msgs = [ [{"role" : "system", "content": f"Respond only in {cipher_name}"},
                            {"role": "user", "content" : encode_text(instruction, cipher = args.cipher)},
                            {"role" : "assistant", "content" : encode_text(output, cipher = args.cipher)}] for instruction, output in zip(instructions, outputs)]
                encoded_prompt_strs = [tokenizer.apply_chat_template(msg, tokenize = False) for msg in encoded_prompt_msgs]
                
                return {"text" : translate_prompt_strs + encoded_prompt_strs}
            
            else:
                if args.cipher == 'base64': 
                    instructions = [f'Only respond in base64. {base64.b64encode(instruction.encode()).decode()}' for instruction in instructions]
                    outputs = [base64.b64encode(output.encode()).decode() for output in outputs]
                elif args.cipher.startswith('rot'): 
                    shift = int(args.cipher.split('_')[1])
                    instructions = [f'Only respond in ROT{shift}. E.g. a -> b, b -> c, c -> d, etc. {rotation_cipher(instruction, shift)}' for instruction in instructions]
                    outputs = [rotation_cipher(output, shift) for output in outputs]
            
                return {'text': [template.format(instruction=instruction) + output for instruction, output in zip(instructions, outputs)]}            
        print(len(dataset))
        dataset = dataset.map(instruction_format, batched=True, remove_columns=['instruction', 'output'])
        print(len(dataset))
    else: 
        raise NotImplementedError
    
    dataset = dataset.shuffle(seed=args.seed)
    print(tokenizer.batch_decode(tokenizer(dataset[15]['text'], truncation = True, max_length = 1024)['input_ids']))

    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    
    def custom_collator(examples):
        
        input_ids = []
        last_token_idxs = []
        for ex in examples: 
            prompt = ex['text']
            input_ids.append(tokenizer(prompt, truncation = True, max_length = 512)['input_ids'])
            last_token_idxs.append(len(input_ids[-1]) - 1)
        
        # input_ids = tokenizer.pad({'input_ids': input_ids}, return_tensors='pt')['input_ids']
        input_ids = tokenizer([ex['text'] for ex in examples], padding=True, return_tensors='pt', truncation=True, max_length=256)['input_ids']
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {'input_ids': input_ids, 'labels': labels}
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * 0.05, 
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # model = model.to(args.device)
    model.train()

    for epoch in range(args.num_epochs):
        total_loss = 0
        pbar = tqdm(train_dataloader)
        print("RUNNING TRAIN")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if step == 0: 
                print(batch['input_ids'].shape, batch['labels'].shape)
                print(tokenizer.decode(batch['input_ids'][0]))

            outputs = model(**batch)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.shape[-1]), batch['labels'].view(-1))
            loss = loss / args.accumulation_steps
            loss.backward()
            total_loss += loss.detach().float()
            
            if step % args.accumulation_steps == 0 and step != 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            torch.cuda.empty_cache()
            del batch
            del outputs
    
            pbar.set_description(f"Step {step} | Loss: {loss}")

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch} | Train Loss: {train_epoch_loss} | Train PPL: {train_ppl}")
        
        if args.save_per_epoch:
            model.save_pretrained(f'{args.path}/{args.output_name}_epoch_{epoch}')
        
        print("RUNNING EVAL")
        model.eval()
        total_loss = 0
        pbar = tqdm(test_dataloader)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.shape[-1]), batch['labels'].view(-1))
            total_loss += loss.detach().float()
            
            torch.cuda.empty_cache()
            del batch
            del outputs
    
            pbar.set_description(f"Step {step} | Loss: {loss}")
            
        test_epoch_loss = total_loss / len(test_dataloader)
        test_ppl = torch.exp(test_epoch_loss)
        print(f"{epoch} | Test Loss: {test_epoch_loss} | Test PPL: {test_ppl}")
            
    model.save_pretrained(f'{args.path}/{args.output_name}')

    # test on dataset[0]
    prompt = dataset['test'][0]['text']
    # just keep the first example
    prompt = split_keep_delimiter(prompt, '<|start_header_id|>assistant<|end_header_id|>\n\n')[0]
    print(prompt)
    toks = tokenizer(prompt, return_tensors='pt').to(args.device)
    output = model.generate(**toks, max_new_tokens=100)
    print(tokenizer.decode(output[0]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3_8b', help='model name')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='dtype')
    parser.add_argument('--path', type=str, default='../data/llama3_8b/ciphers/')
    parser.add_argument('--dataset_name', type=str, default='yahma/alpaca-cleaned')
    parser.add_argument('--cipher', type=str, default='rot_7', help='cipher')
    parser.add_argument('--output_name', type=str, default='alpaca_rot7_llama3', help='name of output')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size per device')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--subsample_size', type=int, default=2500, help='subsample')
    parser.add_argument('--save_per_epoch', action='store_true', default=False, help='save at end')
    parser.add_argument('--use_peft', action='store_true', default=True, help='save at end')

    args = parser.parse_args()
    
    set_seed(args.seed)


    main(args)