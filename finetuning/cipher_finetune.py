import sys 
sys.path.append('../')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import default_data_collator, get_linear_schedule_with_warmup
import torch
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

def main(args): 

    model_config = MODEL_CONFIGS[args.model_name]
    model, tokenizer = load_model_and_tokenizer(**model_config)
        
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
        
        dataset = dataset.map(instruction_format, batched=True, remove_columns = ['instruction', 'output'])
    
    else: 
        raise NotImplementedError
    
    print(dataset[0])
    print(dataset[5])
    raise Exception

    def tokenize_function(examples):
        return {'input_ids': tokenizer(examples['text'], truncation=True, max_length=512)['input_ids']}
    
    dataset = dataset.map(tokenize_function, batched=False)
    train_dataset = dataset.remove_columns('text')

    def custom_collator(examples):
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['input_ids'] for example in examples]
        max_len = max([len(x) for x in input_ids])
        input_ids = [x + [tokenizer.pad_token_id]*(max_len-len(x)) for x in input_ids]
        labels = [x + [-100]*(max_len-len(x)) for x in labels]
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        return {'input_ids': input_ids, 'labels': labels}
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * 0.05, 
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # model = model.to(args.device)
    model.train()

    for epoch in range(args.num_epochs):
        total_loss = 0
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            
            loss = outputs.loss
            loss = loss / args.accumulation_steps
            loss.backward()
            total_loss += loss.detach().float()
            
            if step % args.accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            print(loss)

            torch.cuda.empty_cache()
            del batch
            del outputs
    
            pbar.set_description(f"Step {step} | Loss: {loss}")

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch} | Train Loss: {train_epoch_loss} | Train PPL: {train_ppl}")

        model.save_pretrained(os.path.join(args.path, args.output_name + '_final_adapter'))

    # test on dataset[0]
    inputs = tokenizer(dataset[0]['text'], return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.9, top_k=50, top_p=0.95, num_return_sequences=1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3_8b', help='model name')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='dtype')
    parser.add_argument('--path', type=str, default='../data/llama3_8b')
    parser.add_argument('--train_file_spec', type=str, default='yahma/alpaca-cleaned')
    parser.add_argument('--cipher', type=str, default='rot_7', help='cipher')
    parser.add_argument('--output_name', type=str, default='alpaca_caesar7_llama3', help='name of output')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size per device')
    parser.add_argument('--accumulation_steps', type=int, default=16, help='accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--subsample', type=int, default=None, help='subsample')
    parser.add_argument('--no_save_at_end', action='store_true', default=False, help='save at end')
    args = parser.parse_args()
    
    set_seed(args.seed)


    main(args)