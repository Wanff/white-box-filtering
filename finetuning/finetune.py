import sys 
sys.path.append('../')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, AutoModelForCausalLM
import torch
import os
import numpy as np
from tqdm import tqdm
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS
from datasets import load_from_disk, DatasetDict
import argparse

from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

def main(args): 

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, 
        bias="none",
        modules_to_save=["score"],
        task_type="CAUSAL_LM",
    )

    model_config = MODEL_CONFIGS[args.model_name]
    model, tokenizer = load_model_and_tokenizer(**model_config)
    model.config.use_cache = False
    tokenizer.pad_token = tokenizer.eos_token
    template = get_template(args.model_name, chat_template=model_config.get('chat_template', None))['prompt']

    dataset = load_from_disk(os.path.join(args.path, args.dataset_name))

    if args.subsample_size is not None:
        dataset = dataset.shuffle(seed)
        dataset = dataset.select(range(args.subsample_size))

    if 'alpaca' in args.dataset_name: 

        def instruction_format(examples): 
            instructions, outputs = examples['instruction'], examples['output']
            return {'text': [template.format(instruction=instructions[i]) + outputs[i] for i in range(len(instructions))]}
        
        dataset = dataset.map(instruction_format, batched=True)
        dataset = dataset.remove_columns(['instruction', 'output'])

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=1024)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    dataset = dataset.train_test_split(test_size=0.1)

    targs = TrainingArguments(
        output_dir = os.path.join(args.path, args.output_name),
        evaluation_strategy = 'epoch',
        logging_strategy = 'epoch',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        group_by_length=True,
        lr_scheduler_type="linear", 
        warmup_ratio=0.03, 
        num_train_epochs=num_epochs,
        report_to='none',
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=targs,
    )

    trainer.train()
    trainer.model.save_pretrained(os.path.join(args.path, args.output_name + '_final'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/ubuntu/rowan/white-box-filtering/data/ciphers/llama2_7b', help='path to data')
    parser.add_argument('--dataset_name', type=str, default='alpaca_base64', help='name of dataset')
    parser.add_argument('--subsample_size', type=int, default=1000, help='size of subsample')
    parser.add_argument('--output_name', type=str, default='alpaca_base64_model ', help='name of output')
    parser.add_argument('--model_name', type=str, default='llama2_7b', help='name of model')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    args = parser.parse_args()

    main(args)