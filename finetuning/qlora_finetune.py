import sys 
sys.path.append('../')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import default_data_collator, get_linear_schedule_with_warmup, BitsAndBytesConfigs

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
from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args): 

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    peft_config = LoraConfig(
        target_modules=["a_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "Im_head"],
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_config = MODEL_CONFIGS[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name_or_path'])
    tokenizer.pad_token = tokenizer.eos_token
    template = get_template(args.model_name, chat_template=model_config.get('chat_template', None))['prompt']

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name_or_path'],
        token='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',
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    dataset = load_dataset(args.dataset_name, split='train')
    dataset = dataset.filter(lambda x: len(x['input']) == 0)
    dataset = dataset.remove_columns('input')

    if args.subsample_size is not None:
        dataset = dataset.shuffle(seed)
        dataset = dataset.select(range(args.subsample_size))

    if 'alpaca' in args.dataset_name: 

        def instruction_format(examples): 
            instructions, outputs = examples['instruction'], examples['output']

            if args.cipher == 'base64': 
                instructions = [f'Only respond in base64. {base64.b64encode(instruction.encode()).decode()}' for instruction in instructions]
                outputs = [base64.b64encode(output.encode()).decode() for output in outputs]
            elif args.cipher.startswith('rot'): 
                shift = int(args.cipher.split('_')[1])
                instructions = [f'Only respond in ROT{shift}. E.g. a -> b, b -> c, c -> d, etc. {rotation_cipher(instruction, shift)}' for instruction in instructions]
                outputs = [rotation_cipher(output, shift) for output in outputs]
            
            return {'text': [template.format(instruction=instruction) + output for instruction, output in zip(instructions, outputs)]}
        
        dataset = dataset.map(instruction_format, batched=True)
        dataset = dataset.remove_columns(['instruction', 'output'])
    
    print(dataset[0])

    # with torch.autocast('cuda'): 
    #     for epoch in range(args.num_epochs):
    #         total_loss = 0
    #         for step, batch in enumerate(tqdm(train_dataloader)):
    #             batch = {k: v.to(args.device) for k, v in batch.items()}
    #             outputs = model(**batch)
    #             loss = outputs.loss
    #             print(loss)
    #             total_loss += loss.detach().float()
    #             loss.backward()
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()

    #             torch.cuda.empty_cache()
    #             del batch
    #             del outputs
    #             del loss

    #         train_epoch_loss = total_loss / len(train_dataloader)
    #         train_ppl = torch.exp(train_epoch_loss)
    #         print(f"{epoch} | Train Loss: {train_epoch_loss} | Train PPL: {train_ppl}")

    #         model.save_pretrained(os.path.join(args.path, args.output_name + '_final_adapter'))

    targs = TrainingArguments(
        output_dir = os.path.join(args.path, args.output_name),
        logging_strategy = 'epoch',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=num_epochs,
        report_to='none',
        seed=seed,
        optim="paged_adamw_32bit",
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
        weight_decay=0.001,
    )

    trainer = SFTTrainer(
        model, 
        train_dataset=dataset, 
        dataset_text_field='text',
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=targs,
    )

    #Upcast layer norms to float 32 for stability
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()
    trainer.model.save_pretrained(os.path.join(args.path, args.output_name + '_final_adapter'))

    # test on dataset[0]
    inputs = tokenizer(dataset[0]['text'], return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.9, top_k=50, top_p=0.95, num_return_sequences=1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/data/oam_patel/white-box-filtering/data/ciphers/llama2_7b', help='path to data')
    parser.add_argument('--dataset_name', type=str, default='yahma/alpaca-cleaned', help='name of dataset')
    parser.add_argument('--subsample_size', type=int, default=None, help='size of subsample')
    parser.add_argument('--cipher', type=str, default='rot_1', help='cipher')
    parser.add_argument('--output_name', type=str, default='alpaca_rot_1_model_8bit', help='name of output')
    parser.add_argument('--model_name', type=str, default='llama2_7b', help='name of model')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size per device')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()

    main(args)