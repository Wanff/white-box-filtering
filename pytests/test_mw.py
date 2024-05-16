from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
from typing import List 

from white_box.model_wrapper import ModelWrapper
from white_box.chat_model_utils import load_model_and_tokenizer, get_template, MODEL_CONFIGS


def _test_batch_hiddens(model, model_name, prompts, padding_side : str = "right", tok_idxs : List[int] = [-1]):
    model_config = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side = padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    template = get_template(model_name, chat_template=model_config.get('chat_template', None))['prompt']
    
    mw = ModelWrapper(model, tokenizer, template = template)

    test_tensor = mw.batch_hiddens(prompts, tok_idxs = tok_idxs)['resid']

    formatted_prompts = [template.format(instruction=s) for s in prompts]
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, max_length=2048, truncation=True)

    res = model(**inputs, output_hidden_states = True)['hidden_states']
    hidden_states = torch.stack([r for r in res], dim=1)[:, 1:].float() # remove the first hidden state, which is the input embeddings

    tok_idxs = torch.tensor(tok_idxs).sort().values
    if padding_side == "right":
        if len(prompts) == 2:
            first_prompt_state = hidden_states[0, :, tok_idxs - 2]
            sec_prompt_state = hidden_states[1, :, tok_idxs]
            
            assert torch.allclose(test_tensor[0], first_prompt_state)
            assert torch.allclose(test_tensor[1], sec_prompt_state)
        else:
            assert torch.allclose(test_tensor, hidden_states[:,:, tok_idxs])
    else:
        assert torch.allclose(test_tensor, hidden_states[:, :, tok_idxs])
        
    
def test_batch_hiddens():
    model_name = 'llama2_7b'
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
        torch_dtype=torch.float16, 
        device_map="auto")
    
    prompts = [
        ["Hello, how are you?", "I am doing well, thank you."],
        ["Hello, how are you?"]
    ]
    padding_sides = ["right", "left"]
    tok_idxss = [[-1], [-1, -2, -3, -4]]
    
    for prompt in prompts:
        for padding_side in padding_sides:
            for tok_idxs in tok_idxss:
                _test_batch_hiddens(model, model_name, prompt, padding_side, tok_idxs)
    
    
    