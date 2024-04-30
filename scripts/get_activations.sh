# python3 get_activations.py \
#     --model_name "EleutherAI/pythia-70m" \
#     --dataset "../data/pythia-70m/mem/pile.pkl" \
#     --save_path "../data/pythia-70m/mem" \
#     --N_PROMPTS 500 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging \
#     --mem  &> ../data/pythia-70m/mem/pile.out &

# python3 get_activations.py \
#     --model_name "EleutherAI/pythia-70m" \
#     --dataset "../data/pythia-70m/mem/pythia_evals.pkl" \
#     --save_path "../data/pythia-70m/mem" \
#     --N_PROMPTS 500 \
#     --save_every 100 \
#     --return_prompt_acts \
#     --logging \
#     --mem  &> ../data/pythia-70m/mem/pythia_evals.out &

# python get_activations.py \
#     --model_name "llama2_7b" \
#     --dataset_name_or_path "../data/llama2_7b/jb_metadata.csv" \
#     --save_path "../data/llama2_7b/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "jb_" &> ../data/llama2_7b/jb.out &

# python get_activations.py \
#     --model_name "llama2_7b" \
#     --dataset_name_or_path "../data/llama2_7b/jb_metadata.csv" \
#     --save_path "../data/llama2_7b/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "jb_" &> ../data/llama2_7b/jb.out &
    
# python get_activations.py \
#     --model_name "llama2_7b" \
#     --dataset_name_or_path "../data/llama2_7b/jb_eq_by_jbname_metadata.csv" \
#     --save_path "../data/llama2_7b/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "jb_eq_by_jbname_" &> ../data/llama2_7b/jb_eq_by_jbname.out &

# python get_activations.py \
#     --model_name "akdeniz27/llama-2-7b-hf-qlora-dolly15k-turkish" \
#     --dataset_name_or_path "/home/ubuntu/rowan/white-box-filtering/data/turkish/harmful_behaviors_custom.csv" \
#     --save_path "/home/ubuntu/rowan/white-box-filtering/data/turkish/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --device "cuda" \
#     --file_spec "harmful_behaviors_custom_"

# TURKISH
# python get_activations.py \
#     --model_name "akdeniz27/llama-2-7b-hf-qlora-dolly15k-turkish" \
#     --dataset_name_or_path "../data/turkish/harmless_behaviors_custom.csv" \
#     --save_path "../data/turkish/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "harmless_behaviors_custom_"

# DUTCH
# python get_activations.py \
#     --model_name "llama2_7b_dutch" \
#     --dataset_name_or_path "../data/dutch/harmful_behaviors_custom_metadata.csv" \
#     --save_path "../data/dutch/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "harmful_behaviors_custom_"

# HUNGARIAN
# python get_activations.py \
#     --model_name "llama2_7b_hungarian" \
#     --dataset_name_or_path "../data/hungarian/harmful_behaviors_custom_metadata.csv" \
#     --save_path "../data/hungarian/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "harmful_behaviors_custom_"

HUNGARIAN
python get_activations.py \
    --model_name "llama2_7b_slovenian" \
    --dataset_name_or_path "../data/slovenian/harmless_behaviors_custom_metadata.csv" \
    --save_path "../data/slovenian/" \
    --tok_idxs -1 -2 -3 -4 -5  \
    --file_spec "harmless_behaviors_custom_"