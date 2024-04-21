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

# python get_activations.py dd\
#     --model_name "akdeniz27/llama-2-7b-hf-qlora-dolly15k-turkish" \
#     --dataset_name_or_path "/home/ubuntu/rowan/white-box-filtering/data/llama2_7b/jb_unif_behav_metadata.csv" \
#     --save_path "/home/ubuntu/rowan/white-box-filtering/data/turkish/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "jb_unif_behav_"

python get_activations.py \
    --model_name "llama2_7b" \
    --dataset_name_or_path "/home/ubuntu/rowan/white-box-filtering/data/harmbench_alpaca.csv" \
    --save_path "/home/ubuntu/rowan/white-box-filtering/data/llama2_7b/" \
    --tok_idxs -1 -2 -3 -4 -5  \
    --file_spec "harmbench_alpaca_"