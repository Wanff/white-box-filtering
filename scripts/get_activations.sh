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

python3 get_activations.py \
    --model_name "llama2_7b" \
    --dataset_name_or_path "../data/llama2_7b/jb_metadata.csv" \
    --save_path "../data/llama2_7b/" \
    --tok_idxs -1 -2 -3 -4 -5  \
    --file_spec "jb_" &> ../data/llama2_7b/jb.out &
    