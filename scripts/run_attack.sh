#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=HBAlpacaMMGCG
#SBATCH --output=HBAlpacaMMGCG.out

SAVE_PATH="../data/llama2_7b"
# FILE_SPEC="log_prob_lyr24_take2"

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "log_prob" \
#     --attack_args_path "attack_configs/log_prob_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/jb_" \
#     --probe_layer 24 \
#     --tok_idxs -1  \
#     --file_spec "$FILE_SPEC" \
#     --seed 0  &> "$SAVE_PATH/$FILE_SPEC.out" &

FILE_SPEC="gcg_text_test"
python run_attack.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --attack_type "gcg" \
    --attack_args_path "attack_configs/gcg_config.json" \
    --monitor_type "text" \
    --file_spec "$FILE_SPEC" \
    --seed 0  &> "$SAVE_PATH/$FILE_SPEC.out" &

# python run_gcg.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --monitor_type "text" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 \
