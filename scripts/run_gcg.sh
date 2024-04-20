#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --job-name=LLaMaGuardGCG
#SBATCH --output=LLaMaGuardGCG.out

SAVE_PATH="../data/llama2_7b"
FILE_SPEC="gcg_run_llama_guard"
# FILE_SPEC="gcg_run_lyr24_final"

# python run_gcg.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/jb_" \
#     --probe_layer 24 \
#     --monitor_loss_weight 1 \
#     --file_spec "$FILE_SPEC"  &> "$SAVE_PATH/$FILE_SPEC.out" &

python run_gcg.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --monitor_type "text" \
    --file_spec "$FILE_SPEC"