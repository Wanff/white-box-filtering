#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=LlamaGuardGCG200
#SBATCH --output=LlamaGuardGCG200.out

SAVE_PATH="../data/llama2_7b"
FILE_SPEC="gcg_run_llamaguard_200_"

# python run_gcg.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/jb_" \
#     --probe_layer 24 \
#     --monitor_loss_weight 1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

python run_gcg.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --monitor_type "text" \
    --file_spec "$FILE_SPEC" \
    --seed 0 \
