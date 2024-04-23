#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=HBDollyProbeGCG
#SBATCH --output=HBDollyProbeGCG.out

SAVE_PATH="../data/llama2_7b"
FILE_SPEC="gcg_run_harmbench_dolly"

python run_gcg.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --monitor_type "act" \
    --probe_data_path "../data/llama2_7b/harmbench_dolly_" \
    --probe_layer 24 \
    --monitor_loss_weight 1 \
    --file_spec "$FILE_SPEC"

# python run_gcg.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --monitor_type "text" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 \
