#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=LLaMaGuardGCG
#SBATCH --output=LLaMaGuardGCG.out

SAVE_PATH="../data/llama2_7b"
FILE_SPEC="pair_run_lyr24"
# FILE_SPEC="pure_gcg_20exclam_posidx18"

python run_pair.py \
    --target_model "llama-2" \
    --attack_model "vicuna" \
    --save_path "$SAVE_PATH" \
    --monitor_type "act" \
    --probe_data_path "../data/llama2_7b/jb_" \
    --probe_layer 24 \
    --seed 0 \
    --file_spec "$FILE_SPEC"  &> "$SAVE_PATH/$FILE_SPEC.out" &
