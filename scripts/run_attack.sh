#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=FinalLinearLossScheduleFTLlamaGuardGCG
#SBATCH --output=FinalLinearLossScheduleFTLlamaGuardGCG.out

SAVE_PATH="../data/llama2_7b"
# FILE_SPEC="final_gcg_run_just_ft_llamaguard"

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/jb_" \
#     --probe_layer 24 \
#     --tok_idxs -1 -2 -3 -4 -5 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 

FILE_SPEC="final_gcg_test"
python run_attack.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --attack_type "gcg" \
    --attack_args_path "attack_configs/gcg_config.json" \
    --monitor_type "text" \
    --monitor_path "../data/llama2_7b/llamaguard_generated__model_2" \
    --file_spec "$FILE_SPEC" \
    --seed 0 
    

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

# python run_gcg.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --monitor_type "text" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0
