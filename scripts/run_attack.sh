#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=actual_final_gcg_just_ft_llamaguard_generated_
#SBATCH --output=actual_final_gcg_just_ft_llamaguard_generated_.out

SAVE_PATH="../data/llama2_7b"
# FILE_SPEC="final_gcg_run_just_ft_llamaguard"

FILE_SPEC="gcg_gpt_gen_24_just_last_tok"
python run_attack.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --attack_type "gcg" \
    --attack_args_path "attack_configs/gcg_config.json" \
    --monitor_type "act" \
    --probe_data_path "../data/llama2_7b/all_gpt_gen_" \
    --probe_layer 24 \
    --tok_idxs -1 \
    --file_spec "$FILE_SPEC" \
    --seed 0 &> "$SAVE_PATH/$FILE_SPEC.out" &

# FILE_SPEC="actual_final_gcg_just_ft_llamaguard_generated_"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_7b/llamaguard_new_generated__model_0" \
#     --text_monitor_config "llamaguard+" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 
    

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

python run_attack.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --attack_type "gcg" \
    --attack_args_path "attack_configs/gcg_config.json" \
    --monitor_type "text" \
    --file_spec "$FILE_SPEC" \
    --text_monitor_config "llamaguard" \
    --seed 0
