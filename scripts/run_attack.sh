#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=actual_final_hb_lyr24
#SBATCH --output=actual_final_hb_lyr24.out

# export OPENAI_API_KEY=$(cat .apikey)
# export OPENAI_API_KEY=$(cat ../.apikey)

# SAVE_PATH="../data/llama2_7b"
# FILE_SPEC="final_gcg_run_just_ft_llamaguard"

# FILE_SPEC="actual_final_hb_lyr24"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/joint_gcg_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/harmbench_alpaca_" \
#     --probe_layer 24 \
#     --probe_type "sk" \
#     --tok_idxs -1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

# FILE_SPEC="actual_final_gcg_ft_llamaguard_harmbench_alpaca_"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/joint_gcg_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_7b/llamaguard_harmbench_alpaca__model_0" \
#     --text_monitor_config "llamaguard" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "text" \
#     --file_spec "$FILE_SPEC" \
#     --text_monitor_config "llamaguard" \
#     --seed 0

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "text" \
#     --file_spec "$FILE_SPEC" \
#     --text_monitor_config "llamaguard" \
#     --seed 0

# LLAMA2 FOR HARM CLASSIFICATION

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_7b/llama-2-7b-for-harm-classification_harmbench_alpaca_metadata_model_0_0" \
#     --text_monitor_config "llamaguard-short" \
#     --file_spec "gcg_just_ft_llama2_" \
#     --seed 0

#* log prob
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "log_prob" \
#     --attack_args_path "attack_configs/log_prob_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/all_harmbench_alpaca_" \
#     --probe_layer 24 \
#     --tok_idxs -1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "log_prob" \
#     --attack_args_path "attack_configs/log_prob_config.json" \
#     --monitor_type "text" \
#     --text_monitor_config "llamaguard" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 &> log_prob_llamaguard.log

# #* log prob
# FILE_SPEC="log_prob_just_hb_alpaca_lyr24"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "log_prob" \
#     --attack_args_path "attack_configs/log_prob_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/harmbench_alpaca_" \
#     --probe_layer 24 \
#     --tok_idxs -1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

# FILE_SPEC="log_prob_just_ft_llamaguard_hb_alpaca"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "log_prob" \
#     --attack_args_path "attack_configs/log_prob_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_7b/llamaguard_harmbench_alpaca__model_0" \
#     --text_monitor_config "llamaguard" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 

#*PAIR
FILE_SPEC="pair_probe_lyr24"
python run_attack.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --attack_type "pair" \
    --attack_args_path "attack_configs/pair_config.json" \
    --monitor_type "act" \
    --probe_data_path "../data/llama2_7b/all_harmbench_alpaca_" \
    --probe_layer 24 \
    --tok_idxs -1 \
    --file_spec "$FILE_SPEC" \
    --seed 0 &> ../data/llama2_7b/pair_probe_lyr24.log & 