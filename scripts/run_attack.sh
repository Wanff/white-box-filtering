#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=lguard_pair_llama2_13b
#SBATCH --output=lguard_pair_llama2_13b.out

# export OPENAI_API_KEY=$(cat .apikey)
export OPENAI_API_KEY=$(cat ../.apikey)

SAVE_PATH="../data/llama2_7b"
# SAVE_PATH="../data/llama2_13b"

#* LLAMA 13B GCG
# FILE_SPEC="gcg_llama13b_text"
# python run_attack.py \
#     --model_name "llama2_13b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_13b/llama-2-13b-for-harm-classification_causal_harmbench_alpaca_metadata_model_0_0" \
#     --text_monitor_config "llamaguard-short" \
#     --file_spec "llama13b_text" \
#     --seed 0

# FILE_SPEC="act_gcg_llama13b_lyr39"
# python run_attack.py \
#     --model_name "llama2_13b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_13b/all_harmbench_alpaca_" \
#     --probe_layer 39 \
#     --probe_type "mlp" \
#     --tok_idxs -1 \
#     --file_spec $FILE_SPEC \
#     --seed 0

#LOG PROB
# FILE_SPEC="log_prob_mlp_lyr30_llama13b"
# python run_attack.py \
#     --model_name "llama2_13b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "log_prob" \
#     --attack_args_path "attack_configs/llama13b_log_prob_config.json" \
#     --monitor_type "act" \
#     --probe_type "mlp" \
#     --probe_data_path "../data/llama2_13b/all_harmbench_alpaca_" \
#     --probe_layer 30 \
#     --probe_type "mlp" \
#     --tok_idxs -1 \
#     --file_spec $FILE_SPEC \
#     --seed 0

# FILE_SPEC="log_prob_text"
# python run_attack.py \
#     --model_name "llama2_13b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "log_prob" \
#     --attack_args_path "attack_configs/log_prob_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_13b/llama-2-13b-for-harm-classification_causal_harmbench_alpaca_metadata_model_0_0" \
#     --text_monitor_config "llamaguard-short" \
#     --file_spec $FILE_SPEC \
#     --seed 0

#PAIR
# FILE_SPEC="pair_mlp_lyr30_llama2_13b"
# python run_attack.py \
#     --model_name "llama2_13b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "pair" \
#     --attack_args_path "attack_configs/llama13b_pair_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_13b/llama-2-13b-for-harm-classification_causal_harmbench_alpaca_metadata_model_0_0" \
#     --text_monitor_config "llamaguard-short" \
#     --file_spec $FILE_SPEC \
#     --seed 0


FILE_SPEC="lguard_pair_llama2_13b"
python run_attack.py \
    --model_name "llama2_13b" \
    --save_path "$SAVE_PATH" \
    --attack_type "pair" \
    --attack_args_path "attack_configs/llama13b_pair_config.json" \
    --monitor_type "act" \
    --probe_type "mlp" \
    --probe_data_path "../data/llama2_13b/all_harmbench_alpaca_" \
    --probe_layer 30 \
    --tok_idxs -1 \
    --file_spec "$FILE_SPEC" \
    --seed 0 

#* LLAMAGUARD
# FILE_SPEC="gcg_lyr24_sk_512_batch_llamaguard"
# SAVE_PATH="../data/llamaguard"
# python run_attack.py \
#     --model_name "llamaguard" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llamaguard/all_harmbench_alpaca_" \
#     --probe_layer 24 \
#     --probe_type "sk" \
#     --tok_idxs -1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

#* LLAMA 7B
#*GCG
# FILE_SPEC="gcg_lyr24_sk_512_batch"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/debug_gcg_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/harmbench_alpaca_" \
#     --probe_layer 24 \
#     --probe_type "sk" \
#     --tok_idxs -1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

# FILE_SPEC="joint_gcg_lyr24_mlp_llama2_7b_take2"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/joint_gcg_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/harmbench_alpaca_" \
#     --probe_layer 24 \
#     --probe_type "mlp" \
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

# FILE_SPEC="actual_final_gcg_joint_ft_llamaguard_harmbench_alpaca_pad_fix"
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

# FILE_SPEC="actual_final_gcg_just_ft_llamaguard_harmbench_alpaca_pad_fix"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_7b/llamaguard_harmbench_alpaca__model_0" \
#     --text_monitor_config "llamaguard" \
#     --file_spec "$FILE_SPEC" \
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

#GCG against llama2-head
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "gcg" \
#     --attack_args_path "attack_configs/gcg_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_7b/llama-2-7b-for-harm-classification_head_harmbench_alpaca_metadata_model_0_0" \
#     --text_monitor_config "llama2-short-head" \
#     --file_spec "llama2_short_head_gcg_final_pad_fix" \
#     --seed 0

#* log prob
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
# FILE_SPEC="log_prob_just_hb_alpaca_lyr24_space"
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

# FILE_SPEC="log_prob_mlp_lyr30_llama2_7b"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "log_prob" \
#     --attack_args_path "attack_configs/log_prob_config.json" \
#     --monitor_type "act" \
#     --probe_type "mlp" \
#     --probe_data_path "../data/llama2_7b/harmbench_alpaca_" \
#     --probe_layer 30 \
#     --tok_idxs -1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0

# FILE_SPEC="log_prob_just_ft_llamaguard_hb_alpaca_space"
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

# *PAIR
# FILE_SPEC="pair_mlp_lyr24_llama2_7b_final"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "pair" \
#     --attack_args_path "attack_configs/pair_config.json" \
#     --monitor_type "act" \
#     --probe_type "mlp" \
#     --probe_data_path "../data/llama2_7b/harmbench_alpaca_" \
#     --probe_layer 24 \
#     --tok_idxs -1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 

# FILE_SPEC="pair_probe_lyr24"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "pair" \
#     --attack_args_path "attack_configs/pair_config.json" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/harmbench_alpaca_" \
#     --probe_layer 24 \
#     --tok_idxs -1 \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 


# FILE_SPEC="lguard_pair_llama2_7b_final"
# python run_attack.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --attack_type "pair" \
#     --attack_args_path "attack_configs/pair_config.json" \
#     --monitor_type "text" \
#     --monitor_path "../data/llama2_7b/llamaguard_harmbench_alpaca__model_0" \
#     --text_monitor_config "llamaguard" \
#     --file_spec "$FILE_SPEC" \
#     --seed 0 