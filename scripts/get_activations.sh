#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=get_activations
#SBATCH --output=get_activations.out

# GENERALIZATION
# python get_activations.py \
#     --model_name "llama2_7b" \
#     --dataset_name_or_path "../data/llama2_7b/all_gpt_gen_metadata.csv" \
#     --save_path "../data/llama2_7b/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --padding_side "right" \
#     --file_spec "all_gpt_gen_" &> ../data/llama2_7b/all_gpt_gen.out &

# BATTERY FOR NORMAL PROBE ACTS

# datasets=("harmbench_alpaca_metadata" "harmbench_alpaca_test_metadata" "generated_metadata" "generated_test_metadata")

# for dataset in "${datasets[@]}"; do
#     file_spec="${dataset%_metadata}"  # Removes '_metadata' from the dataset name to form the file_spec prefix
#     python get_activations.py \
#         --model_name "llama2_7b" \
#         --dataset_name_or_path "../data/llama2_7b/${dataset}.csv" \
#         --save_path "../data/llama2_7b/" \
#         --tok_idxs -1 -2 -3 -4 -5 \
#         --padding_side "right" \
#         --file_spec "${file_spec}_"
#     echo "Done with $dataset"
# done

# GENERALIZATION

# python get_activations.py \
#         --model_name "llama2_7b" \
#         --dataset_name_or_path "../data/llama2_7b/all_harmbench_alpaca_metadata.csv" \
#         --save_path "../data/llama2_7b/" \
#         --tok_idxs -1 -2 -3 -4 -5 \
#         --padding_side "right" \
#         --file_spec "all_harmbench_alpaca_"

# ADVBENCH + CUSTOM GPT

# python get_activations.py \
#         --model_name "llama2_7b" \
#         --dataset_name_or_path "../data/harmful_behaviors_metadata.csv" \
#         --save_path "../data/llama2_7b/" \
#         --tok_idxs -1 -2 -3 -4 -5 \
#         --padding_side "right" \
#         --file_spec "harmful_behaviors_"

# python get_activations.py \
#         --model_name "llama2_7b" \
#         --dataset_name_or_path "../data/harmless_behaviors_metadata.csv" \
#         --save_path "../data/llama2_7b/" \
#         --tok_idxs -1 -2 -3 -4 -5 \
#         --padding_side "right" \
#         --file_spec "harmless_behaviors_"

# FPR TUNING
# python get_activations.py \
#         --model_name "llama2_7b" \
#         --dataset_name_or_path "../data/llama2_7b/alpaca_negatives_metadata.csv" \
#         --save_path "../data/llama2_7b/" \
#         --tok_idxs -1 -2 -3 -4 -5 \
#         --padding_side "right" \
#         --file_spec "alpaca_negatives_"

# BATTERY FOR LANGUAGE EXPERIMENTS

# languages=("dutch" "hungarian" "slovenian")
# behaviors=("harmful" "harmless")
# datasets=("harmbench_alpaca_metadata" "generated_metadata")

# for lang in "${languages[@]}"; do
#     for behavior in "${behaviors[@]}"; do
#         python get_activations.py \
#             --model_name "llama2_7b_$lang" \
#             --dataset_name_or_path "../data/$lang/${behavior}_behaviors_custom_metadata.csv" \
#             --save_path "../data/$lang/" \
#             --tok_idxs -1 -2 -3 -4 -5 \
#             --padding_side "right" \
#             --file_spec "${behavior}_behaviors_custom_"
#         echo "Done with $lang $behavior"
#     done
#     for dataset in "${datasets[@]}"; do
#         file_spec="${dataset%_metadata}"  # Removes '_metadata' from the dataset name to form the file_spec prefix
#         python get_activations.py \
#             --model_name "llama2_7b_$lang" \
#             --dataset_name_or_path "../data/$lang/${dataset}.csv" \
#             --save_path "../data/$lang/" \
#             --tok_idxs -1 -2 -3 -4 -5 \
#             --padding_side "right" \
#             --file_spec "${file_spec}_"
#         echo "Done with $lang $dataset"
#     done
# done

# python get_activations.py \
#             --model_name "llama2_7b_dutch" \
#             --dataset_name_or_path "../data/dutch/harmful_behaviors_custom_metadata.csv" \
#             --save_path "../data/dutch/" \
#             --tok_idxs -1 -2 -3 -4 -5 \
#             --padding_side "right" \
#             --file_spec "harmful_behaviors_custom_"

# python get_activations.py \
#             --model_name "llama2_7b_dutch" \
#             --dataset_name_or_path "../data/dutch/harmless_behaviors_custom_metadata.csv" \
#             --save_path "../data/dutch/" \
#             --tok_idxs -1 -2 -3 -4 -5 \
#             --padding_side "right" \
#             --file_spec "harmless_behaviors_custom_"