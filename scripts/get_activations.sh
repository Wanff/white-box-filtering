#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
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

datasets=("harmbench_alpaca_metadata" "harmbench_alpaca_test_metadata" "generated_metadata" "generated_test_metadata")

for dataset in "${datasets[@]}"; do
    file_spec="${dataset%_metadata}"  # Removes '_metadata' from the dataset name to form the file_spec prefix
    python get_activations.py \
        --model_name "llama2_7b" \
        --dataset_name_or_path "../data/llama2_7b/${dataset}.csv" \
        --save_path "../data/llama2_7b/" \
        --tok_idxs -1 -2 -3 -4 -5 \
        --padding_side "right" \
        --file_spec "${file_spec}_"
    echo "Done with $dataset"
done


# BATTERY FOR LANGUAGE EXPERIMENTS

# languages=("dutch" "hungarian" "slovenian")
# behaviors=("harmful" "harmless")

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
# done
