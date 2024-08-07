#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=llamaguard_1_epoch
#SBATCH --output=llamaguard_1_epoch.out

# SIMPLE FINETUNE

python classifier_finetune.py

# FULL FINETUNE WITH SEEDS

# for seed in 0 1 2 3 4; do
#     python classifier_finetune.py \
#         --seed $seed
#     echo "Done with seed $seed"
# done

# SMALL CLASSIFIER FINETUNE

# for seed in 0 1 2 3 4; do
#     python classifier_finetune.py \
#         --seed $seed \
#         --model_name "gemma-2b" \
#         --head
#     echo "Done with seed $seed"
# done

# LLAMA2 INSTEAD OF LLAMAGUARD

# python classifier_finetune.py --head

#LLAMA2-13b
# python classifier_finetune.py --model_name "llama-2-13b-for-harm-classification" --path "../data/llama2_13b" --use_peft


# CLASSIFIER GENERALIZATION

# LEAVE ONE OUT

# categories=("chemical_biological" "cybercrime_intrusion" "harassment_bullying" "harmful" "illegal" "misinformation_disinformation")
# for category in "${categories[@]}"; do
#     # run in dist
#     python classifier_finetune.py \
#         --path "../data/llama2_7b/generalization/leave_one_out" \
#         --train_file_spec "${category}_train" \
#         --test_file_spec "${category}_test" \
#         --no_save_at_end
#     echo "Done with category $category"
#     # run leave one out
#     python classifier_finetune.py \
#         --path "../data/llama2_7b/generalization/leave_one_out" \
#         --train_file_spec "all_but_${category}_train" \
#         --test_file_spec "${category}_test" \
#         --no_save_at_end
#     echo "Done with leave one out category $category"
# done

# TEST ON ALL OTHERS

# categories=("chemical_biological" "cybercrime_intrusion" "harassment_bullying" "harmful" "illegal" "misinformation_disinformation")
# test_file_specs=()
# for category in "${categories[@]}"; do
#     test_file_specs+=("${category}_test")
# done
# test_file_specs_str="${test_file_specs[@]}"

# for train_category in "${categories[@]}"; do
#     python classifier_finetune.py \
#         --path "../data/llama2_7b/generalization/test_on_others" \
#         --train_file_spec "${train_category}_train" \
#         --test_file_spec ${test_file_specs_str} \
#         --no_save_at_end
#     echo "Done with training category $train_category on all test categories"
# done

# # CIPHER FINETUNE

# python cipher_finetune.py --save_per_epoch