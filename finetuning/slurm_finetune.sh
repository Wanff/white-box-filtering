#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=llamaguard_generalization_test_on_all_others
#SBATCH --output=llamaguard_generalization_test_on_all_others.out

# for seed in 0 1 2 3 4; do
#     python classifier_finetune.py \
#         --seed $seed
#     echo "Done with seed $seed"
# done

# CLASSIFIER GENERALIZATION

# LEAVE ONE OUT

# categories=("chemical_biological" "cybercrime_intrusion" "harassment_bullying" "harmful" "illegal" "misinformation_disinformation")
# for category in "${categories[@]}"; do
#     # run in dist
#     python classifier_finetune.py \
#         --path "../data/llama2_7b/generalization" \
#         --train_file_spec "${category}_train" \
#         --test_file_spec "${category}_test" 
#     echo "Done with category $category"
#     # run leave one out
#     python classifier_finetune.py \
#         --path "../data/llama2_7b/generalization" \
#         --train_file_spec "all_but_${category}_train" \
#         --test_file_spec "${category}_test"
#     echo "Done with leave one out category $category"
# done

# TEST ON ALL OTHERS

categories=("chemical_biological" "cybercrime_intrusion" "harassment_bullying" "harmful" "illegal" "misinformation_disinformation")
test_file_specs=()
for category in "${categories[@]}"; do
    test_file_specs+=("${category}_test")
done
test_file_specs_str="${test_file_specs[@]}"

for train_category in "${categories[@]}"; do
    python classifier_finetune.py \
        --path "../data/llama2_7b/generalization/test_on_others" \
        --train_file_spec "${train_category}_train" \
        --test_file_spec ${test_file_specs_str} \
        --no_save_at_end
    echo "Done with training category $train_category on all test categories"
done
