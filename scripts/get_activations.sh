# python get_activations.py \
#     --model_name "llama2_7b" \
#     --dataset_name_or_path "../data/llama2_7b/jb_metadata.csv" \
#     --save_path "../data/llama2_7b/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "jb_" &> ../data/llama2_7b/jb.out &

# python get_activations.py \
#     --model_name "llama2_7b" \
#     --dataset_name_or_path "../data/llama2_7b/jb_metadata.csv" \
#     --save_path "../data/llama2_7b/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "jb_" &> ../data/llama2_7b/jb.out &
    
# python get_activations.py \
#     --model_name "llama2_7b" \
#     --dataset_name_or_path "../data/llama2_7b/jb_eq_by_jbname_metadata.csv" \
#     --save_path "../data/llama2_7b/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "jb_eq_by_jbname_" &> ../data/llama2_7b/jb_eq_by_jbname.out &

# python get_activations.py \
#     --model_name "llama2_7b" \
#     --dataset_name_or_path "../data/llama2_7b/all_gpt_gen_metadata.csv" \
#     --save_path "../data/llama2_7b/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --padding_side "right" \
#     --file_spec "all_gpt_gen_" &> ../data/llama2_7b/all_gpt_gen.out &

python get_activations.py \
    --model_name "llama2_7b" \
    --dataset_name_or_path "../data/llama2_7b/harmbench_alpaca_metadata.csv" \
    --save_path "../data/llama2_7b/" \
    --tok_idxs -1 -2 -3 -4 -5  \
    --padding_side "right" \
    --file_spec "all_gpt_gen_"

# TURKISH
# python get_activations.py \
#     --model_name "akdeniz27/llama-2-7b-hf-qlora-dolly15k-turkish" \
#     --dataset_name_or_path "../data/turkish/harmless_behaviors_custom.csv" \
#     --save_path "../data/turkish/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "harmless_behaviors_custom_"

# DUTCH
# python get_activations.py \
#     --model_name "llama2_7b_dutch" \
#     --dataset_name_or_path "../data/dutch/harmful_behaviors_custom_metadata.csv" \
#     --save_path "../data/dutch/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "harmful_behaviors_custom_"

# HUNGARIAN
# python get_activations.py \
#     --model_name "llama2_7b_hungarian" \
#     --dataset_name_or_path "../data/hungarian/harmful_behaviors_custom_metadata.csv" \
#     --save_path "../data/hungarian/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "harmful_behaviors_custom_"

# HUNGARIAN
# python get_activations.py \
#     --model_name "llama2_7b_slovenian" \
#     --dataset_name_or_path "../data/slovenian/harmless_behaviors_custom_metadata.csv" \
#     --save_path "../data/slovenian/" \
#     --tok_idxs -1 -2 -3 -4 -5  \
#     --file_spec "harmless_behaviors_custom_"
