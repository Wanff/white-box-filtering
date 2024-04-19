SAVE_PATH="../data/llama2_7b"
FILE_SPEC="gcg_run_lyr16_acts_post_suffix"
# python run_gcg.py \
#     --model_name "llama2_7b" \
#     --save_path "$SAVE_PATH" \
#     --monitor_type "act" \
#     --probe_data_path "../data/llama2_7b/jb_" \
#     --probe_layer 16 \

#     --file_spec "$FILE_SPEC"  &> "$SAVE_PATH/$FILE_SPEC.out" &

python run_gcg.py \
    --model_name "llama2_7b" \
    --save_path "$SAVE_PATH" \
    --monitor_type "act" \
    --probe_data_path "../data/llama2_7b/jb_" \
    --probe_layer 16 \
    --file_spec "$FILE_SPEC"  &> "$SAVE_PATH/$FILE_SPEC.out" &