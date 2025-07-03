#    !/bin/bash
set -e

models=("qwen25vl_RegionFocus")
for model in "${models[@]}"
do
    python eval_screenspot_pro_RegionFocus.py  \
        --model_type ${model}  \
        --screenspot_imgs "./images"  \
        --screenspot_test "./annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/${model}.json" \
        --checkpoint_path "./results_mid/${model}.json" \
        --inst_style "instruction"

done