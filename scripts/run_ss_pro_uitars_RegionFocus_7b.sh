#    !/bin/bash
set -e

models=("uitars_RegionFocus_7b")
for model in "${models[@]}"
do
    python eval_screenspot_pro_RegionFocus.py  \
        --model_type ${model}  \
        --screenspot_imgs "../ScreenSpot-Pro/images"  \
        --screenspot_test "../ScreenSpot-Pro/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/${model}.json" \
        --checkpoint_path "./results_mid/${model}.json" \
        --inst_style "instruction"

done