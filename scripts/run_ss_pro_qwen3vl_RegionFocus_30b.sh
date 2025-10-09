#!/bin/bash
set -e

# Create output directories if they don't exist
mkdir -p ./results_qwen3vl_30b
mkdir -p ./results_mid

models=("qwen3vl_RegionFocus_30b")
for model in "${models[@]}"
do
    echo "====================================="
    echo "Starting evaluation for model: ${model}"
    echo "====================================="
    
    python eval_screenspot_pro_RegionFocus.py  \
        --model_type ${model}  \
        --screenspot_imgs "../ScreenSpot-Pro/images"  \
        --screenspot_test "../ScreenSpot-Pro/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results_qwen3vl_30b/${model}.json" \
        --checkpoint_path "./results_mid/${model}.json" \
        --inst_style "instruction" \
        --debug
    
    echo "====================================="
    echo "Completed evaluation for model: ${model}"
    echo "Results saved to: ./results_qwen3vl_30b/${model}.json"
    echo "====================================="

done