# #!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_ATTENTION_BACKEND=FLASHINFER   
MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
PORT=8400
TP=4

vllm serve "$MODEL_PATH" \
  --port $PORT \
  --dtype auto \
  --tensor-parallel-size $TP \
  --gpu-memory-utilization 0.6 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144