#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_ATTENTION_BACKEND=FLASHINFER   
MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct" #-FP8"
PORT=8400
TP=4

# Log GPU visibility for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

vllm serve "$MODEL_PATH" \
  --port $PORT \
  --dtype auto \
  --tensor-parallel-size $TP \
  --gpu-memory-utilization 0.6 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 \
  --disable-log-requests > vllm_server.log 2>&1 &

# Save the PID of the vLLM server
echo $! > vllm_server.pid