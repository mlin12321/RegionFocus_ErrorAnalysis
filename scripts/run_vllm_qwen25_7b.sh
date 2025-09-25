export CUDA_VISIBLE_DEVICES=4,5,6,7
vllm serve Qwen/Qwen2.5-VL-7B-Instruct   --port 8400   --dtype bfloat16   --limit-mm-per-prompt '{"images": 5}'   --tensor-parallel-size 4