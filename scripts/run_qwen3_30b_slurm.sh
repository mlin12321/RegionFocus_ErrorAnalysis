#!/bin/bash
#SBATCH --account=PAS1576
#SBATCH --job-name=qwen3vl_rf
#SBATCH --time=00:10:00
#SBATCH --cluster=ascend
#SBATCH --exclusive
#SBATCH --output=slurm_out/%j_%x.slurm.out
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mlin36963@gmail.com

#set -x
module load cuda/12.8.1
module load miniconda3/24.1.2-py310
conda init
conda activate regionfocus

#echo $CONDA_PREFIX

#conda list

# Prevent pytorch from hogging mem
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set CUDA_VISIBLE_DEVICES to expose GPUs to Ray
# SLURM sets CUDA_VISIBLE_DEVICES, but we need to ensure it's exported
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Ray environment variables for SLURM
export RAY_DEDUP_LOGS=0
export VLLM_RAY_TIMEOUT=600
save_folder=/fs/ess/PAS1576/boyu_gou/gui_pretrain/RegionFocus_EA/results_qwen3_30b
work_dir=/fs/ess/PAS1576/boyu_gou/gui_pretrain/RegionFocus_EA
cd $work_dir

# Create save folder if it doesn't exist
mkdir -p $save_folder

# Function to copy files and cleanup, called on exit
cleanup_and_copy() {
    echo "====================================="
    echo "Copying output files to $save_folder..."
    echo "====================================="
    
    # Copy vLLM server log
    if [ -f vllm_server.log ]; then
        echo "Copying vllm_server.log..."
        cp vllm_server.log $save_folder/vllm_server.log
    else
        echo "Warning: vllm_server.log not found"
    fi
    
    # Copy evaluation output log
    if [ -f eval_output.log ]; then
        echo "Copying eval_output.log..."
        cp eval_output.log $save_folder/eval_output.log
    else
        echo "Warning: eval_output.log not found"
    fi
    
    # Copy result JSON files from results_qwen3vl_30b directory
    if [ -d results_qwen3vl_30b ]; then
        echo "Copying results from results_qwen3vl_30b/..."
        cp results_qwen3vl_30b/*.json $save_folder/ 2>/dev/null || echo "No JSON files in results_qwen3vl_30b/"
    else
        echo "Warning: results_qwen3vl_30b directory not found"
    fi
    
    # Copy checkpoint files from results_mid directory
    if [ -d results_mid ]; then
        echo "Copying checkpoint files from results_mid/..."
        mkdir -p $save_folder/results_mid
        cp results_mid/*.json $save_folder/results_mid/ 2>/dev/null || echo "No JSON files in results_mid/"
    else
        echo "Warning: results_mid directory not found"
    fi
    
    # Kill vLLM server if still running
    if [ -f vllm_server.pid ]; then
        echo "Cleaning up vLLM server..."
        kill $(cat vllm_server.pid) 2>/dev/null || true
        rm vllm_server.pid
    fi
    
    echo "====================================="
    echo "Cleanup and file copy complete."
    echo "Output saved to: $save_folder"
    echo "====================================="
}

# Set trap to ensure cleanup happens on exit (success or failure)
trap cleanup_and_copy EXIT
bash scripts/run_vllm_qwen3_30b.sh
echo "Waiting for vLLM server to start..."

timeout=300  # 5 minutes
elapsed=0
while true; do
    if [ -f vllm_server.log ] && grep -q "INFO: Application startup complete." vllm_server.log 2>/dev/null; then
        echo "vLLM server started successfully"
        break
    fi
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: vLLM server failed to start within $timeout seconds"
        if [ -f vllm_server.log ]; then
            echo "Last 50 lines of vllm_server.log:"
            tail -50 vllm_server.log
        fi
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "Still waiting... ($elapsed seconds elapsed)"
done

python test_script.py
echo "Running screenspot evaluation..."
echo "Evaluation output will be logged to eval_output.log"

# Run evaluation with full output logging (both stdout and stderr)
bash scripts/run_ss_pro_qwen3vl_RegionFocus_30b.sh 2>&1 | tee eval_output.log

echo "Evaluation complete!"
conda deactivate