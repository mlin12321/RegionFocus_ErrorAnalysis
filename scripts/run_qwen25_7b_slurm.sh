#!/bin/bash
#SBATCH --account=PAS1576
#SBATCH --job-name=qwen25vl_rf
#SBATCH --time=00:05:00
#SBATCH --cluster=ascend
#SBATCH --exclusive
#SBATCH --output=%j_%x.slurm.out
#SBATCH --gpus-per-node=1
#SBATCH --mem=40g
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
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
save_folder=/fs/ess/PAS1576/boyu_gou/gui_pretrain/RegionFocus_EA/results_qwen25vl_7b
cd /fs/ess/PAS1576/boyu_gou/gui_pretrain/RegionFocus_EA
bash scripts/run_vllm_qwen25_7b.sh
echo "Waiting for vLLM server to start..."

timeout=300  # 5 minutes
elapsed=0
while ! grep -q "INFO: Application startup complete." vllm_server.log; do
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: vLLM server failed to start within $timeout seconds"
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
echo "vLLM server started successfully"

echo "Running screenspot evaluation..."
bash scripts/run_ss_pro_qwen25vl_RegionFocus_7b.sh
conda deactivate 
ls
cd $TMPDIR
ls
#echo "sucess!"
cp -r * $save_folder 