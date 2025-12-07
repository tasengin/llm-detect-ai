#!/bin/bash
# SLURM script for training on Truba barbun-cuda cluster - Single GPU
# Cluster specs: 2x NVIDIA V100 16GB per node, 40 CPU cores per node

#SBATCH --account=etas
#SBATCH --partition=barbun-cuda
#SBATCH --qos=root
#SBATCH --nodes=1                  # 1 node
#SBATCH --ntasks-per-node=1        # 1 task per node
#SBATCH --gres=gpu:1               # 1 GPU
#SBATCH --cpus-per-task=20         # 20 CPUs per task
#SBATCH --time=3-00:00:00          # Maximum 3 days
#SBATCH --job-name=rd-barbun-1gpu
#SBATCH --chdir=/arf/scratch/etas/llm-detect-ai
#SBATCH --output=slurm_logs/barbun-1gpu-training-%j.out
#SBATCH --error=slurm_logs/barbun-1gpu-training-%j.err
#SBATCH --mail-user=tasengin@hotmail.com
#SBATCH --mail-type=ALL

# Environment variables
export HYDRA_FULL_ERROR=1
export TRUBA=1
export CLUSTER=barbun
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# NCCL settings
export NCCL_DEBUG=WARN

# HuggingFace cache
export HF_DATASETS_CACHE=/datasets/hf_cache

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date --utc)"
echo "GPUs: 1"
echo "=========================================="

mkdir -p ./outputs
mkdir -p ./slurm_logs

# Launch training with torchrun for single GPU
apptainer exec \
    --nv \
    -B /arf/scratch/etas/llm-detect-ai/outputs:/outputs \
    -B /arf/scratch/etas/datasets:/datasets \
    llm_detect.sif \
    torchrun \
        --nproc_per_node=1 \
        ./code/train_r_detect.py \
        --config-name conf_r_detect_mix_v16 \
        use_wandb=false

echo "End Time: $(date --utc)"
