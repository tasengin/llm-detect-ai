#!/bin/bash
# SLURM script for training on Truba akya-cuda cluster - 2 Nodes, 8 GPUs total
# Cluster specs: 4x NVIDIA V100 16GB per node, 40 CPU cores per node

#SBATCH --account=etas
#SBATCH --partition=akya-cuda
#SBATCH --qos=root
#SBATCH --nodes=2                  # 2 nodes
#SBATCH --ntasks-per-node=4        # 4 tasks per node (1 per GPU)
#SBATCH --gres=gpu:4               # 4 GPUs per node (8 GPUs total)
#SBATCH --cpus-per-task=10         # 10 CPUs per task (akya policy)
#SBATCH --time=3-00:00:00          # Maximum 3 days
#SBATCH --job-name=rd-akya-8gpu
#SBATCH --chdir=/arf/scratch/etas/llm-detect-ai
#SBATCH --output=slurm_logs/akya-8gpu-training-%j.out
#SBATCH --error=slurm_logs/akya-8gpu-training-%j.err
#SBATCH --mail-user=tasengin@hotmail.com
#SBATCH --mail-type=ALL

# Environment variables
export HYDRA_FULL_ERROR=1
export TRUBA=1
export CLUSTER=akya
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0

# HuggingFace cache
export HF_DATASETS_CACHE=/datasets/hf_cache

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date --utc)"
echo "GPUs per node: 4 (total 8)"
echo "=========================================="

# Do not create directories in slurm scripts per cluster policy

# Launch training with torchrun across 2 nodes, 4 GPUs per node
apptainer exec \
    --nv \
    -B /arf/scratch/etas/llm-detect-ai/outputs:/outputs \
    -B /arf/scratch/etas/datasets:/datasets \
    llm_detect.sif \
    torchrun \
        --nproc_per_node=4 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
        --master_port=29500 \
        ./code/train_r_detect.py \
        --config-name conf_r_detect_mix_v16 \
        use_wandb=false

echo "End Time: $(date --utc)"
