#!/bin/bash
# SLURM script for training on Truba akya-cuda cluster
# Cluster specs: 4x NVIDIA V100 16GB per node, 40 CPU cores per node
# Multi-node multi-GPU training with torchrun

#SBATCH --account=etas
#SBATCH --partition=akya-cuda
#SBATCH --qos=root
#SBATCH --nodes=2                  # 2 nodes
#SBATCH --ntasks-per-node=1        # 1 task per node (torchrun will handle GPUs)
#SBATCH --gres=gpu:4               # 4 GPUs per node
#SBATCH --cpus-per-task=40         # 40 CPUs per task
#SBATCH --time=3-00:00:00          # Maximum 3 days
#SBATCH --job-name=r-detect-akya
#SBATCH --chdir=/arf/scratch/etas/llm-detect-ai
#SBATCH --output=slurm_logs/akya-training-%j.out
#SBATCH --error=slurm_logs/akya-training-%j.err
#SBATCH --mail-user=tasengin@hotmail.com
#SBATCH --mail-type=ALL

# Environment variables
export HYDRA_FULL_ERROR=1
export TRUBA=1
export CLUSTER=akya
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# NCCL settings optimized for Truba InfiniBand network
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5

# HuggingFace cache
export HF_DATASETS_CACHE=/datasets/hf_cache

# Get the first node name as master address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Start Time: $(date --utc)"
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "=========================================="

mkdir -p ./outputs
mkdir -p ./slurm_logs

# Launch training with srun + torchrun for multi-node execution
srun --ntasks-per-node=1 bash -c "
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export RANK=\$SLURM_NODEID
export WORLD_SIZE=\$SLURM_NNODES

apptainer exec \
    --nv \
    -B /arf/scratch/etas/llm-detect-ai/outputs:/outputs \
    -B /arf/scratch/etas/datasets:/datasets \
    llm_detect.sif \
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --node_rank=\$SLURM_NODEID \
        --rdzv_id=job_$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        ./code/train_r_detect.py \
        --config-name conf_r_detect_mix_v16 \
        use_wandb=false
"

echo "End Time: $(date --utc)"
