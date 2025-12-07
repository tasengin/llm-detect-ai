#!/bin/bash
# SLURM script for training on Truba barbun-cuda cluster

#SBATCH --account=etas
#SBATCH --partition=barbun-cuda
#SBATCH --qos=root
#SBATCH --nodes=2                 # 4 nodes
#SBATCH --ntasks-per-node=2        # 1 task per node to launch torchrun
#SBATCH --gpus-per-node=2          # 2 GPUs per node
#SBATCH --cpus-per-task=20         # Allocate CPUs for the torchrun process and its workers
#SBATCH --time=3-00:00:00          # Maximum 3 days
#SBATCH --job-name=rd-barbun
#SBATCH --chdir=/arf/scratch/etas/llm-detect-ai
#SBATCH --output=slurm_logs/barbun-training-%j.out
#SBATCH --error=slurm_logs/barbun-training-%j.err
#SBATCH --mail-user=tasengin@hotmail.com
#SBATCH --mail-type=ALL

# Log directories are assumed to already exist (no mkdir in SLURM)

# Environment variables for Truba
export HYDRA_FULL_ERROR=1
export TRUBA=1
export CLUSTER=barbun
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL settings optimized for Truba InfiniBand network
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5

# HuggingFace cache (host path, bind-mounted to /datasets)
export HF_DATASETS_CACHE=/datasets/hf_cache

# Get the first node name as master address
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${nodes[0]}
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Start Time: $(date --utc)"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "=========================================="

WORKDIR=$PWD
IMAGE=llm_detect.sif

srun --ntasks-per-node=1 bash -c "
apptainer exec --nv \
  -B ${WORKDIR}:${WORKDIR} \
  -B /arf/scratch/etas/datasets:/datasets \
  -B /arf/scratch/etas/outputs:/outputs \
  $IMAGE \
  torchrun \
       --nnodes=$SLURM_NNODES \
       --nproc_per_node=$SLURM_GPUS_PER_NODE \
       --node_rank=$SLURM_NODEID \
       --rdzv_id=job_$SLURM_JOB_ID \
       --rdzv_backend=c10d \
       --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
       ./code/train_r_detect.py --config-name conf_r_detect_mix_v16 use_wandb=false
"

echo "End Time: $(date --utc)"
