#!/bin/bash
# SLURM script for DeBERTa ranking model training on Truba akya-cuda cluster

#SBATCH --account=etas
#SBATCH --partition=akya-cuda
#SBATCH --qos=tom
#SBATCH --nodes=4                  # 4 nodes
#SBATCH --ntasks-per-node=4        # 4 tasks per node (1 per GPU)
#SBATCH --cpus-per-task=10         # 10 CPUs per task
#SBATCH --gres=gpu:4               # 4 GPUs per node
#SBATCH --time=3-00:00:00          # Maximum 3 days
#SBATCH --job-name=r-ranking-akya
#SBATCH --chdir=/arf/scratch/etas/llm-detect-ai
#SBATCH --output=slurm_logs/akya-ranking-%j.out
#SBATCH --error=slurm_logs/akya-ranking-%j.err
#SBATCH --mail-user=tasengin@hotmail.com
#SBATCH --mail-type=ALL

mkdir -p slurm_logs

export HYDRA_FULL_ERROR=1
export TRUBA=1
export CLUSTER=akya
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5

export HF_DATASETS_CACHE=/datasets/hf_cache
mkdir -p /datasets/hf_cache

CONFIG_PATH="/arf/scratch/etas/llm-detect-ai/conf/fsdp_config_akya.yaml"

# Get the first node name as master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=29500

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Start Time: $(date --utc)"
echo "Training: DeBERTa Ranking Model"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "=========================================="

srun apptainer exec \
    --nv \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    -B /arf/scratch/etas/llm-detect-ai/outputs:/outputs \
    -B /arf/scratch/etas/datasets:/datasets \
    -B $CONFIG_PATH:/opt/fsdp_config.yaml \
    llm_detect.sif \
    bash -c 'export RANK=$SLURM_PROCID; export LOCAL_RANK=$SLURM_LOCALID; export WORLD_SIZE=$SLURM_NTASKS; python3 ./code/train_r_ranking.py --config-name conf_r_ranking_large use_wandb=false'

echo "End Time: $(date --utc)"

