#!/bin/bash
# SLURM script for embedding model training on Truba barbun-cuda cluster

#SBATCH --account=etas
#SBATCH --partition=barbun-cuda
#SBATCH --qos=tom
#SBATCH --nodes=2                  # 2 nodes
#SBATCH --ntasks-per-node=2        # 2 tasks per node (1 per GPU)
#SBATCH --cpus-per-task=20         # 20 CPUs per task
#SBATCH --gres=gpu:2               # 2 GPUs per node
#SBATCH --time=3-00:00:00          # Maximum 3 days
#SBATCH --job-name=r-embed-barbun
#SBATCH --chdir=/arf/scratch/etas/llm-detect-ai
#SBATCH --output=slurm_logs/barbun-embed-%j.out
#SBATCH --error=slurm_logs/barbun-embed-%j.err
#SBATCH --mail-user=tasengin@hotmail.com
#SBATCH --mail-type=ALL

mkdir -p slurm_logs

export HYDRA_FULL_ERROR=1
export TRUBA=1
export CLUSTER=barbun
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
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

CONFIG_PATH="/arf/scratch/etas/llm-detect-ai/conf/fsdp_config_barbun.yaml"

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Start Time: $(date --utc)"
echo "Training: Embedding Model"
echo "=========================================="

srun apptainer exec \
    --nv \
    -B /arf/scratch/etas/llm-detect-ai/outputs:/outputs \
    -B /arf/scratch/etas/datasets:/datasets \
    -B $CONFIG_PATH:/opt/fsdp_config.yaml \
    llm_detect.sif \
    python3 ./code/train_r_embed.py \
    --config-name conf_r_embed use_wandb=false

echo "End Time: $(date --utc)"

