#!/bin/bash

#SBATCH --account=iguzel
#SBATCH --partition=kolyoz-cuda
#SBATCH --constraint=H100
#SBATCH --job-name=h100-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --error=./outputs/slurm-%j.err
#SBATCH --output=./outputs/slurm-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN

# Master node IP'sini bu
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${nodes[0]}
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of nodes: $SLURM_NNODES"

mkdir -p ./outputs


WORKDIR=$PWD
DATADIR=$PWD
IMAGE=nvcr.io/nvidia/pytorch:25.10-py3

# her node srun ile, node ici torchrun ile başlat
srun --ntasks-per-node=1 bash -c "
podman run --rm --ipc=host --network=host \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -v ${WORKDIR}:${WORKDIR} \
  -v ${DATADIR}:${DATADIR} \
  -w ${WORKDIR} \
  $IMAGE \
  torchrun \
       --nnodes=$SLURM_NNODES \
       --nproc_per_node=4 \
       --node_rank=$SLURM_NODEID \
       --rdzv_id=job_$SLURM_JOB_ID \
       --rdzv_backend=c10d \
       --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
       multinode_torchrun.py \
       --batch-size=128 \
       --epochs=10 \
       --model-size=xl \
       --lr=3e-4 \
       --save-checkpoint
     "
exit
