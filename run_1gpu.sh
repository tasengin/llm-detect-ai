#!/bin/bash

#SBATCH --partition=kolyoz-cuda
#SBATCH --constraint=H100
#SBATCH --job-name=h100-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --error=./outputs/slurm-%j.err
#SBATCH --output=./outputs/slurm-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN

IMAGE="nvcr.io/nvidia/pytorch:25.10-py3"
WORKDIR=$PWD
DATADIR=$PWD

mkdir -p ./outputs

podman run --rm --ipc=host --network=host \
      --security-opt=label=disable \
      --device nvidia.com/gpu=all \
      -v ${WORKDIR}:${WORKDIR} \
      -v ${DATADIR}:${DATADIR} \
      -w ${WORKDIR} \
      $IMAGE \
      torchrun \
      --nproc_per_node=1 \
      multinode_torchrun.py  \
      --batch-size=128 \
      --epochs=10 \
      --model-size=xl \
      --lr=3e-4 \
      --save-checkpoint

exit
