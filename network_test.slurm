#!/bin/bash
#SBATCH --account=etas
#SBATCH --partition=debug
#SBATCH --qos=root
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=net_test
#SBATCH --output=net_test.out
#SBATCH --error=net_test.err

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((29500 + RANDOM % 1000))

srun python network_test.py
