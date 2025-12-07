#!/bin/bash
#SBATCH --account=etas
#SBATCH --partition=akya-cuda
#SBATCH --qos=tom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=debug_net
#SBATCH --output=debug_net.out
#SBATCH --error=debug_net.err

srun apptainer exec --nv llm_detect.sif ls /sys/class/net
