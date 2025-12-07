# Truba Cluster Setup Guide

This document provides detailed instructions for running the llm-detect-ai codebase on Truba's GPU clusters.

## Cluster Specifications

### akya-cuda
- **GPUs**: 4x NVIDIA V100 16GB per node (with NVLink)
- **CPUs**: 2x Intel Xeon Gold 6248R (40 cores per node)
- **RAM**: 384 GB per node
- **Minimum per task**: 10 CPUs + 1 GPU
- **Max job duration**: 3 days

### barbun-cuda
- **GPUs**: 2x NVIDIA P100 16GB per node
- **CPUs**: 2x Intel Xeon Gold 6248R (40 cores per node)
- **RAM**: 384 GB per node
- **Minimum per task**: 20 CPUs + 1 GPU
- **Max job duration**: 3 days

## Available SLURM Scripts

### Main Training Scripts

| Script | Cluster | Nodes | GPUs | Description |
|--------|---------|-------|------|-------------|
| `akya.slurm` | akya-cuda | 4 | 16 | LLM detection model training |
| `barbun.slurm` | barbun-cuda | 2 | 4 | LLM detection model training |
| `akya_ranking.slurm` | akya-cuda | 4 | 16 | DeBERTa ranking model training |
| `barbun_ranking.slurm` | barbun-cuda | 2 | 4 | DeBERTa ranking model training |
| `akya_embed.slurm` | akya-cuda | 4 | 16 | Embedding model training |
| `barbun_embed.slurm` | barbun-cuda | 2 | 4 | Embedding model training |
| `akya2.slurm` | akya-cuda | 1 | 4 | PyTorch Lightning training (single node) |

## Quick Start

1. **Ensure datasets are available:**
   ```bash
   ./setup.sh
   ```

2. **Submit a job:**
   ```bash
   # For LLM detection on akya-cuda
   sbatch akya.slurm
   
   # For LLM detection on barbun-cuda
   sbatch barbun.slurm
   ```

3. **Monitor your job:**
   ```bash
   squeue -u $USER
   ```

4. **View logs:**
   ```bash
   tail -f slurm_logs/akya-training-<JOB_ID>.out
   ```

## FSDP Configuration Files

- `conf/fsdp_config_akya.yaml`: Configured for 4 nodes × 4 GPUs = 16 total GPUs
- `conf/fsdp_config_barbun.yaml`: Configured for 2 nodes × 2 GPUs = 4 total GPUs

These are automatically used by the respective SLURM scripts.

## Environment Variables

The SLURM scripts set the following optimized environment variables:

- `NCCL_DEBUG=INFO`: Enable NCCL debugging
- `NCCL_SOCKET_IFNAME=ib0`: Use InfiniBand interface
- `NCCL_IB_HCA=mlx5`: Specify InfiniBand HCA
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Optimize CUDA memory allocation
- `TRUBA=1`: Enable Truba-specific optimizations in code

## Customizing Jobs

To modify training parameters, edit the SLURM script and change the Hydra config overrides:

```bash
python3 ./code/train_r_detect.py \
    --config-name conf_r_detect_mix_v16 \
    use_wandb=false \
    train_params.num_train_epochs=2  # Add your overrides here
```

## Troubleshooting

### Job fails with CUDA OOM
- Reduce batch size in the config file
- Increase gradient accumulation steps
- Use fewer nodes/GPUs

### NCCL errors
- Check InfiniBand connectivity: `ibstatus`
- Verify NCCL environment variables are set correctly
- Check SLURM log files for detailed error messages

### Slow data loading
- Increase `num_workers` in the config file
- Check dataset location and mount points
- Verify disk I/O performance

## References

- [Truba Documentation](https://docs.truba.gov.tr/)
- [Truba Cluster Information](https://docs.truba.gov.tr/2-temel_bilgiler/hesaplama_kumeleri.html)

