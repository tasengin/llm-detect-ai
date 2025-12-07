# TRUBA Multi-GPU Training Scripts

Bu klasör TRUBA kümesinde multi-GPU training için hazırlanmış SLURM betiklerini içermektedir. Tüm betikler TRUBA'nın resmi örneklerinden (`multinode_torchrun.py`, `run_1gpu.sh`, `run_4gpu.sh`, `run_8gpu.sh`) adapte edilmiştir ve `torchrun` kullanmaktadır.

## Önemli Değişiklikler

### 1. `train_r_detect.py` Güncellemesi
- `setup_distributed()` fonksiyonu torchrun ile uyumlu hale getirildi
- Environment variables artık torchrun tarafından otomatik olarak ayarlanıyor
- `dist.init_process_group()` çağrısı torchrun'un beklediği formatta

### 2. Torchrun Kullanımı
Eski yaklaşım (srun ile doğrudan python):
```bash
srun python3 ./code/train_r_detect.py
```

Yeni yaklaşım (srun ile torchrun):
```bash
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=N ./code/train_r_detect.py
```

## Kullanılabilir SLURM Betikleri

### Barbun Cluster (2x V100 per node)
- `barbun_1gpu.slurm` - Tek GPU (1 node, 1 GPU)
- `barbun_2gpu.slurm` - Tek node, 2 GPU (node başına max GPU)
- `barbun.slurm` - Multi-node (2 node, toplam 4 GPU)

### Akya Cluster (4x V100 per node)
- `akya_1gpu.slurm` - Tek GPU (1 node, 1 GPU)
- `akya_2gpu.slurm` - Tek node, 2 GPU
- `akya_4gpu.slurm` - Tek node, 4 GPU (node başına max GPU)
- `akya.slurm` - Multi-node (2 node, toplam 8 GPU)

## Kullanım

### Tek GPU Training
```bash
sbatch barbun_1gpu.slurm  # veya
sbatch akya_1gpu.slurm
```

### Single Node Multi-GPU Training
```bash
sbatch barbun_2gpu.slurm  # barbun için (max 2 GPU)
sbatch akya_4gpu.slurm    # akya için (max 4 GPU)
```

### Multi-Node Multi-GPU Training
```bash
sbatch barbun.slurm       # 2 node x 2 GPU = 4 GPU
sbatch akya.slurm         # 2 node x 4 GPU = 8 GPU
```

## Konfigürasyon Değiştirme

Config dosyasını değiştirmek için slurm betiğinin sonundaki satırı düzenleyin:

```bash
--config-name conf_r_detect_mix_v16 \
```

Diğer parametreleri de ekleyebilirsiniz:
```bash
--config-name conf_r_detect_mix_v16 \
use_wandb=true \
train_params.per_device_train_batch_size=8
```

## Önemli Environment Variables

Her betik şu environment variables'ları ayarlar:
- `HYDRA_FULL_ERROR=1` - Hydra hatalarını detaylı gösterir
- `TRUBA=1` - TRUBA cluster'da olduğumuzu belirtir
- `CLUSTER=barbun|akya` - Hangi cluster'da çalıştığımızı belirtir
- `OMP_NUM_THREADS` - OpenMP thread sayısı
- `NCCL_DEBUG=WARN` - NCCL debug seviyesi
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - CUDA memory allocation optimizasyonu

## Multi-Node Training Detayları

Multi-node training için torchrun şu parametreleri kullanır:
- `--nnodes`: Toplam node sayısı
- `--nproc_per_node`: Node başına GPU/process sayısı
- `--node_rank`: Mevcut node'un sırası (otomatik: `$SLURM_NODEID`)
- `--rdzv_id`: Rendezvous ID (job ID kullanılır)
- `--rdzv_backend`: c10d (PyTorch distributed backend)
- `--rdzv_endpoint`: Master node adresi ve portu

## Log Dosyaları

Tüm loglar `slurm_logs/` klasöründe saklanır:
- `<cluster>-<gpu_count>-training-<job_id>.out` - Standard output
- `<cluster>-<gpu_count>-training-<job_id>.err` - Standard error

## Troubleshooting

### NCCL Hatası
Eğer NCCL timeout hatası alırsanız:
1. `NCCL_DEBUG=INFO` yapın daha detaylı log için
2. InfiniBand ayarlarını kontrol edin

### Out of Memory
Eğer GPU memory yetersiz ise:
1. Config dosyasında `per_device_train_batch_size` değerini düşürün
2. `gradient_accumulation_steps` değerini artırın
3. Gradient checkpointing'i aktif edin

### Distributed Training Sync Problemi
Eğer processler senkronize olamıyorsa:
1. `MASTER_ADDR` ve `MASTER_PORT` değerlerini kontrol edin
2. Firewall kurallarını kontrol edin
3. `dist.barrier()` çağrılarının doğru yerlerde olduğundan emin olun

## Örnek İş Akışı

1. Kodu test etmek için önce tek GPU ile başlayın:
   ```bash
   sbatch barbun_1gpu.slurm
   ```

2. Başarılı olursa, tek node multi-GPU deneyin:
   ```bash
   sbatch barbun_2gpu.slurm
   ```

3. Her şey yolundaysa, multi-node training yapın:
   ```bash
   sbatch barbun.slurm
   ```

## Notlar

- Apptainer (Singularity) container kullanılıyor: `llm_detect.sif`
- Veri setleri `/datasets` altında mount ediliyor
- Çıktılar `/outputs` altında saklanıyor
- Her betik otomatik olarak gerekli klasörleri oluşturur

## İletişim

Herhangi bir sorun için TRUBA destek ekibiyle iletişime geçin veya projenin issue tracker'ını kullanın.
