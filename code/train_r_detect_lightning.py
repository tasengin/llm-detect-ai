import os
import torch
import hydra
from omegaconf import DictConfig
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# PyTorch Lightning importları
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy

# Kendi yazdığınız diğer modülleri import edin
# Bu dosyaların `code` dizini altında doğru yerlerde olduğundan emin olun
from r_detect.ai_model import MistralForDetectAI 
from r_detect.ai_dataset import LLMDataset         

# ==============================================================================
# 1. LIGHTNING DATAMODULE: Veri yönetimi
# ==============================================================================
class LLMDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        # Bu, konfigürasyonun checkpoint ile kaydedilmesini sağlar.
        self.save_hyperparameters(ignore=['tokenizer'])

    def setup(self, stage: str):
        # Bu metod, her bir node üzerinde Lightning tarafından çağrılır.
        if stage == 'fit':
            train_path = os.path.join(self.cfg.data_dir, "train_essays.csv")
            train_essay_df = pd.read_csv(train_path)
            self.train_dataset = LLMDataset(train_essay_df, self.tokenizer, self.cfg)

            val_path = os.path.join(self.cfg.data_dir, "val_essays.csv")
            val_essay_df = pd.read_csv(val_path)
            self.val_dataset = LLMDataset(val_essay_df, self.tokenizer, self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

# ==============================================================================
# 2. LIGHTNING MODULE: Model ve Eğitim Mantığı
# ==============================================================================
class MistralLightningModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Model tanımı burada yapılır. Lightning, bu modeli FSDP ile sarmalayacaktır.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            # device_map KALDIRILDI! Bu çok önemli. Kontrol tamamen FSDP'de.
        )
        self.model = MistralForDetectAI(base_model, self.cfg)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # Bir eğitim adımında yapılacaklar:
        outputs = self(**batch)
        loss = outputs.loss
        
        # Loglama: 'sync_dist=True' ile tüm GPU'lardaki loss değerlerinin ortalaması alınır.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Bir doğrulama adımında yapılacaklar:
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Optimizer ve Scheduler burada tanımlanır.
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(), # self.parameters() yerine bu daha güvenlidir.
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.optimizer.lr,
            total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

# ==============================================================================
# 3. ANA ÇALIŞTIRMA FONKSİYONU
# ==============================================================================
@hydra.main(config_path="../conf", config_name="conf_r_detect_mix_v16", version_base=None)
def run_training(cfg: DictConfig):
    # Ayarları ekrana yazdır (sadece Rank 0'da)
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        print("--- Yapılandırma Ayarları ---")
        print(cfg)
        print("-----------------------------")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = tokenizer.eos_token

    data_module = LLMDataModule(cfg, tokenizer)
    model_module = MistralLightningModule(cfg)

    # FSDP için auto-wrap policy'sini tanımla (Transformer bloklarını otomatik bulur)
    # Mistral'in blok adını import etmemiz gerekiyor.
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={MistralDecoderLayer})

    # Lightning Trainer'ı yapılandır.
    trainer = pl.Trainer(
        num_nodes=cfg.training.num_nodes,
        devices="auto",
        accelerator="cuda",
        strategy=FSDPStrategy(auto_wrap_policy=auto_wrap_policy),
        precision="16-mixed",
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=cfg.optimizer.max_grad_norm,
        max_epochs=cfg.training.epochs,
        default_root_dir=cfg.outputs_dir,
    )
    
    # EĞİTİMİ BAŞLAT
    trainer.fit(model_module, datamodule=data_module)

if __name__ == '__main__':
    run_training()
