import logging
import os
import random
import time
import datetime
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict

import bitsandbytes as bnb
import datasets
import hydra
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import transformers
from omegaconf import OmegaConf
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)


try:
    from r_detect.ai_dataset import AiDataset
    from r_detect.ai_loader import AiCollator, AiCollatorTrain, show_batch
    from r_detect.ai_model import (LlamaForDetectAI, MistralForDetectAI,
                                   PhiForDetectAI)
    from r_detect.ai_optimizer import get_optimizer
    from utils.metric_utils import compute_metrics
    from utils.train_utils import AverageMeter, as_minutes, get_lr


except Exception as e:
    print(e)
    raise ImportError

logger = logging.getLogger(__name__)


def setup_distributed():
    """
    Sets up the distributed training environment.
    Uses environment variables set by torchrun.
    """
    # Get environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    logger.info(f"local_rank: {local_rank}, world_size: {world_size}, rank: {rank}")

    # Set the GPU device for this process
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    # Initialize process group
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))

    return local_rank, rank, world_size


def cleanup():
    """
    Cleans up the distributed training environment.
    """
    dist.destroy_process_group()


def _to_plain_dict(config_section) -> Dict[str, Any]:
    if config_section is None:
        return {}
    if isinstance(config_section, dict):
        return config_section
    return OmegaConf.to_container(config_section, resolve=True)


def _detect_truba_v100() -> bool:
    device_name = ""
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0).lower()
        except RuntimeError:
            device_name = ""

    env_tokens = " ".join(
        [
            os.environ.get("CLUSTER", ""),
            os.environ.get("SLURM_CLUSTER_NAME", ""),
            os.environ.get("HOSTNAME", ""),
        ]
    ).lower()
    explicit_flag = os.environ.get("TRUBA", "").lower() in ("1", "true", "yes")
    return ("v100" in device_name) and (explicit_flag or ("truba" in env_tokens))


def _resolve_hardware_cfg(cfg) -> Dict[str, Any]:
    base_cfg = {
        "train_num_workers": 4,
        "eval_num_workers": 2,
        "train_prefetch_factor": None,
        "eval_prefetch_factor": None,
        "pin_memory": True,
        "persistent_workers": True,
        "gradient_checkpointing": False,
        "compile_model": False,
        "compile_backend": "inductor",
        "override_train_batch_size": None,
        "override_eval_batch_size": None,
        "override_grad_accum_steps": None,
        "set_cudnn_benchmark": True,
        "auto_detect_truba": False,
        "force_profile": None,
        "truba_v100": {},
    }

    if "hardware" in cfg:
        user_cfg = _to_plain_dict(cfg.hardware)
        for key in base_cfg.keys():
            if key in user_cfg and user_cfg[key] is not None:
                base_cfg[key] = user_cfg[key]

        # allow nested overrides such as hardware.profiles.truba_v100
        truba_profile = user_cfg.get("truba_v100", {})
        if isinstance(truba_profile, dict):
            base_cfg["truba_v100"] = truba_profile

    apply_truba = False
    if base_cfg.get("force_profile") == "truba_v100":
        apply_truba = True
    elif base_cfg.get("auto_detect_truba"):
        apply_truba = _detect_truba_v100()

    if apply_truba and base_cfg["truba_v100"]:
        for key, value in base_cfg["truba_v100"].items():
            if value is not None:
                base_cfg[key] = value

    return base_cfg


def run_evaluation(model, valid_dl, valid_ids, local_rank, rank, world_size):
    model.eval()

    all_predictions = []
    all_truths = []

    progress_bar = tqdm(range(len(valid_dl)), disable=rank != 0)

    for step, batch in enumerate(valid_dl):
        # Move batch to the local device for this process
        batch = {k: v.to(local_rank) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.sigmoid(logits)

        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_truths.extend(batch["labels"].cpu().numpy().tolist())

        progress_bar.update(1)

    progress_bar.close()

    # Gather results from all processes
    gathered_predictions = [None] * world_size
    gathered_truths = [None] * world_size
    dist.all_gather_object(gathered_predictions, all_predictions)
    dist.all_gather_object(gathered_truths, all_truths)

    if rank == 0:
        # Flatten the gathered lists of lists
        flat_predictions = [item for sublist in gathered_predictions for item_list in sublist for item in item_list]
        flat_truths = [item for sublist in gathered_truths for item_list in sublist for item in item_list]

        # compute metric
        eval_dict = compute_metrics(flat_predictions, flat_truths)

        result_df = pd.DataFrame()
        result_df["id"] = valid_ids[:len(flat_predictions)]
        result_df["predictions"] = flat_predictions
        result_df["truths"] = flat_truths

        oof_df = deepcopy(result_df)
        oof_df = oof_df.rename(columns={"predictions": "generated"})
        oof_df = oof_df[["id", "generated"]].copy()

        to_return = {
            "scores": eval_dict,
            "result_df": result_df,
            "oof_df": oof_df,
        }
        return to_return

    return None


@hydra.main(version_base=None, config_path="../conf/r_detect", config_name="conf_r_detect")
def run_training(cfg):
    local_rank, rank, world_size = setup_distributed()

    try:
        if cfg.use_wandb and rank == 0:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        hardware_cfg = _resolve_hardware_cfg(cfg)

        if hardware_cfg.get("set_cudnn_benchmark"):
            torch.backends.cudnn.benchmark = True

        if hardware_cfg.get("override_train_batch_size"):
            cfg.train_params.per_device_train_batch_size = hardware_cfg["override_train_batch_size"]
        if hardware_cfg.get("override_eval_batch_size"):
            cfg.train_params.per_device_eval_batch_size = hardware_cfg["override_eval_batch_size"]
        if hardware_cfg.get("override_grad_accum_steps"):
            cfg.train_params.gradient_accumulation_steps = hardware_cfg["override_grad_accum_steps"]

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(f"Process rank: {rank}, device: {local_rank}, world size: {world_size}")

        def print_line():
            if rank == 0:
                prefix, unit, suffix = "#", "~~", "#"
                print(prefix + unit * 50 + suffix)

        if rank == 0:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        print_line()
        if rank == 0:
            print(f"setting seed: {cfg.seed}")
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        if rank == 0:
            os.makedirs(cfg.outputs.model_dir, exist_ok=True)
        print_line()

        print_line()
        data_dir = cfg.input_data_dir

        try:
            essay_df = pd.read_csv(os.path.join(data_dir, "train_essays.csv"))
        except Exception as e:
            essay_df = pd.read_parquet(os.path.join(data_dir, "train_essays.parquet"))

        essay_df = essay_df[~essay_df['text'].isna()].copy()
        essay_df = essay_df.reset_index(drop=True)

        rng = random.Random(cfg.seed)
        essay_df['fold'] = essay_df['text'].apply(lambda x: 'train' if rng.random() < 0.99 else 'valid')
        train_df = essay_df[essay_df['fold'] == 'train'].copy()
        valid_df = essay_df[essay_df['fold'] == 'valid'].copy()

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)

        if rank == 0:
            print(f"shape of train data: {train_df.shape}")
            print(f"{train_df.head()}")
            print(f"shape of validation data: {valid_df.shape}")

        dist.barrier()
        dataset_creator = AiDataset(cfg)

        train_ds = dataset_creator.get_dataset(train_df)
        valid_ds = dataset_creator.get_dataset(valid_df)
        dist.barrier()

        tokenizer = dataset_creator.tokenizer

        train_ds.set_format(
            type=None,
            columns=['id', 'input_ids', 'attention_mask', 'generated']
        )
        valid_ds = valid_ds.sort("input_length")
        valid_ds.set_format(
            type=None,
            columns=['id', 'input_ids', 'attention_mask', 'generated']
        )
        valid_ids = valid_df["id"].tolist()

        data_collator = AiCollator(tokenizer=tokenizer, pad_to_multiple_of=64)
        data_collator_train = AiCollatorTrain(tokenizer=tokenizer, pad_to_multiple_of=64, kwargs=dict(cfg=cfg))

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)

        train_dl_kwargs = dict(
            dataset=train_ds,
            batch_size=cfg.train_params.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator_train,
            pin_memory=hardware_cfg["pin_memory"],
        )
        if hardware_cfg["train_num_workers"] is not None:
            train_dl_kwargs["num_workers"] = hardware_cfg["train_num_workers"]
        if train_dl_kwargs.get("num_workers", 0) > 0:
            train_dl_kwargs["persistent_workers"] = hardware_cfg["persistent_workers"]
            if hardware_cfg.get("train_prefetch_factor") is not None:
                train_dl_kwargs["prefetch_factor"] = hardware_cfg["train_prefetch_factor"]

        valid_dl_kwargs = dict(
            dataset=valid_ds,
            batch_size=cfg.train_params.per_device_eval_batch_size,
            sampler=valid_sampler,
            collate_fn=data_collator,
            pin_memory=hardware_cfg["pin_memory"],
        )
        if hardware_cfg["eval_num_workers"] is not None:
            valid_dl_kwargs["num_workers"] = hardware_cfg["eval_num_workers"]
        if valid_dl_kwargs.get("num_workers", 0) > 0:
            valid_dl_kwargs["persistent_workers"] = hardware_cfg["persistent_workers"]
            if hardware_cfg.get("eval_prefetch_factor") is not None:
                valid_dl_kwargs["prefetch_factor"] = hardware_cfg["eval_prefetch_factor"]

        train_dl = DataLoader(**train_dl_kwargs)
        valid_dl = DataLoader(**valid_dl_kwargs)

        if rank == 0:
            print("data preparation done...")
        print_line()

        if rank == 0:
            print_line()
            for b in train_dl:
                break
            show_batch(b, tokenizer, task='training', print_fn=print)
            print_line()
            for b in valid_dl:
                break
            show_batch(b, tokenizer, task='training', print_fn=print)

        print_line()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        model_class = None
        if 'solar' in cfg.model.backbone_path.lower():
            model_class = LlamaForDetectAI
        elif 'phi' in cfg.model.backbone_path.lower():
            model_class = PhiForDetectAI
        else:
            model_class = MistralForDetectAI

        base_model = model_class.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,
            quantization_config=bnb_config,
            trust_remote_code=True if 'phi' in cfg.model.backbone_path.lower() else False,
        )

        base_model.config.pretraining_tp = 1
        if hardware_cfg["gradient_checkpointing"]:
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
            if hasattr(base_model, "gradient_checkpointing_enable"):
                base_model.gradient_checkpointing_enable()
            if hasattr(base_model, "enable_input_require_grads"):
                base_model.enable_input_require_grads()
            base_model.config.use_cache = False

        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.lora_alpha,
            lora_dropout=cfg.model.lora.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            target_modules=cfg_dict["model"]["lora"]["target_modules"],
            modules_to_save=cfg_dict["model"]["lora"]["modules_to_save"],
        )

        model = get_peft_model(base_model, peft_config)
        model.to(local_rank)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)

        if rank == 0:
            model.module.print_trainable_parameters()

        if hardware_cfg["compile_model"]:
            compile_backend = hardware_cfg.get("compile_backend") or "inductor"
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, backend=compile_backend)
                    if rank == 0:
                        print(f"torch.compile enabled with backend={compile_backend}")
                except Exception as compile_exc:
                    if rank == 0:
                        print(f"torch.compile failed with backend={compile_backend}: {compile_exc}")
            else:
                if rank == 0:
                    print("torch.compile requested but not available in this torch build.")

        dist.barrier()

        print_line()
        optimizer = get_optimizer(cfg, model, print_fn=print if rank == 0 else lambda x: None)

        print_line()
        num_epochs = cfg.train_params.num_train_epochs
        grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
        warmup_pct = cfg.train_params.warmup_pct

        num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
        num_training_steps = num_epochs * num_update_steps_per_epoch
        num_warmup_steps = int(warmup_pct * num_training_steps)

        if rank == 0:
            print(f"# training updates per epoch: {num_update_steps_per_epoch}")
            print(f"# training steps: {num_training_steps}")
            print(f"# warmup steps: {num_warmup_steps}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        best_lb = -1.
        save_trigger = cfg.train_params.save_trigger
        patience_tracker = 0
        current_iteration = 0

        start_time = time.time()
        dist.barrier()

        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            if rank == 0 and epoch != 0:
                progress_bar.close()

            progress_bar = tqdm(range(num_update_steps_per_epoch), disable=rank != 0)
            loss_meter = AverageMeter()

            model.train()
            for step, batch in enumerate(train_dl):
                batch = {k: v.to(local_rank, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch)
                loss = outputs.loss / grad_accumulation_steps
                loss.backward()

                if (step + 1) % grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    loss_meter.update(loss.item() * grad_accumulation_steps)

                    if rank == 0:
                        progress_bar.set_description(
                            f"STEP: {current_iteration + 1:5}/{num_training_steps:5}. "
                            f"LR: {get_lr(optimizer):.4f}. "
                            f"Loss: {loss_meter.avg:.4f}. "
                        )
                        progress_bar.update(1)
                    current_iteration += 1

                    if cfg.use_wandb and rank == 0:
                        wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                        wandb.log({"lr": get_lr(optimizer)}, step=current_iteration)

                    if current_iteration % cfg.train_params.eval_frequency == 0:
                        eval_response = run_evaluation(model, valid_dl, valid_ids, local_rank, rank, world_size)

                        if rank == 0:
                            scores_dict = eval_response["scores"]
                            result_df = eval_response["result_df"]
                            oof_df = eval_response["oof_df"]
                            lb = scores_dict["lb"]

                            print_line()
                            et = as_minutes(time.time() - start_time)
                            print(f">>> Epoch {epoch + 1} | Step {step} | Total Step {current_iteration} | Time: {et}")
                            print_line()
                            print(f">>> Current LB (AUC) = {round(lb, 4)}")
                            print_line()

                            is_best = False
                            if lb >= best_lb:
                                best_lb = lb
                                is_best = True
                                patience_tracker = 0
                                best_dict = {f"{k}_at_best": v for k, v in scores_dict.items()}
                            else:
                                patience_tracker += 1

                            if is_best:
                                oof_df.to_csv(os.path.join(cfg.outputs.model_dir, "oof_df_best.csv"), index=False)
                                result_df.to_csv(os.path.join(cfg.outputs.model_dir, "result_df_best.csv"), index=False)
                            else:
                                print(f">>> patience reached {patience_tracker}/{cfg.train_params.patience}")
                                print(f">>> current best score: {round(best_lb, 4)}")

                            oof_df.to_csv(os.path.join(cfg.outputs.model_dir, "oof_df_last.csv"), index=False)
                            result_df.to_csv(os.path.join(cfg.outputs.model_dir, "result_df_last.csv"), index=False)

                            dist.barrier()  # Wait for main process to finish writing files
                            unwrapped_model = model.module
                            unwrapped_model.save_pretrained(f"{cfg.outputs.model_dir}/last")
                            tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

                            if best_lb > save_trigger:
                                unwrapped_model.save_pretrained(f"{cfg.outputs.model_dir}/best")
                                tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/best")

                            if cfg.use_wandb:
                                wandb.log({"lb": lb, "best_lb": best_lb}, step=current_iteration)
                                for k, v in scores_dict.items():
                                    wandb.log({k: round(v, 4)}, step=current_iteration)
                                for k, v in best_dict.items():
                                    wandb.log({k: round(v, 4)}, step=current_iteration)

                        model.train()
                        torch.cuda.empty_cache()
                        print_line()

                        if patience_tracker >= cfg.train_params.patience:
                            if rank == 0:
                                print("stopping early")
                            return

        if cfg.use_wandb and rank == 0:
            wandb.finish()
    finally:
        cleanup()


if __name__ == "__main__":
    run_training()
