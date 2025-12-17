import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import List
from glob import glob
import random

# Rich & WandB
from rich.console import Console
from rich.traceback import install
import wandb

install()
console = Console()

sys.path.append(os.getcwd()) 

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed
from models.modeling_vae import AcousticVAE, AudioVAEConfig

# --- 原有的 MelDataset 和 data_collator 保持不变 ---
class MelDataset(Dataset):
    def __init__(self, data_dir, subsets=None, crop_size=256, is_eval=False):
        self.files = []
        self.is_eval = is_eval
        
        if subsets:
            subset_list = subsets.split(",")
            for subset in subset_list:
                subset_path = os.path.join(data_dir, subset.strip())
                subset_files = glob(os.path.join(subset_path, "*.pt"))
                if not subset_files:
                    subset_files = glob(os.path.join(subset_path, "**", "*.pt"), recursive=True)
                self.files.extend(subset_files)
                console.print(f"[green]Loaded {len(subset_files)} files from {subset}[/green]")
        else:
            self.files = glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}. Check your path and subsets.")
            
        self.crop_size = crop_size
        console.print(f"[bold blue]Total Dataset loaded: {len(self.files)} files. Eval Mode: {is_eval}[/bold blue]")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            mel = torch.load(self.files[idx], map_location="cpu")
            if not self.is_eval:
                if mel.shape[1] > self.crop_size:
                    start = torch.randint(0, mel.shape[1] - self.crop_size, (1,)).item()
                    mel = mel[:, start:start+self.crop_size]
                else:
                    pad_len = self.crop_size - mel.shape[1]
                    mel = torch.nn.functional.pad(mel, (0, pad_len))
            else:
                if mel.shape[1] > self.crop_size:
                    start = (mel.shape[1] - self.crop_size) // 2
                    mel = mel[:, start:start+self.crop_size]
                else:
                    pad_len = self.crop_size - mel.shape[1]
                    mel = torch.nn.functional.pad(mel, (0, pad_len))
            return {"input_mel": mel}
        except Exception as e:
            console.print(f"[red]Error loading {self.files[idx]}: {e}[/red]")
            return {"input_mel": torch.randn(80, self.crop_size)}

def data_collator(features):
    batch_mels = [f["input_mel"] for f in features]
    batch_mels = torch.stack(batch_mels)
    return {"mel": batch_mels, "labels": batch_mels}

class VAETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        rec_loss = outputs.get("rec_loss")
        kl_loss = outputs.get("kl_loss")
        
        target_kl_weight = self.model.config.kl_weight
        warmup_steps = 5000
        current_step = self.state.global_step
        
        if current_step < warmup_steps:
            current_weight = (current_step / warmup_steps) * target_kl_weight
        else:
            current_weight = target_kl_weight
            
        if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
            # Enhanced logging
            logs = {
                "train/rec_loss": outputs["rec_loss"].item(),
                "train/kl_loss": outputs["kl_loss"].item(),
                "train/ssim_loss": outputs["ssim_loss"].item(),
                "train/kl_weight": current_weight
            }
            self.log(logs)
            
        total_loss = rec_loss + current_weight * kl_loss
        return (total_loss, outputs) if return_outputs else total_loss

@hydra.main(version_base=None, config_path="../config", config_name="vae_config")
def main(cfg: DictConfig):
    # 将 Hydra Config 转换为 Dict，方便操作
    console.rule("[bold red]VAE Training Start[/bold red]")
    console.print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.training.seed)
    
    # 构建 TrainingArguments
    # 注意: OmegaConf 对象可以直接解包，但最好转为 dict
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))

    console.print(f"Initializing VAE with strides: {cfg.model.strides}, latent: {cfg.model.latent_channels}")

    config = AudioVAEConfig(
        hidden_channels=cfg.model.hidden_channels, 
        latent_channels=cfg.model.latent_channels,
        strides=list(cfg.model.strides),
        kl_weight=cfg.model.kl_weight,
        kl_clamp=cfg.model.kl_clamp,
        latent_dropout=cfg.model.latent_dropout,
        norm_num_groups=cfg.model.norm_num_groups
    )
    model = AcousticVAE(config)
    
    total_stride = 1
    for s in cfg.model.strides: total_stride *= s
    console.print(f"Total Compression Rate: {total_stride}x")
    
    console.print(f"Loading Training Data from: {cfg.data.train_subsets}")
    train_dataset = MelDataset(
        data_dir=cfg.data.data_dir, 
        subsets=cfg.data.train_subsets,
        crop_size=cfg.data.crop_size,
        is_eval=False
    )

    console.print(f"Loading Evaluation Data from: {cfg.data.eval_subsets}")
    eval_dataset = MelDataset(
        data_dir=cfg.data.data_dir, 
        subsets=cfg.data.eval_subsets,
        crop_size=cfg.data.crop_size,
        is_eval=True
    )
    
    trainer = VAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    if training_args.resume_from_checkpoint:
        console.print(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    console.print("[bold green]Training finished successfully![/bold green]")

if __name__ == "__main__":
    main()