# train/train_vae.py
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import List
from glob import glob
import random

# 引入 Rich 终端美化 和 WandB 日志
from rich.console import Console
from rich.traceback import install
import wandb

install()
console = Console()

sys.path.append(os.getcwd()) 

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed
# 【对应关系】：导入 VAE 模型定义
from models.modeling_vae import AcousticVAE, AudioVAEConfig

class MelDataset(Dataset):
    """
    功能：加载 Mel 频谱数据集。
    注意：这里加载的是预处理好的 .pt 文件。
    
    【文件间关系】：
    - 读取的数据：假设是 `process_dataset.py` 生成的 .pt 文件。
    - 注意点：如果 .pt 存的是字典 {'latent':..., 'mel':...}，这里直接 load 可能需要修改。
      目前的逻辑是假设 torch.load 直接返回 Mel Tensor。
    """
    def __init__(self, data_dir, subsets=None, crop_size=256, is_eval=False):
        self.files = []
        self.is_eval = is_eval
        
        # 1. 扫描文件
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
            # 2. 加载 .pt 文件
            payload = torch.load(self.files[idx], map_location="cpu", weights_only=False)
            
            # [关键修复]：处理字典格式，提取 "mel"
            if isinstance(payload, dict):
                if "mel" in payload:
                    mel = payload["mel"]
                elif "input_mel" in payload: # 兼容性
                    mel = payload["input_mel"]
                else:
                    # 如果只有 latent 没有 mel，报错
                    raise ValueError(f"File {self.files[idx]} is a dict but misses 'mel' key.")
            else:
                mel = payload # 假设直接存的 Tensor

            # 确保是 Float 类型
            mel = mel.float()
            
            # 3. 随机裁剪 (Random Crop)
            # VAE 训练需要固定长度的片段
            if not self.is_eval:
                # 训练模式：随机切一段
                if mel.shape[1] > self.crop_size:
                    start = torch.randint(0, mel.shape[1] - self.crop_size, (1,)).item()
                    mel = mel[:, start:start+self.crop_size]
                else:
                    # 长度不够则填充
                    pad_len = self.crop_size - mel.shape[1]
                    mel = torch.nn.functional.pad(mel, (0, pad_len))
            else:
                # 评估模式：中心裁剪 (Center Crop)
                if mel.shape[1] > self.crop_size:
                    start = (mel.shape[1] - self.crop_size) // 2
                    mel = mel[:, start:start+self.crop_size]
                else:
                    pad_len = self.crop_size - mel.shape[1]
                    mel = torch.nn.functional.pad(mel, (0, pad_len))
            return {"input_mel": mel}
        
        except Exception as e:
            console.print(f"[red]Error loading {self.files[idx]}: {e}[/red]")
            # 出错返回随机噪声防止崩溃
            return {"input_mel": torch.randn(80, self.crop_size)}

def data_collator(features):
    """
    功能：将 Dataset 返回的列表堆叠成 Batch Tensor。
    """
    batch_mels = [f["input_mel"] for f in features]
    batch_mels = torch.stack(batch_mels)
    # VAE 是自监督任务，输入(mel) 既是 Input 也是 Labels (target)
    return {"mel": batch_mels, "labels": batch_mels}

class VAETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        if "loss" in outputs:
            total = outputs["loss"]  # 直接用模型 forward 的总损失（含 ssim/stft/kl）
        else:
            rec_loss = outputs["rec_loss"]
            kl_loss = outputs["kl_loss"]
            stft_loss = outputs.get("stft_loss", torch.tensor(0.0, device=rec_loss.device))
            ssim_loss = outputs.get("ssim_loss", torch.tensor(0.0, device=rec_loss.device))
            target_kl_weight = self.model.config.kl_weight
            warmup_steps = 5000
            current_step = self.state.global_step
            current_weight = target_kl_weight * min(1.0, current_step / warmup_steps)
            total = rec_loss + current_weight * kl_loss + 0.5 * stft_loss + 0.5 * ssim_loss

        if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
            logs = {
                "train/rec_loss": outputs["rec_loss"].item(),
                "train/kl_loss": outputs["kl_loss"].item(),
                "train/stft_loss": outputs.get("stft_loss", torch.tensor(0)).item(),
                "train/ssim_loss": outputs.get("ssim_loss", torch.tensor(0)).item(),
                "train/kl_weight": self.model.config.kl_weight * min(1.0, self.state.global_step / 5000),
                "train/kl_mean": outputs.get("kl_mean", torch.tensor(0)).item(),
                "train/mu_mean": outputs.get("mu_mean", torch.tensor(0)).item(),
                "train/mu_std": outputs.get("mu_std", torch.tensor(0)).item(),
                "train/var_mean": outputs.get("var_mean", torch.tensor(0)).item(),
                "train/kl_per_dim_max": outputs.get("kl_per_dim_max", torch.tensor(0)).item(),
            }
            self.log(logs)
        return (total, outputs) if return_outputs else total

@hydra.main(version_base=None, config_path="../config", config_name="vae_config")
def main(cfg: DictConfig):
    # 1. 打印配置
    console.rule("[bold red]VAE Training Start[/bold red]")
    console.print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.training.seed)
    
    # 2. 转换 Hydra 配置为 HuggingFace TrainingArguments
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))

    console.print(f"Initializing VAE with strides: {cfg.model.strides}, latent: {cfg.model.latent_channels}")

    # 3. 初始化模型配置
    # 【对应关系】：映射 yaml 中的 model 部分到 AudioVAEConfig
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
    
    # 4. 加载数据集
    # 【对应关系】：使用 yaml 中的 data 部分路径
    console.print(f"Loading Training Data from: {cfg.data.train_subsets}")
    train_dataset = MelDataset(
        data_dir=cfg.data.data_dir, 
        subsets=cfg.data.train_subsets,
        crop_size=cfg.data.crop_size,
        is_eval=False
    )

    console.print(f"Loading Evaluation Data from: {cfg.data.eval_subsets}")
    eval_dataset = MelDataset(
        data_dir=cfg.data.eval_data_dir, 
        subsets=cfg.data.eval_subsets,
        crop_size=cfg.data.crop_size,
        is_eval=True
    )
    
    # 5. 初始化 Trainer
    trainer = VAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    # 6. 开始训练
    if training_args.resume_from_checkpoint:
        console.print(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
        
    # 7. 保存最终模型
    trainer.save_model(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    console.print("[bold green]Training finished successfully![/bold green]")

if __name__ == "__main__":
    main()