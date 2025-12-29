"""
Flow-based Audio-CALM Training Script.
Optimized for Speed, DDP Stability, and Mixture of Adapters (MoA).
VERSION: FINAL_STABLE (Includes SOA training & Explicit Saving)
"""

import os
# 设置环境变量，抑制 Transformers 的过时警告，保持日志清爽
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
import sys
import math
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any
from glob import glob

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import set_peft_model_state_dict

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.traceback import install

# --- Monkey Patch for PyTorch 2.6+ & DeepSpeed ---
# 【修复】：PyTorch 新版本中 torch.load 默认启用了 weights_only=True，
# 这会导致加载旧版 checkpoint 或由 DeepSpeed 保存的复杂对象时报错。
# 这里强制将其改回 weights_only=False 以兼容旧行为。
_orig_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = safe_torch_load
# -------------------------------------------------

# 将当前工作目录加入路径，以便导入 models 模块
sys.path.append(os.getcwd())
# 【对应关系】：导入 modeling_calm.py 中的模型定义
from models.modeling_calm import QwenCALM, QwenCALMConfig

install(show_locals=False)
console = Console()
warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

def _get_rank_safe() -> int:
    """安全获取当前进程的 Rank，用于多卡训练时的条件打印"""
    try: return dist.get_rank()
    except: return 0

# ---------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------
class CalmDataset(Dataset):
    """
    功能：CALM 模型的混合数据集加载器。
    
    【文件间关系】：
    - 输入：读取由 `preprocess/build_manifest.py` 生成的 .jsonl 清单或目录下的 .trans.txt 索引。
    - 依赖：读取由 `preprocess/process_dataset.py` 生成的 .pt (Latent) 文件。
    """
    def __init__(self, latent_dir, subsets, tokenizer, max_text_len=512, 
                 max_audio_len=1024, use_latents=False, task_mode="mix", task_prob_tts=0.5, 
                 max_samples=None):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.task_mode = task_mode
        self.task_prob_tts = task_prob_tts
        self.latent_dir = latent_dir
        
        # 确定 <|im_end|> Token 的 ID，用于 ASR 任务的 Label 截断
        if hasattr(tokenizer, "eod_id"):
             self.im_end_id = tokenizer.eod_id
        else:
             enc = tokenizer.encode("<|im_end|>", add_special_tokens=False)
             self.im_end_id = enc[-1] if len(enc)>0 else tokenizer.eos_token_id

        self.data = []
        if _get_rank_safe() == 0: 
            console.log(f"[green]Scanning Latent Directory: {latent_dir}[/green]")
            console.log(f"[dim]Subsets pattern: {subsets}[/dim]")

        # 1. 扫描转录文件 (.trans.txt)
        # 支持通过逗号分隔的子集列表（如 train-clean-100,train-other-500）
        trans_files = []
        for subset in subsets.split(","):
            subset = subset.strip()
            if subset == ".":
                pattern = os.path.join(latent_dir, "**", "*.trans.txt")
            else:
                pattern = os.path.join(latent_dir, subset, "**", "*.trans.txt")
            
            found = glob(pattern, recursive=True)
            trans_files.extend(found)

        # 2. 解析转录文件，构建内存中的数据索引
        for trans_file in trans_files:
            folder = os.path.dirname(trans_file)
            try:
                with open(trans_file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        # 格式: file_id transcript_text
                        parts = line.strip().split(" ", 1)
                        if len(parts) != 2: continue
                        
                        fid, txt = parts
                        # 假设 Latent 文件名为 {fid}.pt，与 preprocess 阶段一致
                        pt_path = os.path.join(folder, f"{fid}.pt")
                        
                        if os.path.exists(pt_path):
                            self.data.append({"text": txt, "file_path": pt_path})
            except Exception:
                continue
                                    
        if _get_rank_safe() == 0: 
            console.log(f"[bold green]Matched Pairs: {len(self.data)}[/bold green]")
            if len(self.data) == 0:
                console.log(f"[bold red]❌ CRITICAL: No data found in {latent_dir}.[/bold red]")
        
        # 3. 样本数量限制（用于快速调试）
        if max_samples is not None and max_samples > 0:
            if len(self.data) > max_samples:
                self.data = self.data[:max_samples]
                if _get_rank_safe() == 0:
                    console.log(f"[yellow]⚠️ Subsampled dataset to {len(self.data)} items.[/yellow]")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        """
        功能：获取单个样本，并根据任务模式构建 Input IDs 和 Labels。
        """
        try:
            item = self.data[idx]
            
            # 1. 动态决定当前样本的任务模式 (Mix Mode)
            if self.task_mode == "mix":
                # 按概率随机分配 TTS 或 ASR
                mode = "tts" if random.random() < self.task_prob_tts else "asr"
            else:
                mode = self.task_mode

            # 2. 加载音频 Latent
            # 【对应关系】：加载由 process_dataset.py 保存的 .pt 文件
            payload = torch.load(item["file_path"], map_location="cpu")
            # 兼容处理：支持直接存储 Tensor 或存储在 dict 中
            audio = payload.get("latent", payload) if isinstance(payload, dict) else payload
            if audio is None: return {"_valid": False}
            
            # 维度调整：确保是 [Time, Dim] 格式
            # VAE 输出通常是 [Dim=64, Time]，这里需要转置
            if audio.shape[0] == 64: audio = audio.transpose(0, 1)
            
            # 3. 音频长度裁剪逻辑
            if audio.shape[0] > self.max_audio_len:
                if mode == "asr": 
                    # ASR 任务：如果音频太长，这里简单跳过（实际生产中应切片）
                    return {"input_ids": [0], "_valid": False} 
                else:
                    # TTS 任务：[重要修复] 必须从头开始截取 (Start=0)
                    # 因为我们使用了 SOA (Start of Audio) Token，它隐含表示音频的开始。
                    # 如果随机截取中间一段，LLM 会因为上下文不匹配而无法收敛。
                    start = 0 
                    audio = audio[start : start + self.max_audio_len]

            # 4. 构建 Prompt 和 Text IDs
            prompt = f"Read this text:\n{item['text']}" if mode == "tts" else "Transcribe the following audio:"
            # 使用 ChatML 格式
            user_txt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            user_ids = self.tokenizer.encode(user_txt, add_special_tokens=False)
            
            # 5. 构建最终的 IDs 和 Labels
            if mode == "tts":
                # TTS 模式:
                # Input = [Prompt]
                # Label = [-100] (因为文本部分不需要 LLM 预测，LLM 只预测音频 Condition)
                text_ids = user_ids
                labels = [-100] * len(text_ids)
            else:
                # ASR 模式:
                # Input = [Prompt + Transcript]
                # Label = [-100 (Prompt) + Transcript (Target)]
                target_txt = f"{item['text']}<|im_end|>"
                target_ids = self.tokenizer.encode(target_txt, add_special_tokens=False)
                
                # 文本长度截断
                max_target_len = self.max_text_len - len(user_ids)
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]
                    if self.im_end_id is not None:
                        target_ids[-1] = self.im_end_id

                text_ids = user_ids + target_ids
                labels = [-100] * len(user_ids) + target_ids

            # 最终长度截断
            if len(text_ids) > self.max_text_len:
                text_ids = text_ids[:self.max_text_len]
                labels = labels[:self.max_text_len]

            # 返回数据字典
            return {
                "input_ids": torch.tensor(text_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "audio_features": audio, # [Time, Dim]
                "task_mode": mode,
                "_valid": True
            }
        except Exception as e:
            # 异常处理：返回无效标记，Collator 会过滤掉
            return {"input_ids": [0], "_valid": False}

# ---------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------
@dataclass
class CalmCollator:
    """
    功能：数据整理器。
    作用：将 Dataset 返回的样本列表堆叠成 Batch，并进行 Padding 和 特征增强。
    """
    pad_token_id: int
    audio_pad_val: float = 0.0
    training: bool = False

    def _apply_spec_augment(self, audio_feat: torch.Tensor):
        """
        功能：频谱增强 (SpecAugment)。
        作用：在训练 ASR 时，随机掩盖时间段，强迫模型利用上下文信息，防止过拟合。
        """
        D, T = audio_feat.shape
        num_masks = 1 if T < 150 else 2
        for _ in range(num_masks):
            if T > 20:
                mask_len = random.randint(5, 10) 
                t0 = random.randint(0, T - mask_len)
                # 使用当前特征的最小值进行填充（模拟静音/背景底噪）
                min_val = audio_feat.min()
                audio_feat[:, t0 : t0 + mask_len].fill_(min_val)
        return audio_feat

    def __call__(self, features):
        # 1. 过滤无效样本
        valid = [f for f in features if f.get("_valid", False)]
        if not valid:
            # 如果整个 Batch 都无效，返回一个假的最小 Batch 防止训练崩溃
            return {
                "text_input_ids": torch.tensor([[self.pad_token_id]], dtype=torch.long),
                "attention_mask": torch.tensor([[0]], dtype=torch.long),
                "labels": torch.tensor([[-100]], dtype=torch.long),
                "audio_features": torch.zeros(1, 1, 64),
                "audio_lens": torch.tensor([1], dtype=torch.long),
                "task_modes": ["tts"]
            }

        # 2. 处理音频特征
        proc_audio = []
        for f in valid:
            feat = f["audio_features"]
            feat = feat.transpose(0, 1) # 转置为 [Dim, Time] 以便进行 Mask 操作
            if self.training and f["task_mode"] == "asr":
                feat = self._apply_spec_augment(feat.clone())
            proc_audio.append(feat.transpose(0, 1)) # 转回 [Time, Dim]

        # 3. 组装 Batch
        batch = {
            # 文本 Padding (Right Padding)
            "text_input_ids": torch.nn.utils.rnn.pad_sequence(
                [f["input_ids"] for f in valid],
                batch_first=True, 
                padding_value=self.pad_token_id
            ),
            # Label Padding (-100 表示忽略计算 Loss)
            "labels": torch.nn.utils.rnn.pad_sequence(
                [f["labels"] for f in valid], 
                batch_first=True, 
                padding_value=-100
            ),
            # 音频 Padding
            "audio_features": torch.nn.utils.rnn.pad_sequence(
                proc_audio, 
                batch_first=True, 
                padding_value=self.audio_pad_val
            ).transpose(1, 2), # 最终输出 [Batch, Dim, Time] 适配 Conv1d
            "audio_lens": torch.tensor([f.shape[0] for f in proc_audio], dtype=torch.long),
            "task_modes": [f["task_mode"] for f in valid]
        }
        
        # 生成 Attention Mask
        batch["attention_mask"] = (batch["text_input_ids"] != self.pad_token_id).long()
        return batch

# ---------------------------------------------------------------------
# Trainer (Modified for Saving)
# ---------------------------------------------------------------------
class CalmTrainer(Trainer):
    """
    功能：自定义训练器。
    作用：
    1. 实现混合适配器 (MoA) 的动态切换逻辑。
    2. 实现参数分组优化（区分 Head 和 Base Model）。
    3. 自定义模型保存逻辑（保存非 LoRA 参数）。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用于记录分任务的 Loss
        self.loss_meters = {"tts": 0.0, "asr": 0.0, "tts_cnt": 0, "asr_cnt": 0}

    def create_optimizer(self):
        """
        功能：创建优化器。
        作用：将 Projector/Head/SOA Embed 分离出来，允许设置不同的学习率（如果需要）。
        """
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            projector_parameters = []
            
            # [关键] 标记哪些参数属于“头部组件”
            head_keywords = ["input_proj", "output_head", "soa_embed"]
            
            model_to_opt = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
            if hasattr(model_to_opt, "module"): model_to_opt = model_to_opt.module

            for name, param in model_to_opt.named_parameters():
                if not param.requires_grad: continue
                
                is_head = any(k in name for k in head_keywords) and "lora" not in name
                
                if is_head:
                    projector_parameters.append(param)
                else:
                    if "bias" in name or "LayerNorm" in name:
                        no_decay_parameters.append(param)
                    else:
                        decay_parameters.append(param)

            # 参数分组
            optimizer_grouped_parameters = [
                {"params": decay_parameters, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
                {"params": no_decay_parameters, "weight_decay": 0.0, "lr": self.args.learning_rate},
                # Head 部分可以设置更高的 LR (这里暂时设为 1.0 * base_lr)
                {"params": projector_parameters, "weight_decay": self.args.weight_decay, "lr": 1.0 * self.args.learning_rate}, 
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        功能：计算 Loss 步骤。
        作用：
        1. 根据 task_mode 切换 LoRA Adapter。
        2. 调用模型 forward 计算 Loss。
        3. 处理 DDP 模式下的 Ghost Gradients。
        """
        # 1. 切换 Adapter (MoA 核心逻辑)
        peft_model = model.module.llm if hasattr(model, "module") else model.llm
        task_modes = inputs.get("task_modes", ["tts"])
        # 简单策略：根据 Batch 中任务数量的多数派决定激活哪个 Adapter
        target_adapter = "tts" if task_modes.count("tts") >= task_modes.count("asr") else "asr"
        
        if hasattr(peft_model, "set_adapter") and target_adapter in peft_model.peft_config:
            peft_model.set_adapter(target_adapter)
        
        # 2. 前向传播
        # 【对应关系】：调用 modeling_calm.py 中的 QwenCALM.forward
        outputs = model(**inputs)
        loss = outputs["loss"]

        # 3. DDP 兼容处理
        # 在 DDP 模式下，如果 forward 中某些参数没有参与计算（例如 TTS Batch 中 ASR 的参数），
        # 反向传播会报错。这里加一个 dummy loss * 0.0 来欺骗 DDP。
        if self.model.training:
            raw_model = model.module if hasattr(model, "module") else model
            dummy_loss = 0.0
            for name, param in raw_model.named_parameters():
                if param.requires_grad and param.grad is None:
                    dummy_loss += param.sum() * 0.0
            loss += dummy_loss

        # 4. 记录日志
        if self.model.training:
             l_tts = outputs.get("loss_tts", torch.tensor(0., device=loss.device)).detach()
             l_asr = outputs.get("loss_asr", torch.tensor(0., device=loss.device)).detach()
             self.loss_meters["tts"] += l_tts.item()
             self.loss_meters["asr"] += l_asr.item()
             if l_tts > 0: self.loss_meters["tts_cnt"] += 1
             if l_asr > 0: self.loss_meters["asr_cnt"] += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs):
        """重写日志记录，加入 TTS/ASR 分项 Loss"""
        t_c = max(self.loss_meters["tts_cnt"], 1)
        a_c = max(self.loss_meters["asr_cnt"], 1)
        logs["loss_tts"] = round(self.loss_meters["tts"] / t_c, 4)
        logs["loss_asr"] = round(self.loss_meters["asr"] / a_c, 4)
        # 重置计数器
        self.loss_meters = {"tts": 0.0, "asr": 0.0, "tts_cnt": 0, "asr_cnt": 0}
        super().log(logs, *args, **kwargs)
        
    # [关键修复] 自定义保存逻辑
    # 修复了参数签名以兼容新版 HF Trainer，并增加了手动保存逻辑
    def save_model(self, output_dir=None, _internal_call=False, **kwargs):
        """
        功能：保存模型 Checkpoint。
        作用：
        1. 调用父类保存 LoRA Adapter。
        2. 手动保存 Input Projector, Output Head, 和 SOA Embed 为 .bin 文件。
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存 LoRA (调用父类)
        super().save_model(output_dir, _internal_call=_internal_call, **kwargs)
        
        # 2. 手动保存非 LoRA 组件 (仅主进程执行)
        if _get_rank_safe() == 0:
            model = self.model
            if hasattr(model, "module"): 
                model = model.module 
            
            console.print(f"[magenta]💾 Saving Projectors & SOA to {output_dir}...[/magenta]")
            
            try:
                # 保存 Input Projector (ASR 用)
                torch.save(model.input_proj.state_dict(), os.path.join(output_dir, "input_proj.bin"))
                
                # 保存 Output Head (TTS 用)
                torch.save(model.output_head.state_dict(), os.path.join(output_dir, "output_head.bin"))
                
                # 保存 SOA Embed (TTS 用)
                if hasattr(model, "soa_embed"):
                    data_to_save = model.soa_embed.data if isinstance(model.soa_embed, torch.nn.Parameter) else model.soa_embed
                    torch.save({"weight": data_to_save}, os.path.join(output_dir, "soa_embed.bin"))
            except Exception as e:
                console.print(f"[bold red]❌ Error saving custom components: {e}[/bold red]")
            
    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        eval_collator = getattr(self, "eval_collator", None)
        if eval_collator is None:
            eval_collator = CalmCollator(
                pad_token_id=self.tokenizer.pad_token_id, 
                training=False
            )
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=eval_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_soft_restart_components(model, cfg, console):
    """
    功能：软重启/热启动加载。
    作用：从指定的 checkpoint 路径加载 Projector 或 Head 的权重，用于分阶段训练。
    """
    def _load(key, model_attr, name):
        path = cfg.model.get(key, None)
        if path and os.path.exists(path):
            console.print(f"[green]Loading {name} from: {path}[/green]")
            state_dict = torch.load(path, map_location="cpu")
            # 清理 key 名称
            clean_sd = {k.replace(f"{name}.", "").replace(f"input_proj.", "").replace(f"output_head.", ""): v for k, v in state_dict.items()}
            try:
                getattr(model, model_attr).load_state_dict(clean_sd, strict=False)
                console.print(f"[bold green]✅ {name} Loaded.[/bold green]")
            except Exception as e:
                console.print(f"[bold red]❌ {name} Fail: {e}[/bold red]")
        else:
            console.print(f"[yellow]⚠️ {name}: Random Init (Path not found)[/yellow]")

    _load("pretrained_projector_path", "input_proj", "input_proj")
    _load("pretrained_head_path", "output_head", "output_head")

# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../config", config_name="calm_config")
def main(cfg: DictConfig):
    task_mode = cfg.data.task_mode
    console.print(f"[bold]🔄 Task Mode:[/bold] {task_mode.upper()}")

    if task_mode not in cfg.data.datasets:
        raise ValueError(f"❌ Unknown task_mode: '{task_mode}'. Available: {list(cfg.data.datasets.keys())}")

    # 路径解析
    selected_paths = cfg.data.datasets[task_mode]
    with open_dict(cfg):
        cfg.data.latent_dir = selected_paths.latent_dir
        cfg.data.eval_latent_dir = selected_paths.eval_latent_dir
        cfg.data.raw_root = selected_paths.raw_root

    console.print(f"📂 [Data] Training Latents: {cfg.data.latent_dir}")
    
    set_seed(cfg.training.seed)
    
    # 转换参数
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))
    training_args.ddp_find_unused_parameters = True 
    training_args.ignore_data_skip = True
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token_id = tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.eos_token_id
    
    tokenizer.padding_side = "right" # 对齐模型 Right Padding 逻辑

    # 1. 模型初始化
    # 【对应关系】：实例化 modeling_calm.py 中的 QwenCALM
    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path,
        vae_path=cfg.model.vae_path,
        head_type="flow", 
        use_precomputed_latents=cfg.model.use_precomputed_latents,
        latent_dim=cfg.model.latent_dim,
        audio_loss_weight=cfg.model.audio_loss_weight,
        downsample_rate=cfg.data.latent_downsample,
        flow_hidden_dim=cfg.model.flow_hidden_dim,
        flow_num_layers=cfg.model.flow_num_layers,
    )
    model = QwenCALM(config)

    # 2. 组件加载 (Soft Restart)
    console.rule("[bold cyan]Component Loading[/bold cyan]")
    load_soft_restart_components(model, cfg, console)
    
    # 3. LoRA / MoA 初始化
    if cfg.model.use_lora:
        console.print("[blue]Initializing LoRA Config...[/blue]")
        lora_config = LoraConfig(
            r=cfg.model.lora_rank, lora_alpha=cfg.model.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.model.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM,
            modules_to_save=[], # [FIX] 我们在 CalmTrainer 中手动保存，这里留空以避免重复
        )
        
        def load_adapter_if_path_exists(adapter_name, path_key):
            path = cfg.model.get(path_key, None)
            if path and os.path.exists(path):
                console.print(f"[yellow]Loading {adapter_name} from {path}...[/yellow]")
                try:
                    if path.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        sd = load_file(path)
                    else:
                        sd = torch.load(path, map_location="cpu")
                    set_peft_model_state_dict(model.llm, sd, adapter_name=adapter_name)
                    console.print(f"[bold green]✅ {adapter_name} loaded![/bold green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to load {adapter_name}: {e}[/red]")
            else:
                console.print(f"[dim]ℹ️  {adapter_name} initialized from scratch[/dim]")
        
        # 根据任务模式配置 Adapter
        if cfg.data.task_mode == "tts":
            console.print("[green] -> Mode: TTS Only[/green]")
            model.llm = get_peft_model(model.llm, lora_config, adapter_name="tts")
            load_adapter_if_path_exists("tts", "pretrained_lora_path_tts")
            
        elif cfg.data.task_mode == "asr":
            console.print("[green] -> Mode: ASR Only[/green]")
            model.llm = get_peft_model(model.llm, lora_config, adapter_name="asr")
            load_adapter_if_path_exists("asr", "pretrained_lora_path_asr")
            
        else:
            console.print("[green] -> Mode: Mix (MoA)[/green]")
            # 混合模式：同时注入两个 Adapter
            model.llm = get_peft_model(model.llm, lora_config, adapter_name="tts")
            model.llm.add_adapter("asr", lora_config)
            load_adapter_if_path_exists("tts", "pretrained_lora_path_tts")
            load_adapter_if_path_exists("asr", "pretrained_lora_path_asr")

    # 4. 冻结策略
    # 根据配置决定是否冻结 Input Projector (保护 ASR 能力)
    should_freeze_proj = cfg.model.get("freeze_projector", False)
    
    # Projector
    model.input_proj.requires_grad_(not should_freeze_proj)
    if should_freeze_proj:
        model.input_proj.eval()
        console.print("[bold yellow]🔒 Input Projector Frozen (Protecting ASR capabilities)[/bold yellow]")
    
    # Head 始终训练
    model.output_head.requires_grad_(True)
    
    # [FIX] 显式解冻 SOA Embed (TTS 任务必须)
    if hasattr(model, "soa_embed"):
        model.soa_embed.requires_grad_(True)
        console.print("[bold green]🔓 SOA Embed Unfrozen (Ready for TTS training)[/bold green]")

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    console.print(f"🔥 Trainable Modules: {[n for n in trainable_params if 'bias' not in n][:10]} ...")

    console.rule()

    # 5. 构建 Trainer
    # 初始化训练集
    train_ds = CalmDataset(
        latent_dir=cfg.data.latent_dir, 
        subsets=cfg.data.train_subsets, 
        tokenizer=tokenizer, 
        max_text_len=cfg.data.max_text_len, 
        max_audio_len=cfg.data.max_audio_len, 
        use_latents=cfg.model.use_precomputed_latents, 
        task_mode=cfg.data.task_mode, 
        task_prob_tts=cfg.data.task_prob_tts,
        max_samples=None 
    )
    
    # 初始化验证集
    eval_max_samples = cfg.training.get("eval_max_samples", 200)
    eval_ds = CalmDataset(
        latent_dir=cfg.data.eval_latent_dir or cfg.data.latent_dir, 
        subsets=cfg.data.eval_subsets, 
        tokenizer=tokenizer, 
        max_text_len=cfg.data.max_text_len, 
        max_audio_len=cfg.data.max_audio_len, 
        use_latents=cfg.model.use_precomputed_latents, 
        task_mode=cfg.data.task_mode, 
        task_prob_tts=cfg.data.task_prob_tts,
        max_samples=eval_max_samples
    )
    
    # 初始化 Collator
    train_collator = CalmCollator(tokenizer.pad_token_id, training=True)
    eval_collator = CalmCollator(tokenizer.pad_token_id, training=False)

    trainer = CalmTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=train_collator
    )
    
    trainer.eval_collator = eval_collator
    trainer.tokenizer = tokenizer

    # 6. 开始训练
    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # 7. 最终保存
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()