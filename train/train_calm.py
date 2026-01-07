"""
Omni-Flow Audio-CALM Training Script.
Status: Fully aligned with DiT-based Architecture (TTS & ASR). 
Includes deterministic sorting fix for DDP.
"""

import os
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import set_peft_model_state_dict

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.traceback import install

# --- Safety Patch for Torch Load ---
_orig_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = safe_torch_load

sys.path.append(os.getcwd())
# Ensure models.modeling_calm exists and contains QwenCALM
from models.modeling_calm import QwenCALM, QwenCALMConfig

install(show_locals=False)
console = Console()
warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

def _get_rank_safe() -> int:
    try: return dist.get_rank()
    except: return 0

# ---------------------------------------------------------------------
# Dataset Definition (Fixed: Deterministic Sorting & Grouping)
# ---------------------------------------------------------------------
class CalmDataset(Dataset):
    def __init__(self, 
                 asr_latent_dir=None, asr_subsets=None,
                 tts_latent_dir=None, tts_subsets=None,
                 tokenizer=None, 
                 max_text_len=512, max_audio_len=1024, 
                 task_mode="mix", task_prob_tts=0.5,
                 max_samples=None):
        
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.task_mode = task_mode
        self.task_prob_tts = task_prob_tts

        # 存放统一的条目与长度，用于 group_by_length
        self.items = []
        self.lengths = []  # 必须与 items 等长且均 >=1

        def scan_data(root_dir, subsets, mode):
            data_list = []
            if not root_dir or not subsets:
                return data_list
            if _get_rank_safe() == 0:
                console.log(f"[green]Scanning {root_dir} for {subsets} ({mode})...[/green]")
            subset_list = subsets.split(",") if isinstance(subsets, str) else []
            files = []
            for subset in subset_list:
                pattern = os.path.join(root_dir, subset.strip(), "**", "*.trans.txt")
                files.extend(sorted(glob(pattern, recursive=True)))
            for trans_file in files:
                folder = os.path.dirname(trans_file)
                try:
                    with open(trans_file, "r", encoding="utf-8") as fh:
                        for line in fh:
                            parts = line.strip().split(" ", 1)
                            if len(parts) != 2: 
                                continue
                            fid, txt = parts
                            pt_path = os.path.join(folder, f"{fid}.pt")
                            if os.path.exists(pt_path):
                                data_list.append({
                                    "text": txt,
                                    "file_path": pt_path,
                                    "mode": mode,
                                })
                except:
                    continue
            return data_list
        asr_list = scan_data(asr_latent_dir, asr_subsets, "asr") if task_mode in ["asr", "mix"] else []
        tts_list = scan_data(tts_latent_dir, tts_subsets, "tts") if task_mode in ["tts", "mix"] else []

        # 组合策略：简单拼接，交由 Trainer 打乱；长度用于分桶
        if task_mode == "mix":
            self.items = asr_list + tts_list
        elif task_mode == "asr":
            self.items = asr_list
        else:
            self.items = tts_list

        # 计算长度（至少为1），防止 0 长度导致 LengthGroupedSampler 死循环
        for it in self.items:
            l = max(1, min(len(it["text"]), self.max_text_len))
            self.lengths.append(l)

        # 截断 max_samples
        if max_samples:
            self.items = self.items[:max_samples]
            self.lengths = self.lengths[:max_samples]

        if _get_rank_safe() == 0:
            console.print(f"  -> Final dataset size: {len(self.items)} (mode={task_mode})")

        # 准备模板
        self.asr_prompt_ids = self.tokenizer.encode(
            "<|im_start|>user\nTranscribe audio to text embedding.<|im_end|>\n<|im_start|>assistant\n", 
            add_special_tokens=False
        )
        self.tts_prompt_template = "<|im_start|>user\nRead this text:\n{}\n<|im_end|>\n<|im_start|>assistant\n"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            item = self.items[idx]
            current_mode = item["mode"]
            payload = torch.load(item["file_path"], map_location="cpu")
            audio = payload.get("latent", payload) if isinstance(payload, dict) else payload
            # Expect (T, D); 如果存储为 (D, T) 且 D=latent_dim，转置
            if audio.dim() == 2 and audio.shape[0] in (64, 80, 128, 192):  # latent_dim 或 mel_dim
                audio = audio.transpose(0, 1)
            if audio.shape[0] > self.max_audio_len:
                audio = audio[:self.max_audio_len]

            if current_mode == "tts":
                prompt_txt = self.tts_prompt_template.format(item["text"])
                input_ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
                labels = [-100] * len(input_ids)
            else:
                input_ids = self.asr_prompt_ids[:]
                target_txt = f"{item['text']}<|im_end|>"
                target_ids = self.tokenizer.encode(target_txt, add_special_tokens=False)
                if len(target_ids) > self.max_text_len:
                    target_ids = target_ids[:self.max_text_len]
                labels = target_ids

            if len(input_ids) > self.max_text_len:
                input_ids = input_ids[:self.max_text_len]
                if current_mode == "tts":
                    labels = labels[:self.max_text_len]

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "audio_features": audio,
                "task_mode": current_mode,
                "_valid": True
            }
        except Exception as e:
            return {"input_ids": torch.tensor([0], dtype=torch.long), "_valid": False}

# ---------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------
@dataclass
class CalmCollator:
    pad_token_id: int
    audio_pad_val: float = 0.0
    training: bool = False

    def _apply_spec_augment(self, audio_feat: torch.Tensor):
        # audio_feat: (D, T)
        D, T = audio_feat.shape
        if T > 20 and self.training:
            mask_len = random.randint(5, 10) 
            t0 = random.randint(0, T - mask_len)
            audio_feat[:, t0 : t0 + mask_len].fill_(0.0) 
        return audio_feat

    def __call__(self, features):
        valid = [f for f in features if f.get("_valid", False)]
        if not valid: return self._dummy_batch()

        proc_audio = []
        for f in valid:
            # features["audio_features"] is (T, D)
            feat = f["audio_features"].transpose(0, 1) 
            if self.training and f["task_mode"] == "asr":
                feat = self._apply_spec_augment(feat.clone())
            proc_audio.append(feat.transpose(0, 1)) 

        batch = {
            "text_input_ids": torch.nn.utils.rnn.pad_sequence(
                [f["input_ids"] for f in valid], batch_first=True, padding_value=self.pad_token_id
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [f["labels"] for f in valid], batch_first=True, padding_value=-100
            ),
            # Pad (T, D) -> Batch is (B, T, D)
            "audio_features": torch.nn.utils.rnn.pad_sequence(
                proc_audio, batch_first=True, padding_value=self.audio_pad_val
            ).transpose(1, 2), # Transpose to (B, D, T) (Channels First) for Model input requirements
            "audio_lens": torch.tensor([f.shape[0] for f in proc_audio], dtype=torch.long),
            "task_modes": [f["task_mode"] for f in valid]
        }
        
        batch["attention_mask"] = (batch["text_input_ids"] != self.pad_token_id).long()
        return batch

    def _dummy_batch(self):
        return {
            "text_input_ids": torch.tensor([[self.pad_token_id]], dtype=torch.long),
            "attention_mask": torch.tensor([[0]], dtype=torch.long),
            "labels": torch.tensor([[-100]], dtype=torch.long),
            "audio_features": torch.zeros(1, 64, 1), # (B, D, T)
            "audio_lens": torch.tensor([1], dtype=torch.long),
            "task_modes": ["tts"]
        }

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class CalmTrainer(Trainer):
    def __init__(self, lr_multipliers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_meters = {
            "tts": 0.0, "asr": 0.0, 
            "len": 0.0, "dur": 0.0,  # [NEW]
            "tts_cnt": 0, "asr_cnt": 0,
            "len_cnt": 0, "dur_cnt": 0  # [NEW]
        }
        self.lr_multipliers = lr_multipliers or {"soa": 1.0, "proj": 1.0, "head": 1.0}
        if _get_rank_safe() == 0:
            console.print("[magenta]🔧 LR Multipliers:[/magenta]", self.lr_multipliers)

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            
            input_proj_parameters = []
            flow_head_parameters = [] # Merged TTS and ASR heads
            soa_parameters = []
            
            model_to_opt = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
            if hasattr(model_to_opt, "module"): model_to_opt = model_to_opt.module

            for name, param in model_to_opt.named_parameters():
                if not param.requires_grad: continue
                
                # Grouping Logic
                if "soa_embed" in name:
                    soa_parameters.append(param)
                elif "input_proj" in name and "lora" not in name:
                    input_proj_parameters.append(param)
                elif "tts_flow_head" in name or "asr_flow_head" in name or "asr_cross_attn" in name:
                    # [OPTIMIZED] Include Cross-Attn in high LR group for faster alignment
                    flow_head_parameters.append(param)
                elif "bias" in name or "LayerNorm" in name:
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)
                    
            mult_soa = self.lr_multipliers.get("soa", 1.0)
            mult_proj = self.lr_multipliers.get("proj", 1.0)
            mult_head = self.lr_multipliers.get("head", 1.0)
            
            optimizer_grouped_parameters = [
                {"params": decay_parameters, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
                {"params": no_decay_parameters, "weight_decay": 0.0, "lr": self.args.learning_rate},
                {"params": input_proj_parameters, "weight_decay": self.args.weight_decay, "lr": mult_proj * self.args.learning_rate},
                {"params": flow_head_parameters, "weight_decay": self.args.weight_decay, "lr": mult_head * self.args.learning_rate},
                {"params": soa_parameters, "weight_decay": 0.0, "lr": mult_soa * self.args.learning_rate},
            ]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        # Logging accumulators
        if self.model.training:
            l_tts = outputs.get("loss_tts", torch.tensor(0., device=loss.device)).detach()
            l_asr = outputs.get("loss_asr", torch.tensor(0., device=loss.device)).detach()
            l_len = outputs.get("loss_len", torch.tensor(0., device=loss.device)).detach()
            l_dur = outputs.get("loss_dur", torch.tensor(0., device=loss.device)).detach()
            
            self.loss_meters["tts"] += l_tts.item()
            self.loss_meters["asr"] += l_asr.item()
            self.loss_meters["len"] += l_len.item()  # [NEW]
            self.loss_meters["dur"] += l_dur.item()  # [NEW]
            
            if l_tts > 0: self.loss_meters["tts_cnt"] += 1
            if l_asr > 0: self.loss_meters["asr_cnt"] += 1
            if l_len > 0: self.loss_meters["len_cnt"] += 1  # [NEW]
            if l_dur > 0: self.loss_meters["dur_cnt"] += 1  # [NEW]
            
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs):
        t_c = max(self.loss_meters["tts_cnt"], 1)
        a_c = max(self.loss_meters["asr_cnt"], 1)
        l_c = max(self.loss_meters["len_cnt"], 1)
        d_c = max(self.loss_meters["dur_cnt"], 1)
        
        logs["loss_tts"] = round(self.loss_meters["tts"] / t_c, 4)
        logs["loss_asr"] = round(self.loss_meters["asr"] / a_c, 4)
        logs["loss_len"] = round(self.loss_meters["len"] / l_c, 4)  # [NEW]
        logs["loss_dur"] = round(self.loss_meters["dur"] / d_c, 4)  # [NEW]
        
        # Reset
        self.loss_meters = {
            "tts": 0.0, "asr": 0.0, 
            "len": 0.0, "dur": 0.0,
            "tts_cnt": 0, "asr_cnt": 0,
            "len_cnt": 0, "dur_cnt": 0
        }
        super().log(logs, *args, **kwargs)
        
    def save_model(self, output_dir=None, _internal_call=False, **kwargs):
        if output_dir is None: output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        super().save_model(output_dir, _internal_call=_internal_call, **kwargs)
        
        if _get_rank_safe() == 0:
            model = self.model
            if hasattr(model, "module"): model = model.module 
            console.print(f"[magenta]💾 Saving Custom Components to {output_dir}...[/magenta]")
            try:
                torch.save(model.input_proj.state_dict(), os.path.join(output_dir, "input_proj.bin"))
                torch.save(model.tts_flow_head.state_dict(), os.path.join(output_dir, "tts_flow_head.bin"))
                torch.save(model.asr_flow_head.state_dict(), os.path.join(output_dir, "asr_flow_head.bin"))
                torch.save(model.tts_len_predictor.state_dict(), os.path.join(output_dir, "tts_len_predictor.bin"))
                torch.save(model.tts_dur_predictor.state_dict(), os.path.join(output_dir, "tts_dur_predictor.bin"))
                torch.save(model.asr_query_embed.state_dict(), os.path.join(output_dir, "asr_query_embed.bin"))
                torch.save(model.asr_cross_attn.state_dict(), os.path.join(output_dir, "asr_cross_attn.bin"))

                if hasattr(model, "soa_embed"):
                    data_to_save = model.soa_embed.data if isinstance(model.soa_embed, torch.nn.Parameter) else model.soa_embed
                    torch.save({"weight": data_to_save}, os.path.join(output_dir, "soa_embed.bin"))
            except Exception as e:
                console.print(f"[bold red]❌ Error saving custom components: {e}[/bold red]")

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None: eval_dataset = self.eval_dataset
        eval_collator = getattr(self, "eval_collator", None) or CalmCollator(self.tokenizer.pad_token_id, training=False)
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
    def _load(key, model_attr, name):
        # Safety check: if model doesn't have this attribute, skip
        if not hasattr(model, model_attr):
            return 

        path = cfg.model.get(key, None)
        if path and os.path.exists(path):
            console.print(f"[green]Loading {name} from: {path}[/green]")
            state_dict = torch.load(path, map_location="cpu")
            # Clean DDP prefixes if present
            clean_sd = {k.replace(f"{model_attr}.", "").replace("module.", ""): v for k, v in state_dict.items()}
            try:
                getattr(model, model_attr).load_state_dict(clean_sd, strict=False)
                console.print(f"[bold green]✅ {name} Loaded.[/bold green]")
            except Exception as e:
                console.print(f"[bold red]❌ {name} Fail: {e}[/bold red]")
        else:
            console.print(f"[yellow]⚠️ {name}: Random Init[/yellow]")
    
    _load("pretrained_projector_path", "input_proj", "Input Projector")
    _load("pretrained_tts_head_path", "tts_flow_head", "TTS Flow Head")
    _load("pretrained_asr_head_path", "asr_flow_head", "ASR Flow Head")
    _load("pretrained_tts_len_pred_path", "tts_len_predictor", "TTS Len Predictor")  # [NEW]
    _load("pretrained_asr_query_path", "asr_query_embed", "ASR Query Embed")         # [NEW]

# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../config", config_name="calm_config")
def main(cfg: DictConfig):
    task_mode = cfg.data.task_mode
    console.print(f"[bold]🔄 Task Mode:[/bold] {task_mode.upper()}")

    set_seed(cfg.training.seed)
    train_conf = OmegaConf.to_container(cfg.training, resolve=True)
    
    # Clean keys that are not valid TrainingArguments
    custom_keys = [
        "soa_lr_mult", 
        "proj_lr_mult",       
        "head_lr_mult", 
        "eval_max_samples", 
    ]
    for k in custom_keys:
        if k in train_conf: del train_conf[k]
    
    training_args = TrainingArguments(**train_conf)
    training_args.ddp_find_unused_parameters = cfg.training.get("ddp_find_unused_parameters", True)
    training_args.gradient_checkpointing = cfg.training.get("gradient_checkpointing", True) 
    training_args.ignore_data_skip = True
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token_id = tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Initialize Model
    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path,
        vae_path=cfg.model.vae_path,
        use_precomputed_latents=cfg.model.use_precomputed_latents,
        latent_dim=cfg.model.latent_dim,
        tts_loss_weight=cfg.model.tts_loss_weight,
        asr_loss_weight=cfg.model.get("asr_loss_weight", 1.0),
        downsample_rate=cfg.data.latent_downsample,
        max_audio_len=cfg.data.max_audio_len,
        tts_flow_hidden_dim=cfg.model.tts_flow_hidden_dim,
        tts_flow_num_layers=cfg.model.tts_flow_num_layers,
        asr_flow_hidden_dim=cfg.model.asr_flow_hidden_dim,
        asr_flow_num_layers=cfg.model.asr_flow_num_layers,
        max_text_len=cfg.data.max_text_len,                 # [NEW]
        len_pred_loss_weight=cfg.model.get("len_pred_loss_weight", 0.1),  # [NEW]
        dur_pred_loss_weight=cfg.model.get("dur_pred_loss_weight", 0.0),
        mel_mean=cfg.model.mel_mean,
        mel_std=cfg.model.mel_std,
        latent_mean=cfg.model.latent_mean,
        latent_std=cfg.model.latent_std,
    )
    model = QwenCALM(config)

    console.rule("[bold cyan]Component Loading[/bold cyan]")
    load_soft_restart_components(model, cfg, console)
    
    if cfg.model.use_lora:
        console.print("[blue]Initializing Unified LoRA...[/blue]")
        lora_config = LoraConfig(
            r=cfg.model.lora_rank, 
            lora_alpha=cfg.model.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.model.lora_dropout, 
            bias="none", 
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=[], 
        )
        model.llm = get_peft_model(model.llm, lora_config)
        
        lora_path = cfg.model.get("pretrained_lora_path", None)
        if lora_path and os.path.exists(lora_path):
            console.print(f"[yellow]Loading LoRA from {lora_path}...[/yellow]")
            try:
                if lora_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    sd = load_file(lora_path)
                else:
                    sd = torch.load(lora_path, map_location="cpu")
                set_peft_model_state_dict(model.llm, sd)
                console.print(f"[bold green]✅ LoRA loaded![/bold green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to load LoRA: {e}[/red]")
        else:
            console.print("[dim]ℹ️  LoRA initialized from scratch[/dim]")

    # Freeze/Unfreeze logic
    should_freeze_proj = cfg.model.get("freeze_projector", False)
    model.input_proj.requires_grad_(not should_freeze_proj)
    if should_freeze_proj:
        model.input_proj.eval()
        console.print("[bold yellow]🔒 Input Projector Frozen[/bold yellow]")
    
    model.tts_flow_head.requires_grad_(True)
    model.asr_flow_head.requires_grad_(True)

    if hasattr(model, "soa_embed"):
        model.soa_embed.requires_grad_(True)
        console.print("[bold green]🔓 SOA Embed Unfrozen[/bold green]")
    
    # === 根据任务模式，有选择地冻结另一侧头部 ===
    if task_mode == "tts":
        # TTS-only: 冻结所有 ASR 专用模块
        if hasattr(model, "asr_flow_head"):
            model.asr_flow_head.requires_grad_(False)
        if hasattr(model, "asr_cross_attn"):
            model.asr_cross_attn.requires_grad_(False)
        if hasattr(model, "asr_query_embed"):
            model.asr_query_embed.requires_grad_(False)
        console.print("[yellow]🧊 ASR-specific heads frozen during TTS-only training[/yellow]")
    elif task_mode == "asr":
        # ASR-only: 冻结所有 TTS 专用模块
        if hasattr(model, "tts_flow_head"):
            model.tts_flow_head.requires_grad_(False)
        if hasattr(model, "tts_len_predictor"):
            model.tts_len_predictor.requires_grad_(False)
        if hasattr(model, "tts_dur_predictor"):
            model.tts_dur_predictor.requires_grad_(False)
        console.print("[yellow]🧊 TTS-specific heads frozen during ASR-only training[/yellow]")

    # ---------------- Param Count + FLOPs (TTS & ASR) ----------------
    def _count_params(m): return sum(p.numel() for p in m.parameters())
    total_params = _count_params(model)
    backbone_params = _count_params(model.llm)
    non_backbone_params = total_params - backbone_params
    console.print(f"[cyan]Params (non-backbone): {non_backbone_params/1e6:.2f} M[/cyan]")
    console.print(f"[cyan]Params (total):        {total_params/1e6:.2f} M[/cyan]")

    try:
        from thop import profile
        model.eval()
        with torch.no_grad():
            # 仅在单卡且未初始化分布式时进行 FLOPs 估计，避免跨设备冲突
            if (not dist.is_available() or not dist.is_initialized()) and torch.cuda.device_count() <= 1:
                dev = next(model.parameters()).device
                dtype = next(model.llm.parameters()).dtype
                B = 1
                # TTS dummy
                T_txt, T_aud = 32, 64
                tts_text = torch.zeros(B, T_txt, dtype=torch.long, device=dev)
                tts_attn = torch.ones(B, T_txt, dtype=torch.long, device=dev)
                tts_labels = torch.full((B, T_txt), -100, dtype=torch.long, device=dev)
                tts_audio = torch.zeros(B, config.latent_dim, T_aud, device=dev, dtype=dtype)
                tts_lens = torch.tensor([T_aud], dtype=torch.long, device=dev)
                macs_tts, _ = profile(
                    model,
                    inputs=(tts_text, tts_audio, tts_attn, tts_labels, ["tts"], tts_lens),
                    verbose=False
                )
                console.print(f"[cyan]Approx FLOPs (TTS, {T_txt} txt, {T_aud} frames): {macs_tts/1e9:.2f} GFLOPs[/cyan]")
                # ASR dummy
                T_txt_asr, T_aud_asr = 40, 64
                asr_text = torch.zeros(B, 16, dtype=torch.long, device=dev)  # prompt ids placeholder
                asr_attn = torch.ones(B, 16, dtype=torch.long, device=dev)
                asr_labels = torch.randint(100, 200, (B, T_txt_asr), dtype=torch.long, device=dev)
                asr_audio = torch.zeros(B, config.latent_dim, T_aud_asr, device=dev, dtype=dtype)
                asr_lens = torch.tensor([T_aud_asr], dtype=torch.long, device=dev)
                macs_asr, _ = profile(
                    model,
                    inputs=(asr_text, asr_audio, asr_attn, asr_labels, ["asr"], asr_lens),
                    verbose=False
                )
                console.print(f"[cyan]Approx FLOPs (ASR, {T_txt_asr} tgt, {T_aud_asr} frames): {macs_asr/1e9:.2f} GFLOPs[/cyan]")
            else:
                console.print("[yellow]FLOPs estimation skipped (multi-GPU/DDP detected).[/yellow]")
    except Exception as e:
        console.print(f"[yellow]FLOPs estimation skipped: {e}[/yellow]")

    model.train()
    # ---------------------------------------------------------------

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    console.print(f"🔥 Trainable Modules: {len(trainable_params)} tensors")

    console.rule()

    # Data Setup
    asr_cfg = cfg.data.datasets.asr
    tts_cfg = cfg.data.datasets.tts
    
    train_ds = CalmDataset(
        asr_latent_dir=asr_cfg.latent_dir,
        asr_subsets=cfg.data.train_subsets,
        tts_latent_dir=tts_cfg.latent_dir,
        tts_subsets=cfg.data.train_subsets,
        tokenizer=tokenizer,
        max_text_len=cfg.data.max_text_len,
        max_audio_len=cfg.data.max_audio_len,
        task_mode=cfg.data.task_mode,
        task_prob_tts=cfg.data.task_prob_tts,
    )
    
    eval_max_samples = cfg.training.get("eval_max_samples", 1000)
    eval_ds = CalmDataset(
        asr_latent_dir=asr_cfg.eval_latent_dir,
        asr_subsets=cfg.data.eval_subsets,
        tts_latent_dir=tts_cfg.eval_latent_dir,
        tts_subsets=cfg.data.eval_subsets,
        tokenizer=tokenizer, 
        max_text_len=cfg.data.max_text_len, 
        max_audio_len=cfg.data.max_audio_len, 
        task_mode=cfg.data.task_mode, 
        task_prob_tts=cfg.data.task_prob_tts,
        max_samples=eval_max_samples
    )
    
    train_collator = CalmCollator(tokenizer.pad_token_id, training=True)
    eval_collator = CalmCollator(tokenizer.pad_token_id, training=False)

    lr_multipliers = {
        "soa": cfg.training.get("soa_lr_mult", 1.0),
        "proj": cfg.training.get("proj_lr_mult", 1.0),
        "head": cfg.training.get("head_lr_mult", 1.0),
    }

    trainer = CalmTrainer(
        model=model, 
        args=training_args,
        train_dataset=train_ds, 
        eval_dataset=eval_ds,
        data_collator=train_collator,
        lr_multipliers=lr_multipliers
    )
    trainer.eval_collator = eval_collator
    trainer.tokenizer = tokenizer

    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()