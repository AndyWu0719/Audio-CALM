"""
Flow-based Audio-CALM Training Script.
Optimized for Speed, DDP Stability, and Mixture of Adapters (MoA).
VERSION: FINAL_STABLE (Includes SOA training & Explicit Saving)
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
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import set_peft_model_state_dict

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.traceback import install

# --- Monkey Patch for PyTorch 2.6+ & DeepSpeed ---
_orig_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = safe_torch_load
# -------------------------------------------------

sys.path.append(os.getcwd())
from models.modeling_calm import QwenCALM, QwenCALMConfig

install(show_locals=False)
console = Console()
warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

def _get_rank_safe() -> int:
    try: return dist.get_rank()
    except: return 0

# ---------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------
class CalmDataset(Dataset):
    def __init__(self, latent_dir, subsets, tokenizer, max_text_len=512, 
                 max_audio_len=1024, use_latents=False, task_mode="mix", task_prob_tts=0.5, 
                 max_samples=None):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.task_mode = task_mode
        self.task_prob_tts = task_prob_tts
        self.latent_dir = latent_dir
        
        if hasattr(tokenizer, "eod_id"):
             self.im_end_id = tokenizer.eod_id
        else:
             enc = tokenizer.encode("<|im_end|>", add_special_tokens=False)
             self.im_end_id = enc[-1] if len(enc)>0 else tokenizer.eos_token_id

        self.data = []
        if _get_rank_safe() == 0: 
            console.log(f"[green]Scanning Latent Directory: {latent_dir}[/green]")
            console.log(f"[dim]Subsets pattern: {subsets}[/dim]")

        trans_files = []
        for subset in subsets.split(","):
            subset = subset.strip()
            if subset == ".":
                pattern = os.path.join(latent_dir, "**", "*.trans.txt")
            else:
                pattern = os.path.join(latent_dir, subset, "**", "*.trans.txt")
            
            found = glob(pattern, recursive=True)
            trans_files.extend(found)

        for trans_file in trans_files:
            folder = os.path.dirname(trans_file)
            try:
                with open(trans_file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split(" ", 1)
                        if len(parts) != 2: continue
                        
                        fid, txt = parts
                        pt_path = os.path.join(folder, f"{fid}.pt")
                        
                        if os.path.exists(pt_path):
                            self.data.append({"text": txt, "file_path": pt_path})
            except Exception:
                continue
                                    
        if _get_rank_safe() == 0: 
            console.log(f"[bold green]Matched Pairs: {len(self.data)}[/bold green]")
            if len(self.data) == 0:
                console.log(f"[bold red]❌ CRITICAL: No data found in {latent_dir}.[/bold red]")
        
        if max_samples is not None and max_samples > 0:
            if len(self.data) > max_samples:
                self.data = self.data[:max_samples]
                if _get_rank_safe() == 0:
                    console.log(f"[yellow]⚠️ Subsampled dataset to {len(self.data)} items.[/yellow]")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            if self.task_mode == "mix":
                mode = "tts" if random.random() < self.task_prob_tts else "asr"
            else:
                mode = self.task_mode

            payload = torch.load(item["file_path"], map_location="cpu")
            audio = payload.get("latent", payload) if isinstance(payload, dict) else payload
            if audio is None: return {"_valid": False}
            
            if audio.shape[0] == 64: audio = audio.transpose(0, 1)
            
            if audio.shape[0] > self.max_audio_len:
                if mode == "asr": 
                    # ASR: Usually keep full audio or truncate end
                    return {"input_ids": [0], "_valid": False} # Simple skip for now
                else:
                    # TTS: Random crop is acceptable
                    start = random.randint(0, audio.shape[0] - self.max_audio_len)
                    audio = audio[start : start + self.max_audio_len]

            prompt = f"Read this text:\n{item['text']}" if mode == "tts" else "Transcribe the following audio:"
            user_txt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            user_ids = self.tokenizer.encode(user_txt, add_special_tokens=False)
            
            if mode == "tts":
                text_ids = user_ids
                labels = [-100] * len(text_ids)
            else:
                target_txt = f"{item['text']}<|im_end|>"
                target_ids = self.tokenizer.encode(target_txt, add_special_tokens=False)
                
                max_target_len = self.max_text_len - len(user_ids)
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]
                    if self.im_end_id is not None:
                        target_ids[-1] = self.im_end_id

                text_ids = user_ids + target_ids
                labels = [-100] * len(user_ids) + target_ids

            if len(text_ids) > self.max_text_len:
                text_ids = text_ids[:self.max_text_len]
                labels = labels[:self.max_text_len]

            return {
                "input_ids": torch.tensor(text_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "audio_features": audio,
                "task_mode": mode,
                "_valid": True
            }
        except Exception as e:
            return {"input_ids": [0], "_valid": False}

# ---------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------
@dataclass
class CalmCollator:
    pad_token_id: int
    audio_pad_val: float = 0.0
    training: bool = False

    def _apply_spec_augment(self, audio_feat: torch.Tensor):
        D, T = audio_feat.shape
        num_masks = 1 if T < 150 else 2
        for _ in range(num_masks):
            if T > 20:
                mask_len = random.randint(5, 10) 
                t0 = random.randint(0, T - mask_len)
                min_val = audio_feat.min() # 通常约为 -11.5
                audio_feat[:, t0 : t0 + mask_len].fill_(min_val)
        return audio_feat

    def __call__(self, features):
        valid = [f for f in features if f.get("_valid", False)]
        if not valid:
            # Fallback for empty batch
            return {
                "text_input_ids": torch.tensor([[self.pad_token_id]], dtype=torch.long),
                "attention_mask": torch.tensor([[0]], dtype=torch.long),
                "labels": torch.tensor([[-100]], dtype=torch.long),
                "audio_features": torch.zeros(1, 1, 64),
                "audio_lens": torch.tensor([1], dtype=torch.long),
                "task_modes": ["tts"]
            }

        proc_audio = []
        for f in valid:
            feat = f["audio_features"]
            feat = feat.transpose(0, 1) 
            if self.training and f["task_mode"] == "asr":
                feat = self._apply_spec_augment(feat.clone())
            proc_audio.append(feat.transpose(0, 1))

        batch = {
            "text_input_ids": torch.nn.utils.rnn.pad_sequence(
                [f["input_ids"] for f in valid],
                batch_first=True, 
                padding_value=self.pad_token_id
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [f["labels"] for f in valid], 
                batch_first=True, 
                padding_value=-100
            ),
            "audio_features": torch.nn.utils.rnn.pad_sequence(
                proc_audio, 
                batch_first=True, 
                padding_value=self.audio_pad_val
            ).transpose(1, 2),
            "audio_lens": torch.tensor([f.shape[0] for f in proc_audio], dtype=torch.long),
            "task_modes": [f["task_mode"] for f in valid]
        }
        
        batch["attention_mask"] = (batch["text_input_ids"] != self.pad_token_id).long()
        return batch

# ---------------------------------------------------------------------
# Trainer (Modified for Saving)
# ---------------------------------------------------------------------
class CalmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_meters = {"tts": 0.0, "asr": 0.0, "tts_cnt": 0, "asr_cnt": 0}

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            projector_parameters = []
            
            # [FIX] Added "soa_embed" to head_keywords
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

            optimizer_grouped_parameters = [
                {"params": decay_parameters, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
                {"params": no_decay_parameters, "weight_decay": 0.0, "lr": self.args.learning_rate},
                {"params": projector_parameters, "weight_decay": self.args.weight_decay, "lr": 1.0 * self.args.learning_rate}, 
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Handle Peft Adapter Switching
        peft_model = model.module.llm if hasattr(model, "module") else model.llm
        task_modes = inputs.get("task_modes", ["tts"])
        # Majority vote for batch adapter
        target_adapter = "tts" if task_modes.count("tts") >= task_modes.count("asr") else "asr"
        
        if hasattr(peft_model, "set_adapter") and target_adapter in peft_model.peft_config:
            peft_model.set_adapter(target_adapter)
        
        outputs = model(**inputs)
        loss = outputs["loss"]

        # DDP Dummy Loss for unused parameters
        if self.model.training:
            raw_model = model.module if hasattr(model, "module") else model
            dummy_loss = 0.0
            for name, param in raw_model.named_parameters():
                if param.requires_grad and param.grad is None:
                    dummy_loss += param.sum() * 0.0
            loss += dummy_loss

        # Logging
        if self.model.training:
             l_tts = outputs.get("loss_tts", torch.tensor(0., device=loss.device)).detach()
             l_asr = outputs.get("loss_asr", torch.tensor(0., device=loss.device)).detach()
             self.loss_meters["tts"] += l_tts.item()
             self.loss_meters["asr"] += l_asr.item()
             if l_tts > 0: self.loss_meters["tts_cnt"] += 1
             if l_asr > 0: self.loss_meters["asr_cnt"] += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs):
        t_c = max(self.loss_meters["tts_cnt"], 1)
        a_c = max(self.loss_meters["asr_cnt"], 1)
        logs["loss_tts"] = round(self.loss_meters["tts"] / t_c, 4)
        logs["loss_asr"] = round(self.loss_meters["asr"] / a_c, 4)
        self.loss_meters = {"tts": 0.0, "asr": 0.0, "tts_cnt": 0, "asr_cnt": 0}
        super().log(logs, *args, **kwargs)
        
    # [FIX] Custom Save Model (Fixed Argument Signature & DDP Safety)
    def save_model(self, output_dir=None, _internal_call=False, **kwargs):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        super().save_model(output_dir, _internal_call=_internal_call, **kwargs)
        if _get_rank_safe() == 0:
            model = self.model
            if hasattr(model, "module"): 
                model = model.module 
            
            console.print(f"[magenta]💾 Saving Projectors & SOA to {output_dir}...[/magenta]")
            
            try:
                # Save Input Projector
                torch.save(model.input_proj.state_dict(), os.path.join(output_dir, "input_proj.bin"))
                
                # Save Output Head
                torch.save(model.output_head.state_dict(), os.path.join(output_dir, "output_head.bin"))
                
                # Save SOA Embed
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
    def _load(key, model_attr, name):
        path = cfg.model.get(key, None)
        if path and os.path.exists(path):
            console.print(f"[green]Loading {name} from: {path}[/green]")
            state_dict = torch.load(path, map_location="cpu")
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

    selected_paths = cfg.data.datasets[task_mode]

    with open_dict(cfg):
        cfg.data.latent_dir = selected_paths.latent_dir
        cfg.data.eval_latent_dir = selected_paths.eval_latent_dir
        cfg.data.raw_root = selected_paths.raw_root

    console.print(f"📂 [Data] Training Latents: {cfg.data.latent_dir}")
    
    set_seed(cfg.training.seed)
    
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))
    training_args.ddp_find_unused_parameters = True 
    training_args.ignore_data_skip = True
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token_id = tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.eos_token_id
    
    tokenizer.padding_side = "right" 

    # 1. Model Initialization
    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path,
        vae_path=cfg.model.vae_path,
        head_type="flow", 
        use_precomputed_latents=cfg.model.use_precomputed_latents,
        latent_dim=cfg.model.latent_dim,
        audio_loss_weight=cfg.model.audio_loss_weight,
        downsample_rate=cfg.data.latent_downsample,
    )
    model = QwenCALM(config)

    # 2. Component Loading
    console.rule("[bold cyan]Component Loading[/bold cyan]")
    load_soft_restart_components(model, cfg, console)
    
    # 3. LoRA / MoA Initialization
    if cfg.model.use_lora:
        console.print("[blue]Initializing LoRA Config...[/blue]")
        lora_config = LoraConfig(
            r=cfg.model.lora_rank, lora_alpha=cfg.model.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.model.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM,
            modules_to_save=[], # [FIX] We handle saving manually in Trainer
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
            model.llm = get_peft_model(model.llm, lora_config, adapter_name="tts")
            model.llm.add_adapter("asr", lora_config)
            load_adapter_if_path_exists("tts", "pretrained_lora_path_tts")
            load_adapter_if_path_exists("asr", "pretrained_lora_path_asr")

    # 4. Freeze Strategy (Modified for Safety)
    should_freeze_proj = cfg.model.get("freeze_projector", False)
    
    # Input Projector (ASR Part)
    model.input_proj.requires_grad_(not should_freeze_proj)
    if should_freeze_proj:
        model.input_proj.eval()
        console.print("[bold yellow]🔒 Input Projector Frozen (Protecting ASR capabilities)[/bold yellow]")
    
    # Output Head (TTS Part) - Always Train
    model.output_head.requires_grad_(True)
    
    # [FIX] Explicitly unfreeze SOA Embed (Critical for TTS)
    if hasattr(model, "soa_embed"):
        model.soa_embed.requires_grad_(True)
        console.print("[bold green]🔓 SOA Embed Unfrozen (Ready for TTS training)[/bold green]")

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    console.print(f"🔥 Trainable Modules: {[n for n in trainable_params if 'bias' not in n][:10]} ...")

    console.rule()

    # 5. Trainer Setup
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
    
    train_collator = CalmCollator(tokenizer.pad_token_id, training=True)
    eval_collator = CalmCollator(tokenizer.pad_token_id, training=False)

    trainer = CalmTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=train_collator
    )
    
    trainer.eval_collator = eval_collator
    trainer.tokenizer = tokenizer

    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Final Save
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()