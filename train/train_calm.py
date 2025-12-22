"""
Flow-based Audio-CALM Training Script.
Optimized for Speed, DDP Stability (Ghost Gradients), and Mixture of Adapters (MoA).
"""

import os
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
                 max_audio_len=1024, use_latents=False, task_mode="mix", task_prob_tts=0.5):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.task_mode = task_mode
        self.task_prob_tts = task_prob_tts
        self.latent_dir = latent_dir
        
        # Handle tokenizer special tokens
        if hasattr(tokenizer, "eod_id"):
             self.im_end_id = tokenizer.eod_id
        else:
             enc = tokenizer.encode("<|im_end|>", add_special_tokens=False)
             self.im_end_id = enc[-1] if len(enc)>0 else tokenizer.eos_token_id

        # =================================================================
        # Scanning Logic (Recursive Scan in Latent Dir)
        # =================================================================
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
                console.log(f"   Please check if your config 'train_subsets' matches the folder structure.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            # 1. Determine Task Mode
            if self.task_mode == "mix":
                mode = "tts" if random.random() < self.task_prob_tts else "asr"
            else:
                mode = self.task_mode

            # 2. Load Audio Latent
            payload = torch.load(item["file_path"], map_location="cpu")
            audio = payload.get("latent", payload) if isinstance(payload, dict) else payload
            if audio is None: return {"_valid": False}
            
            # Ensure shape is [T, Dim]
            if audio.shape[0] == 64: audio = audio.transpose(0, 1)
            
            # 3. Audio Cropping
            # ASR requires strict alignment (drop if too long), TTS can be random cropped
            if audio.shape[0] > self.max_audio_len:
                if mode == "asr": 
                    return {"input_ids": [0], "_valid": False}
                else:
                    start = random.randint(0, audio.shape[0] - self.max_audio_len)
                    audio = audio[start : start + self.max_audio_len]

            # 4. Construct Prompt
            prompt = f"Read this text:\n{item['text']}" if mode == "tts" else "Transcribe the following audio:"
            user_txt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            user_ids = self.tokenizer.encode(user_txt, add_special_tokens=False)
            
            if mode == "tts":
                text_ids = user_ids
                labels = [-100] * len(text_ids)
            else:
                target_txt = f"{item['text']}<|im_end|>"
                target_ids = self.tokenizer.encode(target_txt, add_special_tokens=False)
                
                # Smart truncation to ensure EOS token exists
                max_target_len = self.max_text_len - len(user_ids)
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]
                    if self.im_end_id is not None:
                        target_ids[-1] = self.im_end_id

                text_ids = user_ids + target_ids
                labels = [-100] * len(user_ids) + target_ids

            # Final length check
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
                audio_feat[:, t0 : t0 + mask_len].fill_(0.0)
        return audio_feat

    def __call__(self, features):
        valid = [f for f in features if f.get("_valid", False)]
        
        # Handle empty batch
        if not valid:
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
            "labels": torch.nn.utils.rnn.pad_sequence([f["labels"] for f in valid], batch_first=True, padding_value=-100),
            "audio_features": torch.nn.utils.rnn.pad_sequence(proc_audio, batch_first=True, padding_value=self.audio_pad_val).transpose(1, 2),
            "audio_lens": torch.tensor([f.shape[0] for f in proc_audio], dtype=torch.long),
            "task_modes": [f["task_mode"] for f in valid]
        }
        
        batch["attention_mask"] = (batch["text_input_ids"] != self.pad_token_id).long()
        return batch

# ---------------------------------------------------------------------
# Trainer & Optimization
# ---------------------------------------------------------------------
class CalmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_meters = {"tts": 0.0, "asr": 0.0, "tts_cnt": 0, "asr_cnt": 0}

    def create_optimizer(self):
        """
        Custom optimizer setup: 
        Applies 20x Learning Rate to the Projector/Head vs the LLM Backbone.
        """
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            projector_parameters = []
            
            head_keywords = ["input_proj", "output_head"]
            
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
                # High LR for Projectors
                {"params": projector_parameters, "weight_decay": self.args.weight_decay, "lr": 1.0 * self.args.learning_rate}, 
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Unpack Model
        peft_model = model.module.llm if hasattr(model, "module") else model.llm
        raw_model = model.module if hasattr(model, "module") else model

        # 2. Switch Adapter (MoA Strategy)
        task_modes = inputs.get("task_modes", ["tts"])
        # Majority vote determines batch adapter context
        target_adapter = "tts" if task_modes.count("tts") >= task_modes.count("asr") else "asr"
        
        if hasattr(peft_model, "set_adapter"):
            if target_adapter in peft_model.peft_config:
                peft_model.set_adapter(target_adapter)
        
        # 3. Forward Pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        # 4. [Ghost Gradients] DDP Deadlock Prevention
        if self.model.training:
            dummy_loss = sum(p.view(-1)[0] * 0.0 for p in raw_model.parameters() if p.requires_grad)
            loss += dummy_loss

        # 5. Logging
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
    """Loads specific components (Projector/Head) from a checkpoint."""
    def _load(key, model_attr, name):
        path = cfg.model.get(key, None)
        if path and os.path.exists(path):
            console.print(f"[green]Loading {name} from: {path}[/green]")
            state_dict = torch.load(path, map_location="cpu")
            # Clean keys
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
    print(f"🔄 [Config] Task Mode: {task_mode.upper()}")

    if task_mode not in cfg.data.datasets:
        raise ValueError(f"❌ Unknown task_mode: '{task_mode}'. Available: {list(cfg.data.datasets.keys())}")

    selected_paths = cfg.data.datasets[task_mode]

    with open_dict(cfg):
        cfg.data.latent_dir = selected_paths.latent_dir
        cfg.data.eval_latent_dir = selected_paths.eval_latent_dir
        cfg.data.raw_root = selected_paths.raw_root

    print(f"📂 [Data] Training Latents: {cfg.data.latent_dir}")
    print(f"📂 [Data] Eval Latents:     {cfg.data.eval_latent_dir}")

    set_seed(cfg.training.seed)
    
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))
    training_args.ignore_data_skip = True
    training_args.ddp_timeout = 10800
    
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

    # 2. Soft Restart (Load Head/Projector)
    console.rule("[bold cyan]Component Loading[/bold cyan]")
    load_soft_restart_components(model, cfg, console)
    
    # 3. LoRA / MoA Initialization
    if cfg.model.use_lora:
        console.print("[blue]Initializing LoRA Config...[/blue]")
        lora_config = LoraConfig(
            r=cfg.model.lora_rank, lora_alpha=cfg.model.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.model.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM,
            modules_to_save=["input_proj", "output_head"], 
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
        
        # Initialize Adapters based on Task Mode
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

    # 4. Freeze Strategy
    should_freeze_proj = cfg.model.get("freeze_projector", False)
    model.input_proj.requires_grad_(not should_freeze_proj)
    model.output_head.requires_grad_(True)
    
    for n, p in model.llm.named_parameters():
        if "lora_" in n: p.requires_grad = True

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
        task_prob_tts=cfg.data.task_prob_tts
    )
    eval_ds = CalmDataset(
        latent_dir=cfg.data.eval_latent_dir or cfg.data.latent_dir, 
        subsets=cfg.data.eval_subsets, 
        tokenizer=tokenizer, 
        max_text_len=cfg.data.max_text_len, 
        max_audio_len=cfg.data.max_audio_len, 
        use_latents=cfg.model.use_precomputed_latents, 
        task_mode=cfg.data.task_mode, 
        task_prob_tts=cfg.data.task_prob_tts
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
    
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()