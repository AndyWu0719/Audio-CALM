"""
GMM joint training script for Audio-CALM (supports mix / tts / asr modes).
Refactored for clarity: English comments, sections, small robustness fixes.
"""

import os
import sys
import math
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from glob import glob

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# project imports
sys.path.append(os.getcwd())
from models.modeling_gmm import QwenCALM, QwenCALMConfig

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.traceback import install

install(show_locals=False)
console = Console()
warnings.filterwarnings("ignore", module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)

# DeepSpeed compatibility (keep as-is)
try:
    import deepspeed
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    torch.serialization.add_safe_globals([LossScaler])
except (ImportError, AttributeError):
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get_rank_safe() -> int:
    """Return process rank or 0 when not running distributed."""
    try:
        return dist.get_rank()
    except Exception:
        return 0

# ---------------------------------------------------------------------
# Dataset (supports mix / tts / asr)
# ---------------------------------------------------------------------
class CalmDataset(Dataset):
    """
    Dataset that pairs pretrained latent/mel files with LibriSpeech transcripts.
    Each item selects a task mode: 'tts', 'asr', or mix (random choice).
    """

    def __init__(
        self,
        data_dir: str,
        librispeech_root: str,
        subsets: str,
        tokenizer,
        max_text_len: int = 512,
        max_audio_len: int = 512,
        use_latents: bool = False,
        task_mode: str = "mix",
        task_prob_tts: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.use_latents = use_latents
        self.task_mode = task_mode
        self.task_prob_tts = task_prob_tts

        self.im_end_id = getattr(tokenizer, "im_end_id", None)
        if self.im_end_id is None:
            ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            self.im_end_id = ids[-1] if len(ids) > 0 else tokenizer.eos_token_id

        self.data: List[Dict[str, str]] = []
        subset_list = subsets.split(",")

        if _get_rank_safe() == 0:
            console.log(f"[green]Scanning subsets: {subset_list}[/green]")

        # Build index of available .pt data files
        file_index: Dict[str, str] = {}
        search_path = os.path.join(data_dir, "**", "*.pt")
        files = glob(search_path, recursive=True)
        for f in files:
            key = os.path.splitext(os.path.basename(f))[0]
            file_index[key] = f

        # Match transcripts in LibriSpeech subsets
        matched_count = 0
        for subset in subset_list:
            subset_dir = os.path.join(librispeech_root, subset.strip())
            if not os.path.exists(subset_dir):
                continue

            for root, dirs, txt_files in os.walk(subset_dir):
                for file in txt_files:
                    if file.endswith(".trans.txt"):
                        try:
                            with open(os.path.join(root, file), "r") as fh:
                                for line in fh:
                                    parts = line.strip().split(" ", 1)
                                    if len(parts) != 2:
                                        continue
                                    file_id, text = parts
                                    if file_id in file_index:
                                        self.data.append({"text": text, "file_path": file_index[file_id]})
                                        matched_count += 1
                        except Exception:
                            continue

        if _get_rank_safe() == 0:
            console.log(f"[green]Total matched pairs: {matched_count}[/green]")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            A dict containing 'text_ids', 'labels', 'audio_features', 'task_mode', '_valid'.
        """
        try:
            item = self.data[idx]
            text_content = item["text"]

            # Choose task mode dynamically based on config
            if self.task_mode == "tts":
                current_mode = "tts"
            elif self.task_mode == "asr":
                current_mode = "asr"
            else:
                current_mode = "tts" if random.random() < self.task_prob_tts else "asr"

            payload = torch.load(item["file_path"], map_location="cpu")
            if isinstance(payload, dict):
                audio = payload.get("latent", payload.get("mel", None))
            else:
                audio = payload

            if audio is None:
                return {"_valid": False}

            target_dim = 64  # expected latent dim
            # Fix shape: accept (T, D) or (D, T) forms
            if audio.shape[0] == target_dim and audio.shape[1] != target_dim:
                audio = audio.transpose(0, 1)
            elif audio.shape[1] == target_dim:
                pass
            else:
                return {"_valid": False}

            if audio.shape[1] != target_dim:
                return {"_valid": False}

            T = audio.shape[0]

            # Truncate long audio for TTS only (ASR requires full audio)
            if T > self.max_audio_len:
                if current_mode == "asr":
                    return {"_valid": False}
                else:
                    start = torch.randint(0, T - self.max_audio_len, (1,)).item()
                    audio = audio[start : start + self.max_audio_len, :]

            # Build prompt and tokenization
            if current_mode == "tts":
                prompt = f"Read this text:\n{text_content}"
            else:
                prompt = "Transcribe the following audio:"

            user_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            user_ids = self.tokenizer.encode(user_text, add_special_tokens=False)

            if current_mode == "tts":
                text_ids = user_ids
                labels = [-100] * len(text_ids)
            else:
                target_text = f"{text_content}<|im_end|>"
                target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
                text_ids = user_ids + target_ids
                labels = [-100] * len(user_ids) + target_ids

            # enforce max text length
            if len(text_ids) > self.max_text_len:
                text_ids = text_ids[: self.max_text_len]
                labels = labels[: self.max_text_len]

            return {
                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "audio_features": audio,
                "task_mode": current_mode,
                "_valid": True,
            }

        except Exception:
            return {"_valid": False}

# ---------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------
@dataclass
class CalmCollator:
    pad_token_id: int
    audio_pad_val: float = 0.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f.get("_valid", False)]
        if not valid_features:
            # return a minimal dummy batch to avoid crashes
            return {
                "text_input_ids": torch.tensor([[self.pad_token_id]], dtype=torch.long),
                "attention_mask": torch.tensor([[0]], dtype=torch.long),
                "labels": torch.tensor([[-100]], dtype=torch.long),
                "audio_features": torch.zeros(1, 1, 64),
                "audio_lens": torch.tensor([1], dtype=torch.long),
                "task_modes": ["tts"],
            }

        text_ids = [f["text_ids"] for f in valid_features]
        labels = [f["labels"] for f in valid_features]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (padded_input_ids != self.pad_token_id).long()

        audio_feats = [f["audio_features"] for f in valid_features]
        audio_lens = torch.tensor([f.shape[0] for f in audio_feats], dtype=torch.long)

        padded_audio = torch.nn.utils.rnn.pad_sequence(audio_feats, batch_first=True, padding_value=self.audio_pad_val)
        padded_audio = padded_audio.transpose(1, 2)  # [B, D, T]

        task_modes = [f["task_mode"] for f in valid_features]

        return {
            "text_input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "audio_features": padded_audio,
            "audio_lens": audio_lens,
            "task_modes": task_modes,
        }

# ---------------------------------------------------------------------
# Custom Trainer (tracks per-task loss)
# ---------------------------------------------------------------------
class CalmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_meters = {"tts": 0.0, "asr": 0.0, "tts_cnt": 0, "asr_cnt": 0}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            text_input_ids=inputs["text_input_ids"],
            audio_features=inputs["audio_features"],
            attention_mask=inputs["attention_mask"],
            audio_lens=inputs["audio_lens"],
            labels=inputs["labels"],
            task_modes=inputs["task_modes"],
        )
        loss = outputs["loss"]

        if self.model.training:
            task_modes = inputs.get("task_modes", [])
            loss_tts = outputs.get("loss_tts", torch.tensor(0.0)).detach()
            loss_asr = outputs.get("loss_asr", torch.tensor(0.0)).detach()

            num_tts = task_modes.count("tts")
            num_asr = task_modes.count("asr")

            self.loss_meters["tts"] += loss_tts.item() * num_tts
            self.loss_meters["asr"] += loss_asr.item() * num_asr
            self.loss_meters["tts_cnt"] += num_tts
            self.loss_meters["asr_cnt"] += num_asr

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        tts_cnt = max(self.loss_meters["tts_cnt"], 1)
        asr_cnt = max(self.loss_meters["asr_cnt"], 1)
        logs["loss_tts"] = round(self.loss_meters["tts"] / tts_cnt, 4)
        logs["loss_asr"] = round(self.loss_meters["asr"] / asr_cnt, 4)
        self.loss_meters = {"tts": 0.0, "asr": 0.0, "tts_cnt": 0, "asr_cnt": 0}
        super().log(logs, *args, **kwargs)

# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../config", config_name="gmm_config")
def main(cfg: DictConfig):
    console.rule("[magenta]GMM Joint Training (Unified Strategy)[/magenta]")

    set_seed(cfg.training.seed)
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.padding_side = "right"

    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path,
        vae_path=cfg.model.vae_path,
        num_mixtures=cfg.model.num_mixtures,
        use_precomputed_latents=cfg.model.use_precomputed_latents,
        latent_dim=cfg.model.latent_dim,
        audio_loss_weight=cfg.model.audio_loss_weight,
        downsample_rate=cfg.data.latent_downsample,
    )
    model = QwenCALM(config)

    # 1) Try to load pretrained projector if provided
    if cfg.model.get("pretrained_projector_path", None):
        path = cfg.model.pretrained_projector_path
        console.print(f"[green]Loading pretrained input projector from: {path}[/green]")
        state_dict = torch.load(path, map_location="cpu")
        # support keys like 'input_proj.*' or raw param keys
        if any(k.startswith("input_proj.") for k in state_dict.keys()):
            state_dict = {k.replace("input_proj.", ""): v for k, v in state_dict.items()}

        try:
            model.input_proj.load_state_dict(state_dict)
        except Exception as e:
            console.print(f"[red]Error loading projector: {e}[/red]")

    # 2) Configure LoRA (if enabled)
    if cfg.model.use_lora:
        lora_config = LoraConfig(
            r=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.model.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["input_proj", "output_head"],  # ensure these stay available for saving
        )
        model.llm = get_peft_model(model.llm, lora_config)

        # Load pretrained LoRA adapter if provided
        if cfg.model.get("pretrained_lora_path", None):
            lora_path = cfg.model.pretrained_lora_path
            console.print(f"[green]Loading pretrained LoRA adapter from: {lora_path}[/green]")

            adapter_bin_path = os.path.join(lora_path, "adapter_model.bin")
            adapter_safe_path = os.path.join(lora_path, "adapter_model.safetensors")

            adapter_weights = None
            if os.path.exists(adapter_safe_path):
                console.print(f"[green]Detected safetensors: {adapter_safe_path}[/green]")
                from safetensors.torch import load_file as load_safetensors
                adapter_weights = load_safetensors(adapter_safe_path)
            elif os.path.exists(adapter_bin_path):
                console.print(f"[green]Detected bin: {adapter_bin_path}[/green]")
                adapter_weights = torch.load(adapter_bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"Could not find adapter_model.bin or adapter_model.safetensors in {lora_path}")

            from peft.utils import set_peft_model_state_dict
            set_peft_model_state_dict(model.llm, adapter_weights)

        console.print("[green]LoRA initialized.[/green]")

    # -----------------------------------------------------------------
    # Dynamic freeze strategy
    # -----------------------------------------------------------------
    should_freeze_proj = cfg.model.get("freeze_projector", False)

    if should_freeze_proj:
        model.input_proj.requires_grad_(False)
        console.print("[yellow]Strategy: Mix/TTS -> Input projector is FROZEN.[/yellow]")
        model.output_head.requires_grad_(True)
        # Ensure input_proj parameters remain frozen (PEFT may change state)
        for n, p in model.named_parameters():
            if "input_proj" in n:
                p.requires_grad = False
    else:
        model.input_proj.requires_grad_(True)
        console.print("[green]Strategy: ASR -> Input projector is TRAINABLE.[/green]")
        model.output_head.requires_grad_(True)

    # Always keep LoRA params trainable
    for n, p in model.llm.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    console.print("[bold]Active training modules (top-level):[/bold]")
    active_modules = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            module_name = n.split(".")[0]
            if module_name == "base_model":  # peft wrapper case
                module_name = n.split(".")[2]
            active_modules.add(module_name)
    console.print(active_modules)
    console.print(f"[cyan]Total Trainable: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})[/cyan]")

    # Initialize datasets and trainer
    train_dataset = CalmDataset(
        data_dir=cfg.data.mel_dir,
        librispeech_root=cfg.data.librispeech_root,
        subsets=cfg.data.train_subsets,
        tokenizer=tokenizer,
        max_text_len=cfg.data.max_text_len,
        max_audio_len=cfg.data.max_audio_len,
        use_latents=cfg.model.use_precomputed_latents,
        task_mode=cfg.data.task_mode,
        task_prob_tts=cfg.data.task_prob_tts,
    )

    eval_dataset = CalmDataset(
        data_dir=cfg.data.eval_mel_dir or cfg.data.mel_dir,
        librispeech_root=cfg.data.librispeech_root,
        subsets=cfg.data.eval_subsets,
        tokenizer=tokenizer,
        max_text_len=cfg.data.max_text_len,
        max_audio_len=cfg.data.max_audio_len,
        use_latents=cfg.model.use_precomputed_latents,
        task_mode="mix",
        task_prob_tts=0.5,
    )

    trainer = CalmTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CalmCollator(pad_token_id=tokenizer.pad_token_id),
    )

    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()