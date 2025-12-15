import os
import sys
import torch
import random
from dataclasses import dataclass, field
import torch.distributed as dist
from typing import List, Dict, Any
from glob import glob
import math

sys.path.append(os.getcwd())

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed, AutoTokenizer
from torch.optim import AdamW
from models.modeling_gmm import QwenCALM, QwenCALMConfig 
from peft import LoraConfig, get_peft_model, TaskType
from functools import partial

import warnings
warnings.filterwarnings("ignore", module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*barrier().*")
try:
    import deepspeed
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    torch.serialization.add_safe_globals([LossScaler])
except ImportError:
    pass
except AttributeError:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class ModelArguments:
    qwen_path: str = field(metadata={"help": "Path to Qwen2-7B-Instruct folder"})
    vae_path: str = field(metadata={"help": "Path to trained VAE checkpoint folder"})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA"})
    lora_rank: int = field(default=64, metadata={"help": "LoRA Rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA Alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA Dropout"})
    use_precomputed_latents: bool = field(default=False, metadata={"help": "Set True if using latent .pt files"})
    num_mixtures: int = field(default=8, metadata={"help": "Number of Gaussian Mixtures for MDN Head"})
    latent_dim: int = field(default=64, metadata={"help": "Dimension of VAE Latents"})
    audio_loss_weight: float = field(default=1.0, metadata={"help": "Weight for GMM Loss"})

@dataclass
class DataArguments:
    librispeech_root: str = field(metadata={"help": "Path to original LibriSpeech dataset root"})
    mel_dir: str = field(metadata={"help": "Path to data directory (Mel .pt or Latent .pt)"})
    eval_mel_dir: str = field(default=None, metadata={"help": "Path to EVAL data directory"})
    train_subsets: str = field(default="train-clean-100,train-clean-360,train-other-500", metadata={"help": "Training subsets"})
    eval_subsets: str = field(default="dev-clean", metadata={"help": "Eval subsets"})
    max_text_len: int = field(default=512, metadata={"help": "Max text sequence length"})
    max_audio_len: int = field(default=512, metadata={"help": "Max audio latent sequence length"})
    latent_downsample: int = field(default=16, metadata={"help": "Downsample rate of the VAE (e.g. 4 or 16)"})
    task_mode: str = field(default="mix", metadata={"help": "mix | tts | asr"})
    task_prob_tts: float = field(default=0.5, metadata={"help": "Probability of sampling TTS in mix mode"})

# =============================================================================
# 1. Improved Dataset (Smart Truncation + Exception Filtering)
# =============================================================================
class CalmDataset(Dataset):
    def __init__(self, data_dir, librispeech_root, subsets, tokenizer, max_text_len=512, max_audio_len=512, 
                 use_latents=False, task_mode="mix", task_prob_tts=0.5):
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

        self.data = []
        subset_list = subsets.split(",")
        if torch.distributed.get_rank() == 0:
            print(f"Scanning pairs in {subset_list}...")

        # 1. Build Index
        file_index = {}
        for subset in subset_list:
            subset_dir = os.path.join(data_dir, subset.strip())
            if not os.path.exists(subset_dir): continue
            
            # Recursive scan
            files = glob(os.path.join(subset_dir, "**", "*.pt"), recursive=True)
            for f in files:
                key = os.path.splitext(os.path.basename(f))[0]
                file_index[key] = f
        
        # 2. Match Transcripts
        matched_count = 0
        for subset in subset_list:
            subset_dir = os.path.join(librispeech_root, subset.strip())
            if not os.path.exists(subset_dir): continue

            for root, dirs, files in os.walk(subset_dir):
                for file in files:
                    if file.endswith(".trans.txt"):
                        try:
                            with open(os.path.join(root, file), "r") as f:
                                for line in f:
                                    parts = line.strip().split(" ", 1)
                                    if len(parts) != 2: continue
                                    file_id, text = parts

                                    if file_id in file_index:
                                        self.data.append({
                                            "text": text,
                                            "file_path": file_index[file_id]
                                        })
                                        matched_count += 1
                        except Exception:
                            continue
        if torch.distributed.get_rank() == 0:
            print(f"Total matched pairs: {matched_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            text_content = item['text']

            # Determine Task Mode
            if self.task_mode == "tts":
                current_mode = "tts"
            elif self.task_mode == "asr":
                current_mode = "asr"
            else:
                current_mode = "tts" if random.random() < self.task_prob_tts else "asr"

            payload = torch.load(item['file_path'], map_location="cpu")
            if isinstance(payload, dict):
                audio = payload["latent"]
            else:
                audio = payload

            # audio: [64, T] or [T, 64]，统一处理
            if audio.dim() == 2:
                # 假设预处理保存的是 [64, T]，需要 transpose
                if audio.shape[0] == 64:
                    audio = audio.transpose(0, 1)  # -> [T, 64]

            T = audio.shape[0]

            # ================== [FIX] ASR 不允许随机裁剪 ==================
            if T > self.max_audio_len:
                if current_mode == "asr":
                    # 丢弃超长样本，避免音频与 transcript 不对齐
                    return {"_valid": False}
                else:
                    # TTS 可以随机裁剪
                    start = torch.randint(0, T - self.max_audio_len, (1,)).item()
                    audio = audio[start:start + self.max_audio_len, :]

            # ================== Build Prompt ==================
            if current_mode == "tts":
                messages = [{"role": "user", "content": f"Read this text:\n{text_content}"}]
            else:
                messages = [
                    {"role": "user", "content": "Transcribe the following audio:"},
                    {"role": "assistant", "content": text_content}
                ]

            user_text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
            user_ids = self.tokenizer.encode(user_text, add_special_tokens=False)

            if current_mode == "tts":
                text_ids = user_ids
                labels = [-100] * len(text_ids)
            else:
                target_text = f"{messages[1]['content']}<|im_end|>"
                target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)

                # [FIX] 截断时保留 im_end
                max_target_len = max(1, self.max_text_len - len(user_ids))
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]
                    if self.im_end_id is not None:
                        target_ids[-1] = self.im_end_id

                text_ids = user_ids + target_ids
                labels = [-100] * len(user_ids) + target_ids

            # 最终保险
            if len(text_ids) > self.max_text_len:
                text_ids = text_ids[:self.max_text_len]
                labels = labels[:self.max_text_len]

            return {
                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "audio_features": audio, 
                "task_mode": current_mode,
                "_valid": True
            }

        except Exception as e:
            return {"_valid": False}

# =============================================================================
# 2. Dynamic Data Collator (With Filtering)
# =============================================================================
@dataclass
class CalmCollator:
    pad_token_id: int
    audio_pad_val: float = 0.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f.get("_valid", False)]
        
        if not valid_features:
            dummy = {
                "text_ids": torch.tensor([self.pad_token_id] * 2, dtype=torch.long),
                "labels": torch.tensor([-100] * 2, dtype=torch.long),
                "audio_features": torch.zeros(2, 64),
                "task_mode": "asr",
                "_valid": True,
            }
            valid_features = [dummy]

        # 2. Text Padding (Right Padding)
        text_ids = [f["text_ids"] for f in valid_features]
        labels = [f["labels"] for f in valid_features]
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            text_ids, batch_first=True, padding_value=self.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = (padded_input_ids != self.pad_token_id).long()

        # 3. Audio Padding (Dynamic)
        audio_feats = [f["audio_features"] for f in valid_features] 
        audio_lens = torch.tensor([f.shape[0] for f in audio_feats], dtype=torch.long)
        
        padded_audio = torch.nn.utils.rnn.pad_sequence(
            audio_feats, batch_first=True, padding_value=self.audio_pad_val
        )
        # [B, T, D] -> [B, D, T] (Model Expectation)
        padded_audio = padded_audio.transpose(1, 2) 

        task_modes = [f["task_mode"] for f in valid_features]

        return {
            "text_input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "audio_features": padded_audio, 
            "audio_lens": audio_lens,
            "task_modes": task_modes
        }

# =============================================================================
# 3. Custom Trainer
# =============================================================================
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
            task_modes=inputs["task_modes"]
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

    def create_optimizer(self):
        if self.optimizer is None:
            decay_params, no_decay_params = [], []
            projector_params = []
            
            head_keywords = ["input_proj", "output_head"]

            for name, param in self.model.named_parameters():
                if not param.requires_grad: continue

                if any(k in name for k in head_keywords):
                    projector_params.append(param)
                else:
                    if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

            opt_grouped_params = [
                {
                    "params": decay_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate, 
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": projector_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 10.0, 
                },
            ]

            self.optimizer = AdamW(
                opt_grouped_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer

# =============================================================================
# 4. Main Execution
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.padding_side = 'right' 
    
    # 2. Config & Model
    config = QwenCALMConfig(
        qwen_path=model_args.qwen_path,
        vae_path=model_args.vae_path,
        num_mixtures=model_args.num_mixtures,
        use_precomputed_latents=model_args.use_precomputed_latents,
        latent_dim=model_args.latent_dim,
        audio_loss_weight=model_args.audio_loss_weight,
        downsample_rate=data_args.latent_downsample 
    )
    model = QwenCALM(config)

    # 3. LoRA Setup
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["input_proj", "output_head"] 
        )
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm.print_trainable_parameters()
        
        print("Checking trainable parameters...")
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Params: {total_trainable}")
    
    # Ensure Projector/Head are trainable
    for p in model.input_proj.parameters(): p.requires_grad = True
    for p in model.output_head.parameters(): p.requires_grad = True

    # 4. Data
    train_dataset = CalmDataset(
        data_dir=data_args.mel_dir,
        librispeech_root=data_args.librispeech_root,
        subsets=data_args.train_subsets,
        tokenizer=tokenizer,
        max_text_len=data_args.max_text_len,
        max_audio_len=data_args.max_audio_len,
        use_latents=model_args.use_precomputed_latents,
        task_mode=data_args.task_mode,
        task_prob_tts=data_args.task_prob_tts
    )

    eval_dir = data_args.eval_mel_dir if data_args.eval_mel_dir else data_args.mel_dir
    eval_dataset = CalmDataset(
        data_dir=eval_dir,
        librispeech_root=data_args.librispeech_root,
        subsets=data_args.eval_subsets,
        tokenizer=tokenizer,
        max_text_len=data_args.max_text_len,
        max_audio_len=data_args.max_audio_len,
        use_latents=model_args.use_precomputed_latents,
        task_mode=data_args.task_mode,
        task_prob_tts=data_args.task_prob_tts 
    )

    trainer = CalmTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CalmCollator(pad_token_id=tokenizer.pad_token_id)
    )

    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()