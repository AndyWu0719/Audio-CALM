import os
import sys
import torch
import random
from dataclasses import dataclass, field
import torch.distributed as dist
from typing import List, Dict
from glob import glob

sys.path.append(os.getcwd())

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed, AutoTokenizer
from torch.optim import AdamW
from models.modeling_calm import QwenCALM, QwenCALMConfig
from peft import LoraConfig, get_peft_model, TaskType
from functools import partial

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class ModelArguments:
    qwen_path: str = field(metadata={"help": "Path to Qwen-Audio pretrained folder"})
    vae_path: str = field(metadata={"help": "Path to trained VAE checkpoint folder"})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA"})
    lora_rank: int = field(default=64, metadata={"help": "LoRA Rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA Alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA Dropout"})
    use_precomputed_latents: bool = field(default=False, metadata={"help": "If True, load latents directly from disk to speed up training."})

@dataclass
class DataArguments:
    librispeech_root: str = field(metadata={"help": "Path to original LibriSpeech dataset root"})
    mel_dir: str = field(metadata={"help": "Path to data directory (Mel .pt or Latent .pt)"})
    eval_mel_dir: str = field(default=None, metadata={"help": "Path to EVAL data directory. If None, use mel_dir"})
    train_subsets: str = field(default="train-clean-100,train-clean-360,train-other-500")
    eval_subsets: str = field(default="dev-clean")
    max_text_len: int = field(default=256)
    max_audio_len: int = field(default=2048, metadata={"help": "Max Mel frames. If using latents, this will be scaled down by 16 automatically inside dataset logic."})
    latent_downsample: int = field(default=16)

class CalmDataset(Dataset):
    def __init__(self, data_dir, librispeech_root, subsets, tokenizer, max_text_len=256, max_audio_len=2048, use_latents=False, latent_downsample=16):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.use_latents = use_latents
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eod_id
        self.latent_downsample = latent_downsample
        
        self.data = []
        subset_list = subsets.split(",")
        print(f"Scanning pairs in {subset_list}...")
        
        # 1. Build file index
        file_index = {}
        for subset in subset_list:
            subset_dir = os.path.join(data_dir, subset.strip())
            if not os.path.exists(subset_dir): 
                print(f"Warning: Directory not found: {subset_dir}")
                continue
            
            files = glob(os.path.join(subset_dir, "*.pt"))
            for f in files:
                key = os.path.splitext(os.path.basename(f))[0]
                file_index[key] = f
        
        print(f"Indexed {len(file_index)} audio files.")
        
        # 2. Scan text
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
                        except Exception as e:
                            pass
        print(f"Total matched pairs: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text_ids = [self.pad_id]
        labels = [-100]
        audio = torch.zeros(64 if self.use_latents else 80, 10)
        task_mode = "tts" 
        real_audio_len = 10

        try:
            item = self.data[idx]
            text_content = item['text']
            
            # === Task ===
            task_mode = "tts" if random.random() < 0.5 else "asr"
            
            if task_mode == "tts":
                prompt = f"<|im_start|>user\nRead this: {text_content}<|im_end|>\n<|im_start|>assistant\n"
                text_ids = self.tokenizer.encode(prompt)
                labels = [-100] * len(text_ids)
            else:
                prompt = f"<|im_start|>user\nTranscribe audio:<|im_end|>\n<|im_start|>assistant\n"
                prompt_ids = self.tokenizer.encode(prompt)
                target_ids = self.tokenizer.encode(text_content + "<|im_end|>")
                text_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids

            if len(text_ids) > self.max_text_len:
                text_ids = text_ids[:self.max_text_len]
                labels = labels[:self.max_text_len]

            # === Audio ===
            # Load: latents, shape [64, T] or mel, shape [80, T]
            audio = torch.load(item['file_path'], map_location="cpu") 
            
            # Calculate Target Length
            if self.use_latents:
                # Max Audio Len refers to Mel frames, Latent should be / total_stride
                TARGET_LEN = self.max_audio_len // self.latent_downsample
            else:
                TARGET_LEN = self.max_audio_len

            real_audio_len = audio.shape[1]
            
            if audio.shape[1] > TARGET_LEN:
                start = torch.randint(0, audio.shape[1] - TARGET_LEN, (1,)).item()
                audio = audio[:, start:start+TARGET_LEN]
                real_audio_len = TARGET_LEN
            else:
                pad_len = TARGET_LEN - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, pad_len))
                # real_audio_len remains the original value
                
        except Exception as e:
            print(f"[Dataset Error] idx={idx}: {e}")

        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "audio_features": audio,
            "task_mode": task_mode,
            "real_audio_len": real_audio_len
        }

def data_collator(features, pad_id):
    valid_features = [f for f in features if f["audio_features"].shape[1] > 0]
    if not valid_features:
        valid_features = features

    text_ids = [f["text_ids"] for f in valid_features]
    labels = [f["labels"] for f in valid_features]

    max_len = max(len(t) for t in text_ids)
    padded_text_ids, padded_labels, text_attention_masks = [], [], []

    for t, l in zip(text_ids, labels):
        pad_len = max_len - len(t)
        p_t = torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), t])
        p_l = torch.cat([torch.full((pad_len,), -100, dtype=torch.long), l])
        mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(len(t), dtype=torch.long)])
        padded_text_ids.append(p_t)
        padded_labels.append(p_l)
        text_attention_masks.append(mask)

    text_input_ids = torch.stack(padded_text_ids)
    labels_padded = torch.stack(padded_labels)
    attention_mask = torch.stack(text_attention_masks)
    
    # Audio
    audio_features = torch.stack([f["audio_features"] for f in valid_features])
    audio_lens = torch.tensor([f["real_audio_len"] for f in valid_features], dtype=torch.long)
    task_modes = [f["task_mode"] for f in valid_features]
    
    return {
        "text_input_ids": text_input_ids,
        "attention_mask": attention_mask,
        "audio_features": audio_features,
        "audio_lens": audio_lens,
        "labels": labels_padded,
        "task_modes": task_modes
    }

class CalmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meter_tts_loss = 0.0
        self.meter_tts_count = 0
        self.meter_asr_loss = 0.0
        self.meter_asr_count = 0

    def compute_loss(self, model, inputs, return_outputs=False):
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
            num_tts = task_modes.count("tts")
            num_asr = task_modes.count("asr")
            
            avg_tts = outputs.get("loss_tts", torch.tensor(0.0)).item()
            avg_asr = outputs.get("loss_asr", torch.tensor(0.0)).item()
            
            self.meter_tts_loss += avg_tts * num_tts
            self.meter_tts_count += num_tts
            self.meter_asr_loss += avg_asr * num_asr
            self.meter_asr_count += num_asr
             
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        if self.args.world_size > 1:
            metrics = torch.tensor([
                self.meter_tts_loss, float(self.meter_tts_count),
                self.meter_asr_loss, float(self.meter_asr_count)
            ], device=self.args.device)
            
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            tts_sum, tts_cnt = metrics[0].item(), metrics[1].item()
            asr_sum, asr_cnt = metrics[2].item(), metrics[3].item()
        else:
            tts_sum, tts_cnt = self.meter_tts_loss, self.meter_tts_count
            asr_sum, asr_cnt = self.meter_asr_loss, self.meter_asr_count

        logs["tts_loss"] = round(tts_sum / tts_cnt, 4) if tts_cnt > 0 else 0.0
        logs["asr_loss"] = round(asr_sum / asr_cnt, 4) if asr_cnt > 0 else 0.0
        
        self.meter_tts_loss = 0.0
        self.meter_tts_count = 0
        self.meter_asr_loss = 0.0
        self.meter_asr_count = 0

        super().log(logs)

    def create_optimizer(self):
        """
        Override the default optimizer to set specific learning rates for Projector and Head.
        """
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            
            # Define parameter groups for specific components
            projector_parameters = []
            
            # 1. Distinguish Projector/Head and other parameters (LLM/LoRA)
            # Note: model.input_proj and model.output_head are members of QwenCALM
            head_names = ["input_proj", "output_head"]
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Check if belongs to Projector or Head
                is_head = any(n in name for n in head_names)
                
                if is_head:
                    projector_parameters.append(param)
                else:
                    # Belongs to LLM (LoRA) parameters
                    if "bias" in name or "LayerNorm" in name:
                        no_decay_parameters.append(param)
                    else:
                        decay_parameters.append(param)

            # 2. Set parameter groups
            
            optimizer_grouped_parameters = [
                {
                    "params": decay_parameters,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": projector_parameters,
                    "weight_decay": self.args.weight_decay,
                    "lr": 20 * self.args.learning_rate,
                },
            ]

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            
        return self.optimizer

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    if not training_args.ddp_find_unused_parameters:
        print("Warning: Enforcing ddp_find_unused_parameters=True for mixed tasks.")
        training_args.ddp_find_unused_parameters = True
    
    # 1. Tokenizer (Set Left Padding)
    tokenizer = AutoTokenizer.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.padding_side = 'left' 
    print(f"Tokenizer padding side: {tokenizer.padding_side}")

    # 2. Config & Model
    calm_config = QwenCALMConfig(
        qwen_path=model_args.qwen_path,
        vae_path=model_args.vae_path,
        num_mixtures=8,
        use_precomputed_latents=model_args.use_precomputed_latents # Pass to config
    )
    
    model = QwenCALM(calm_config)
    
    # 3. LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=["c_attn", "c_proj", "w1", "w2"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm.print_trainable_parameters()
    
    # 4. Datasets
    train_dataset = CalmDataset(
        data_dir=data_args.mel_dir, 
        librispeech_root=data_args.librispeech_root,
        subsets=data_args.train_subsets,
        tokenizer=tokenizer,
        max_text_len=data_args.max_text_len,
        max_audio_len=data_args.max_audio_len,
        use_latents=model_args.use_precomputed_latents,
        latent_downsample=data_args.latent_downsample
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
        latent_downsample=data_args.latent_downsample
    )
    
    trainer = CalmTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(data_collator, pad_id=tokenizer.pad_token_id)
    )
    
    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()