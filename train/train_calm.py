import os
import sys
import torch
import random
from dataclasses import dataclass, field
from typing import List
from glob import glob

sys.path.append(os.getcwd())

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed, AutoTokenizer
from torch.optim import AdamW
from models.modeling_calm import QwenCALM, QwenCALMConfig
from peft import LoraConfig, get_peft_model, TaskType

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
    # 新增参数
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

class CalmDataset(Dataset):
    def __init__(self, data_dir, librispeech_root, subsets, tokenizer, max_text_len=256, max_audio_len=2048, use_latents=False):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.use_latents = use_latents
        
        # 优先使用 eod 作为 pad，如果没有则用 pad_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eod_id
        
        self.data = []
        subset_list = subsets.split(",")
        print(f"Scanning pairs in {subset_list}...")
        
        # 1. 建立文件索引
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
        
        # 2. 扫描文本
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
        # 默认值
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
            # Load: 如果是 latents, shape [64, T]; 如果是 mel, shape [80, T]
            audio = torch.load(item['file_path'], map_location="cpu") 
            
            # 计算 Target Length
            if self.use_latents:
                # Max Audio Len 指的是 Mel 帧数，Latent 应该是 / 16
                TARGET_LEN = self.max_audio_len // 16
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
                # real_audio_len 保持原始值
                
        except Exception as e:
            print(f"[Dataset Error] idx={idx}: {e}")

        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "audio_features": audio,
            "task_mode": task_mode,
            "real_audio_len": real_audio_len
        }

def data_collator(features):
    valid_features = [f for f in features if f["audio_features"].shape[1] > 0]
    if not valid_features: valid_features = features
        
    text_ids = [f["text_ids"] for f in valid_features]
    labels = [f["labels"] for f in valid_features]
    
    # Pad Token ID (从 valid_features 中无法直接获取 tokenizer，假设 eod 151643)
    # 更好的做法是在 main 中把 tokenizer 传给 data_collator，或者硬编码
    PAD_ID = 151643 
    
    # === Manual Left Padding ===
    max_len = max(len(t) for t in text_ids)
    padded_text_ids = []
    padded_labels = []
    text_attention_masks = []
    
    for t, l in zip(text_ids, labels):
        pad_len = max_len - len(t)
        # Text: [Pad, ..., Pad, T1, T2...]
        p_t = torch.cat([torch.full((pad_len,), PAD_ID, dtype=torch.long), t])
        # Labels: [-100, ..., -100, L1, L2...]
        p_l = torch.cat([torch.full((pad_len,), -100, dtype=torch.long), l])
        # Mask: [0, ..., 0, 1, 1...]
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
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            text_input_ids=inputs["text_input_ids"],
            audio_features=inputs["audio_features"], # 名字统一
            attention_mask=inputs["attention_mask"],
            audio_lens=inputs["audio_lens"],
            labels=inputs["labels"],
            task_modes=inputs["task_modes"]
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        覆盖默认优化器，为 Projector 和 Head 设置特定的学习率。
        """
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            
            # 定义特定组件的参数组
            projector_parameters = []
            
            # 1. 区分 Projector/Head 和其他参数 (LLM/LoRA)
            # 注意：model.input_proj 和 model.output_head 是 QwenCALM 的成员
            head_names = ["input_proj", "output_head"]
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # 检查是否属于 Projector 或 Head
                is_head = any(n in name for n in head_names)
                
                if is_head:
                    projector_parameters.append(param)
                else:
                    # 属于 LLM (LoRA) 的参数
                    if "bias" in name or "LayerNorm" in name:
                        no_decay_parameters.append(param)
                    else:
                        decay_parameters.append(param)

            # 2. 设置参数组
            # 假设: 
            # - args.learning_rate 是给 LoRA 的 (e.g. 2e-4)
            # - 我们希望 Projector 也是这个速度，或者更大 (e.g. 1e-3)
            # 这里简单起见，让 Projector 跟随主 LR，或者你可以手动乘个系数
            
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
                    "lr": self.args.learning_rate * 10, # 示例：给 Projector 10倍 LR (如果主 LR 很小)
                    # 或者直接用 self.args.learning_rate，取决于你设置的基础 LR 是多少
                    # 只要确保它足够大 (通常 >= 1e-4) 即可
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
    
    # 1. Tokenizer (设置 Left Padding)
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
        use_latents=model_args.use_precomputed_latents
    )

    eval_dir = data_args.eval_mel_dir if data_args.eval_mel_dir else data_args.mel_dir
    eval_dataset = CalmDataset(
        data_dir=eval_dir,
        librispeech_root=data_args.librispeech_root,
        subsets=data_args.eval_subsets,
        tokenizer=tokenizer,
        max_text_len=data_args.max_text_len,
        max_audio_len=data_args.max_audio_len,
        use_latents=model_args.use_precomputed_latents
    )
    
    trainer = CalmTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()