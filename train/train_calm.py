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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
    latent_dim: int = 64
    noise_size: int = field(default=64, metadata={"help": "Dimension of the noise vector for Energy Head"})
    num_mlp_layers: int = field(default=2, metadata={"help": "Number of layers in the MLP Generator"})
    num_samples: int = field(default=8, metadata={"help": "Number of samples for Energy Loss calculation"})
    beta: float = field(default=0.25, metadata={"help": "Beta parameter for Energy distance"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature during training"})
    
@dataclass
class DataArguments:
    librispeech_root: str = field(metadata={"help": "Path to original LibriSpeech dataset root"})
    mel_dir: str = field(metadata={"help": "Path to data directory (Mel .pt or Latent .pt)"})
    eval_mel_dir: str = field(default=None, metadata={"help": "Path to EVAL data directory. If None, use mel_dir"})
    train_subsets: str = field(default="train-clean-100,train-clean-360,train-other-500")
    eval_subsets: str = field(default="dev-clean")
    max_text_len: int = field(default=256)
    max_audio_len: int = field(default=512, metadata={"help": "Max Mel frames. If using latents, this will be scaled down by 16 automatically inside dataset logic."})
    latent_downsample: int = field(default=4)

class CalmDataset(Dataset):
    def __init__(self, data_dir, librispeech_root, subsets, tokenizer, max_text_len=256, max_audio_len=2048, use_latents=False, latent_downsample=16, is_eval=False):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.use_latents = use_latents
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eod_id
        self.latent_downsample = latent_downsample
        self.is_eval = is_eval
        
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
        
        # 3. RAM Cache Preloading
        print("Pre-loading all latents into RAM (approx 3-4GB)...")
        self.cached_latents = {}
        
        def load_file(idx_item_tuple):
            i, item = idx_item_tuple
            try:
                # Load to CPU, float16 to save RAM
                tensor = torch.load(item['file_path'], map_location="cpu").to(torch.float16)
                return item['file_path'], tensor
            except Exception as e:
                return None, None

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(load_file, (i, item)) for i, item in enumerate(self.data)]
            for future in tqdm(as_completed(futures), total=len(self.data), desc="Loading Latents"):
                path, tensor = future.result()
                if path is not None:
                    self.cached_latents[path] = tensor
        print(f"Cached {len(self.cached_latents)} latent files.")

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
            
            if self.is_eval:
                task_mode = "tts" if idx % 2 == 0 else "asr"
            else:
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

            # === [FIXED] Audio Loading (Check Cache First) ===
            path = item['file_path']
            if path in self.cached_latents:
                audio = self.cached_latents[path].float() # Convert back to float32
            else:
                audio = torch.load(path, map_location="cpu")
            
            # Calculate Target Length
            if self.use_latents:
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
        return {}

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
        # Initialize accumulators
        self.meter_tts_loss = 0.0
        self.meter_tts_count = 0
        self.meter_asr_loss = 0.0
        self.meter_asr_count = 0
        self.meter_div_loss = 0.0
        self.meter_fid_loss = 0.0
        
        self.eval_meters = {}
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return loss, logits, labels

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            text_input_ids=inputs["text_input_ids"],
            audio_features=inputs["audio_features"],
            attention_mask=inputs["attention_mask"],
            audio_lens=inputs["audio_lens"],
            labels=inputs["labels"],
            task_modes=inputs["task_modes"]
        )
        
        # Access attributes safely from CalmModelOutput
        loss = outputs.loss
        
        # === [FIXED] Accumulate metrics here, DO NOT call self.log() directly ===
        if self.model.training:
            # We access the raw tensors. For proper DDP average, the custom log() below handles reduce.
            # Here we just accumulate local sums.
            task_modes = inputs.get("task_modes", [])
            num_tts = task_modes.count("tts")
            num_asr = task_modes.count("asr")
            
            # Using getattr to be safe
            loss_tts = getattr(outputs, "loss_tts", torch.tensor(0.0)).item()
            loss_asr = getattr(outputs, "loss_asr", torch.tensor(0.0)).item()
            loss_div = getattr(outputs, "loss_diversity", torch.tensor(0.0)).item()
            loss_fid = getattr(outputs, "loss_fidelity", torch.tensor(0.0)).item()

            self.meter_tts_loss += loss_tts * num_tts
            self.meter_tts_count += num_tts
            self.meter_asr_loss += loss_asr * num_asr
            self.meter_asr_count += num_asr
            self.meter_div_loss += loss_div * num_tts
            self.meter_fid_loss += loss_fid * num_tts
        else:
            # Evaluation accumulation
            task_modes = inputs.get("task_modes", [])
            num_tts = task_modes.count("tts")
            num_asr = task_modes.count("asr")
            
            loss_tts = getattr(outputs, "loss_tts", torch.tensor(0.0)).item()
            loss_asr = getattr(outputs, "loss_asr", torch.tensor(0.0)).item()
            loss_div = getattr(outputs, "loss_diversity", torch.tensor(0.0)).item()
            loss_fid = getattr(outputs, "loss_fidelity", torch.tensor(0.0)).item()

            if "eval_tts_loss" not in self.eval_meters:
                self.eval_meters = {
                    "eval_tts_loss": 0.0, "eval_tts_count": 0,
                    "eval_asr_loss": 0.0, "eval_asr_count": 0,
                    "eval_div_loss": 0.0, "eval_fid_loss": 0.0
                }
            
            self.eval_meters["eval_tts_loss"] += loss_tts * num_tts
            self.eval_meters["eval_tts_count"] += num_tts
            self.eval_meters["eval_asr_loss"] += loss_asr * num_asr
            self.eval_meters["eval_asr_count"] += num_asr
            self.eval_meters["eval_div_loss"] += loss_div * num_tts
            self.eval_meters["eval_fid_loss"] += loss_fid * num_tts

        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.eval_meters = {}
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if "eval_tts_count" in self.eval_meters and self.eval_meters["eval_tts_count"] > 0:
            count = self.eval_meters["eval_tts_count"]
            output[f"{metric_key_prefix}_tts_loss"] = self.eval_meters["eval_tts_loss"] / count
            output[f"{metric_key_prefix}_div_loss"] = self.eval_meters["eval_div_loss"] / count
            output[f"{metric_key_prefix}_fid_loss"] = self.eval_meters["eval_fid_loss"] / count
        
        if "eval_asr_count" in self.eval_meters and self.eval_meters["eval_asr_count"] > 0:
            count = self.eval_meters["eval_asr_count"]
            output[f"{metric_key_prefix}_asr_loss"] = self.eval_meters["eval_asr_loss"] / count

        self.log(output)
        return output
    
    def log(self, logs: Dict[str, float]) -> None:
        # Calculate averages from accumulators
        if self.args.world_size > 1:
            metrics = torch.tensor([
                self.meter_tts_loss, float(self.meter_tts_count),
                self.meter_asr_loss, float(self.meter_asr_count),
                self.meter_div_loss, self.meter_fid_loss
            ], device=self.args.device)
            
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            tts_sum, tts_cnt = metrics[0].item(), metrics[1].item()
            asr_sum, asr_cnt = metrics[2].item(), metrics[3].item()
            div_sum, fid_sum = metrics[4].item(), metrics[5].item()
        else:
            tts_sum, tts_cnt = self.meter_tts_loss, self.meter_tts_count
            asr_sum, asr_cnt = self.meter_asr_loss, self.meter_asr_count
            div_sum, fid_sum = self.meter_div_loss, self.meter_fid_loss

        logs["tts_loss"] = round(tts_sum / tts_cnt, 4) if tts_cnt > 0 else 0.0
        logs["asr_loss"] = round(asr_sum / asr_cnt, 4) if asr_cnt > 0 else 0.0
        logs["diversity_loss"] = round(div_sum / tts_cnt, 4) if tts_cnt > 0 else 0.0
        logs["fidelity_loss"] = round(fid_sum / tts_cnt, 4) if tts_cnt > 0 else 0.0
        
        # Reset accumulators after logging
        self.meter_tts_loss = 0.0
        self.meter_tts_count = 0
        self.meter_asr_loss = 0.0
        self.meter_asr_count = 0
        self.meter_div_loss = 0.0
        self.meter_fid_loss = 0.0

        super().log(logs)

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            projector_parameters = []
            head_names = ["input_proj", "output_head"]
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad: continue
                
                is_head = any(n in name for n in head_names)
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
                {"params": projector_parameters, "weight_decay": self.args.weight_decay, "lr": 20 * self.args.learning_rate},
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
        training_args.ddp_find_unused_parameters = True
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.padding_side = 'left' 

    # 2. Config & Model
    calm_config = QwenCALMConfig(
        qwen_path=model_args.qwen_path,
        vae_path=model_args.vae_path,
        use_precomputed_latents=model_args.use_precomputed_latents,
        latent_dim=model_args.latent_dim,
        noise_size=model_args.noise_size,
        num_mlp_layers=model_args.num_mlp_layers,
        num_samples=model_args.num_samples,
        beta=model_args.beta,
        temperature=model_args.temperature
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
        latent_downsample=data_args.latent_downsample,
        is_eval=False
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
        latent_downsample=data_args.latent_downsample,
        is_eval=True
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