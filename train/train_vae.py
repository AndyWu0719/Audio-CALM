import os
import sys
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from glob import glob

sys.path.append(os.getcwd()) 

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from models.modeling_vae import AcousticVAE, AudioVAEConfig

@dataclass
class ModelArguments:
    hidden_channels: int = field(default=512, metadata={"help": "Hidden channels in VAE"})
    latent_channels: int = field(default=64, metadata={"help": "Latent dimension size"})
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2], metadata={"help": "List of strides for compression"})
    kl_weight: float = field(default=0.0001, metadata={"help": "KL divergence weight"})
    norm_num_groups: int = field(default=32, metadata={"help": "Group Norm groups"})

@dataclass
class DataArguments:
    data_dir: str = field(default="./data/mel_features", metadata={"help": "Root path to processed .pt files"})
    train_subsets: str = field(default="train-clean-100,train-clean-360,train-other-500", metadata={"help": "Subdirectories to use for training"})
    eval_subsets: str = field(default="dev-clean", metadata={"help": "Subdirectories to use for evaluation"})
    crop_size: int = field(default=256, metadata={"help": "Crop size (frames) for training"})


class MelDataset(Dataset):
    def __init__(self, data_dir, subsets=None, crop_size=256, is_eval=False):
        self.files = []
        self.is_eval = is_eval
            
        if subsets:
            subset_list = subsets.split(",")
            for subset in subset_list:
                subset_path = os.path.join(data_dir, subset.strip())
                subset_files = glob(os.path.join(subset_path, "*.pt"))
                self.files.extend(subset_files)
                print(f"Loaded {len(subset_files)} files from {subset}")
        else:
            self.files = glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}. Check your path and subsets.")
            
        self.crop_size = crop_size
        print(f"Total Dataset loaded: {len(self.files)} files. Crop size: {crop_size}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            # Load: [80, T]
            mel = torch.load(self.files[idx], map_location="cpu")
            
            # Random Crop
            if mel.shape[1] > self.crop_size:
                if self.is_eval:
                    # Center Crop
                    start = (mel.shape[1] - self.crop_size) // 2
                else:
                    # Random Crop
                    start = torch.randint(0, mel.shape[1] - self.crop_size, (1,)).item()
                mel = mel[:, start:start+self.crop_size]
            else:
                # Pad if too short
                pad_len = self.crop_size - mel.shape[1]
                mel = torch.nn.functional.pad(mel, (0, pad_len))
            
            return {"input_mel": mel}
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return {"input_mel": torch.randn(80, self.crop_size)}

# === 3. Collator & Trainer ===
def data_collator(features):
    batch_mels = [f["input_mel"] for f in features]
    batch_mels = torch.stack(batch_mels) # [B, 80, T]
    return {
        "mel": batch_mels, 
        "labels": batch_mels 
    }

class VAETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["mel"])
        loss = outputs["loss"]
        
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "rec_loss": outputs["rec_loss"].item(),
                "kl_loss": outputs["kl_loss"].item()
            })
            
        return (loss, outputs) if return_outputs else loss

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)
    
    print(f"Initializing VAE with strides: {model_args.strides}, latent: {model_args.latent_channels}")
    config = AudioVAEConfig(
        hidden_channels=model_args.hidden_channels, 
        latent_channels=model_args.latent_channels,
        strides=model_args.strides,
        kl_weight=model_args.kl_weight
    )
    model = AcousticVAE(config)
    
    total_stride = 1
    for s in model_args.strides: total_stride *= s
    print(f"Total Compression Rate: {total_stride}x")
    
    print(f"Loading Training Data from: {data_args.train_subsets}")
    train_dataset = MelDataset(
        data_dir=data_args.data_dir, 
        subsets=data_args.train_subsets,
        crop_size=data_args.crop_size,
        is_eval=False
    )

    print(f"Loading Evaluation Data from: {data_args.eval_subsets}")
    eval_dataset = MelDataset(
        data_dir=data_args.data_dir, 
        subsets=data_args.eval_subsets,
        crop_size=data_args.crop_size,
        is_eval=True
    )
    
    trainer = VAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    if training_args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()