"""
Unified Evaluation Script for Audio-CALM (v2 Refactored).
Supports: ASR (Speech-to-Text) and TTS (Text-to-Speech)
"""

import os
import sys
import json
import csv
import logging
import random
import torch
import torchaudio
import soundfile as sf
import hydra
import wandb
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import PeftModel
import evaluate
from rich.logging import RichHandler
from rich.console import Console

# --- Environment Patches ---
# Fix for torchaudio backend issues on some environments
if not hasattr(torchaudio, "list_audio_backends"):
    try:
        import torchaudio.backend
        torchaudio.list_audio_backends = getattr(torchaudio.backend, "list_audio_backends", lambda: ["soundfile"])
    except ImportError:
        torchaudio.list_audio_backends = lambda: []

sys.path.append(os.getcwd())
# Ensure modeling_calm.py is in the models/ folder
from models.modeling_calm import QwenCALM, QwenCALMConfig

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, show_path=False)])
logger = logging.getLogger("eval")
console = Console()

# Metric setup
wer_metric = evaluate.load("wer")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset_jsonl(path: str, max_samples: int = -1) -> List[Dict]:
    if not os.path.exists(path):
        logger.error(f"Dataset not found: {path}")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    if max_samples > 0 and max_samples < len(data):
        logger.info(f"Subsampling {max_samples} from {len(data)} total samples.")
        random.shuffle(data)
        data = data[:max_samples]
    return data

# ==============================================================================
# 1. Vocoder (With Interpolation for TTS)
# ==============================================================================
class Vocoder:
    def __init__(self, device="cuda"):
        self.device = device
        logger.info("🔧 Initializing Vocoder...")
        self.hifi = None
        try:
            from speechbrain.inference.vocoders import HIFIGAN
            self.hifi = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-libritts-16kHz",
                savedir="tmp_hifigan",
                run_opts={"device": device}
            )
            logger.info("✅ SpeechBrain HiFi-GAN loaded.")
        except Exception as e:
            logger.warning(f"⚠️ HiFi-GAN not found ({e}). Will use Griffin-Lim.")

        self.n_fft = 1024
        self.n_mels = 80
        self.sample_rate = 16000
        self.hop_length = 256
        self.win_length = 1024
        
        mel_fb = torchaudio.transforms.MelScale(
            n_mels=self.n_mels, sample_rate=self.sample_rate,
            f_min=0, f_max=8000, n_stft=self.n_fft // 2 + 1,
            norm="slaney", mel_scale="slaney"
        ).to(device).fb 
        self.inverse_mel_basis = torch.linalg.pinv(mel_fb).to(device)
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft, n_iter=60, win_length=self.win_length,
            hop_length=self.hop_length, power=1.0
        ).to(device)

    def decode(self, mel):
        mel = mel.to(self.device).to(torch.float32)
        if mel.dim() == 2: mel = mel.unsqueeze(0)
        if mel.shape[-1] == 80: mel = mel.transpose(1, 2)

        # Interpolate to fix potential temporal resolution mismatch (Chipmunk effect)
        mel = torch.nn.functional.interpolate(
            mel, scale_factor=2.0, mode='linear', align_corners=False
        )

        if self.hifi is not None:
            mel_log10 = mel * 0.43429
            try: return self.hifi.decode_batch(mel_log10.transpose(1, 2)).squeeze(1)
            except: 
                try: return self.hifi.decode_batch(mel_log10).squeeze(1)
                except: pass

        # Fallback to Griffin-Lim
        energy_mel = torch.exp(mel) 
        linear_energy = torch.matmul(energy_mel.transpose(1, 2), self.inverse_mel_basis).transpose(1, 2)
        linear_mag = torch.sqrt(torch.clamp(linear_energy, min=1e-8))
        wav = self.griffin_lim(linear_mag)
        
        # Peak Normalization
        peak = torch.max(torch.abs(wav))
        if peak > 1.0: wav = wav / peak
        return wav.squeeze(1)

# ==============================================================================
# 2. Model Loading (Fixed for Native SOA Support)
# ==============================================================================
def load_model(cfg, device):
    logger.info(f"🤖 Loading Model Base: {cfg.model.qwen_path}")
    
    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path, 
        vae_path=cfg.model.vae_path, 
        latent_dim=cfg.model.latent_dim,
        flow_hidden_dim=cfg.model.get("flow_hidden_dim", 2048), 
        flow_num_layers=cfg.model.get("flow_num_layers", 4),
        use_precomputed_latents=False 
    )
    
    # Initialize model structure
    model = QwenCALM(config)
    
    ckpt_dir = cfg.evaluation.checkpoint_path
    logger.info(f"📂 Loading Checkpoints from: {ckpt_dir}")

    # 1. Load LLM Adapters (LoRA)
    if os.path.exists(os.path.join(ckpt_dir, "asr")) or os.path.exists(os.path.join(ckpt_dir, "tts")):
        if os.path.exists(os.path.join(ckpt_dir, "asr")):
            logger.info("  - Loading ASR LoRA...")
            model.llm = PeftModel.from_pretrained(model.llm, os.path.join(ckpt_dir, "asr"), adapter_name="asr")
        if os.path.exists(os.path.join(ckpt_dir, "tts")):
            logger.info("  - Loading TTS LoRA...")
            if isinstance(model.llm, PeftModel):
                try: model.llm.load_adapter(os.path.join(ckpt_dir, "tts"), adapter_name="tts")
                except: pass
            else:
                model.llm = PeftModel.from_pretrained(model.llm, os.path.join(ckpt_dir, "tts"), adapter_name="tts")
    else:
        # Fallback: single adapter at root
        if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
            logger.info("  - Loading Single LoRA...")
            model.llm = PeftModel.from_pretrained(model.llm, ckpt_dir)

    # 2. Load Projectors (Input/Output/SOA)
    # Load input_proj and output_head
    for component in ["input_proj", "output_head"]:
        bin_path = os.path.join(ckpt_dir, f"{component}.bin")
        if os.path.exists(bin_path):
            logger.info(f"  - Loading {component}...")
            state_dict = torch.load(bin_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            getattr(model, component).load_state_dict(state_dict)
        else:
            logger.warning(f"  ⚠️  {component}.bin not found! Model may not work.")

    # [FIX] Load soa_embed explicitly
    soa_path = os.path.join(ckpt_dir, "soa_embed.bin")
    if os.path.exists(soa_path):
        logger.info(f"  - Loading soa_embed...")
        soa_data = torch.load(soa_path, map_location="cpu")
        model.soa_embed.data = soa_data # 假设这里加载成功
    
        # [建议新增] 确保 dtype 一致
        if cfg.training.get("bf16", False) and soa_data.dtype != torch.bfloat16:
             model.soa_embed.data = model.soa_embed.data.to(torch.bfloat16)
            
        # Handle different saving formats (dict vs raw tensor)
        if isinstance(soa_data, dict):
            # Try common keys or take the first value
            key = next((k for k in ["weight", "soa_embed"] if k in soa_data), None)
            if key:
                model.soa_embed.data = soa_data[key]
            else:
                model.soa_embed.data = list(soa_data.values())[0]
        else:
            model.soa_embed.data = soa_data
    else:
        logger.warning(f"  ⚠️  soa_embed.bin not found! TTS will produce noise.")

    model.to(device).eval()
    
    # Mixed Precision Setup
    if cfg.training.get("bf16", False): 
        logger.info("  - Converting to bfloat16 (VAE remains fp32)")
        model.to(torch.bfloat16)
        model.vae.to(torch.float32) # VAE usually needs FP32 stability
        
    return model

# ==============================================================================
# 3. ASR Inference Logic
# ==============================================================================
@torch.no_grad()
def run_asr_inference(model, tokenizer, latent_path, device):
    # Switch Adapter
    if hasattr(model.llm, "set_adapter") and hasattr(model.llm, "peft_config"):
        if "asr" in model.llm.peft_config:
            model.llm.set_adapter("asr")

    # Load Audio Latent
    if not os.path.exists(latent_path): return ""
    payload = torch.load(latent_path, map_location="cpu")
    audio = payload.get("latent", payload) if isinstance(payload, dict) else payload
    
    # Shape check
    if audio.dim() == 2:
        if audio.shape[0] == 64: audio = audio.transpose(0, 1) # Ensure [T, D]
        audio = audio.unsqueeze(0) # [1, T, D]
    
    audio = audio.to(device).to(model.llm.dtype)
    
    # Encode Audio (offset=0 implies start of sequence)
    audio_embeds = model.input_proj(audio, offset=0) 

    # Construct Prompt
    prompt = "Transcribe the audio content into text."
    prefix_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prefix_embeds = model.get_input_embeddings()(prefix_ids)

    # Concat: [Audio] + [Text Prompt]
    inputs_embeds = torch.cat([audio_embeds, prefix_embeds], dim=1)

    # Generate
    outputs = model.llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=256,
        num_beams=5,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0 
    )
    
    transcription = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return transcription

def eval_task_asr(cfg, model, tokenizer, data):
    console.print("[bold green]>>> Running ASR Evaluation (Beam=5)[/bold green]")
    
    # Normalizer Setup
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        normalizer = BasicTextNormalizer() 
    except ImportError:
        import re
        def normalizer(text): return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    out_path = os.path.join(cfg.evaluation.output_dir, "asr_results.csv")
    csv_file = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "wer", "text_ref", "text_pred", "norm_ref", "norm_pred"])

    wers = []
    
    for i, item in enumerate(tqdm(data, desc="ASR Decoding")):
        text_ref = item.get("text", "")
        # Support various dataset keys
        latent_path = item.get("audio") or item.get("latent_path") or item.get("file_path")
        
        if not latent_path: continue
        
        try:
            text_pred = run_asr_inference(model, tokenizer, latent_path, cfg.device)
            norm_ref = normalizer(text_ref)
            norm_pred = normalizer(text_pred)
            
            if len(norm_ref) == 0:
                wer = 0.0 if len(norm_pred) == 0 else 1.0
            else:
                wer = wer_metric.compute(predictions=[norm_pred], references=[norm_ref])
            
            wers.append(wer)
            writer.writerow([i, f"{wer:.4f}", text_ref, text_pred, norm_ref, norm_pred])
            
            if (i+1) % 10 == 0:
                avg_wer = sum(wers) / len(wers)
                console.print(f"[Sample {i+1}] Current Avg WER: {avg_wer:.2%}")
                
        except Exception as e:
            logger.error(f"Error sample {i}: {e}")

    csv_file.close()
    if len(wers) > 0:
        final_wer = sum(wers) / len(wers)
        console.print(f"[bold blue]✅ Final WER: {final_wer:.2%}[/bold blue]")

# ==============================================================================
# 4. TTS Logic (Flow Matching)
# ==============================================================================
def generate_one_step_flow(model, condition, steps, cfg_scale, device):
    """
    Performs Flow Matching ODE integration to generate one frame of latent audio.
    """
    # Initialize noise
    noise = torch.randn(1, 1, model.config.latent_dim, device=device, dtype=model.llm.dtype)
    dt = 1.0 / steps
    x = noise
    
    for i in range(steps):
        t = torch.full((1,), i/steps, device=device, dtype=x.dtype)
        
        # [FIXED HERE] -----------------------------------------
        # Definition: forward(condition, noisy_x, t)
        v_cond = model.output_head(condition, x, t)
        v_uncond = model.output_head(torch.zeros_like(condition), x, t)
        # ------------------------------------------------------
        
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + v * dt
        
    return x

@torch.no_grad()
def run_tts_inference(model, tokenizer, vocoder, text, steps=10, cfg_scale=1.0, device="cuda"):
    # Switch Adapter
    if hasattr(model.llm, "set_adapter") and hasattr(model.llm, "peft_config"):
        if "tts" in model.llm.peft_config:
            model.llm.set_adapter("tts")

    # 1. Prepare Text Prompt
    prompt = f"Read this text:\n{text}"
    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    text_embeds = model.get_input_embeddings()(text_ids)
    
    # 2. Append SOA Token (Condition for first audio frame)
    # model.soa_embed is [1, 1, Hidden]
    soa_token = model.soa_embed.expand(1, -1, -1) 
    inputs_embeds = torch.cat([text_embeds, soa_token], dim=1)
    
    # 3. Run LLM Prefill
    out = model.llm(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None, output_hidden_states=True)
    past_kv = out.past_key_values
    
    # The last hidden state corresponds to the SOA token -> Condition for Audio_0
    condition = out.hidden_states[-1][:, -1:, :] 
    
    # 4. Generate First Frame (Audio_0) via Flow
    curr_latent = generate_one_step_flow(model, condition, steps, cfg_scale, device)
    history_latents = [curr_latent]

    # 5. Autoregressive Loop
    # Limit max length to avoid infinite generation
    max_frames = 500 
    pbar = tqdm(range(max_frames), desc="Gen Audio", leave=False)
    
    for i in pbar:
        # Input to LLM: Current Audio Latent (projected)
        # Offset i corresponds to position i in the audio sequence
        input_latent = curr_latent
        curr_embeds = model.input_proj(input_latent, offset=i)
        
        # LLM Step
        out = model.llm(inputs_embeds=curr_embeds, use_cache=True, past_key_values=past_kv, output_hidden_states=True)
        past_kv = out.past_key_values
        
        # Get Condition for Next Frame
        condition = out.hidden_states[-1][:, -1:, :]
        
        # Stop Token Check (Optional: Detect silence or specific latent pattern)
        # For now, we rely on fixed length or basic checks (not implemented here)
        
        # Generate Next Frame via Flow
        curr_latent = generate_one_step_flow(model, condition, steps, cfg_scale, device)
        history_latents.append(curr_latent)

    # 6. Decode Latents to Waveform
    latents = torch.cat(history_latents, dim=1).transpose(1, 2).to(torch.float32) # [1, Latent, T]
    
    # VAE Decode
    mel = model.vae.decode(latents)
    
    # Vocoder
    wav = vocoder.decode(mel)
    return wav.cpu()

def eval_task_tts(cfg, model, tokenizer, vocoder, data):
    wav_dir = os.path.join(cfg.evaluation.output_dir, "generated_wavs")
    os.makedirs(wav_dir, exist_ok=True)
    csv_file = open(os.path.join(cfg.evaluation.output_dir, "tts_results.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "text", "wav_path"])
    
    console.print(f"[bold green]>>> Running TTS Evaluation (Steps={cfg.evaluation.flow_steps})[/bold green]")

    for i, item in enumerate(data):
        text = item.get("text", "")
        if not text: continue
        try:
            scale = cfg.evaluation.get("cfg_scale", 1.0)
            steps = cfg.evaluation.get("flow_steps", 10)
            
            wav = run_tts_inference(
                model, tokenizer, vocoder, text, 
                steps=steps, 
                cfg_scale=scale, 
                device=cfg.device
            )
            
            wav_np = wav.squeeze().numpy()
            save_path = os.path.join(wav_dir, f"sample_{i}.wav")
            sf.write(save_path, wav_np, 16000)
            
            writer.writerow([i, text, save_path])
            if (i+1) % 5 == 0: 
                console.print(f"[Sample {i+1}] Generated: {save_path}")
                
        except Exception as e:
            logger.error(f"Error sample {i}: {e}")

    csv_file.close()

# ==============================================================================
# Main Entry
# ==============================================================================
@hydra.main(version_base=None, config_path="../config", config_name="calm_config")
def main(cfg: DictConfig):
    set_seed(cfg.evaluation.get("seed", 42))
    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)
    
    with open_dict(cfg): 
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not cfg.evaluation.get("web_demo", False):
        wandb.init(project=cfg.evaluation.get("wandb_project", "Audio-CALM-Eval"), config=OmegaConf.to_container(cfg))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    
    # Load Model with Fixed Logic
    model = load_model(cfg, cfg.device)
    
    # Load Data
    data = load_dataset_jsonl(cfg.evaluation.test_file, cfg.evaluation.max_samples)
    
    # Dispatch Task
    task = cfg.evaluation.task.lower()
    if task == "tts":
        vocoder = Vocoder(cfg.device)
        eval_task_tts(cfg, model, tokenizer, vocoder, data)
    elif task == "asr":
        eval_task_asr(cfg, model, tokenizer, data)
    elif task == "mix":
        vocoder = Vocoder(cfg.device)
        console.rule("[bold]Starting TTS Evaluation[/bold]")
        eval_task_tts(cfg, model, tokenizer, vocoder, data)
        console.rule("[bold]Starting ASR Evaluation[/bold]")
        eval_task_asr(cfg, model, tokenizer, data)
    else:
        logger.error(f"Unknown task: {task}")

if __name__ == "__main__":
    main()