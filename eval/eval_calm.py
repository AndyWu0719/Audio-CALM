"""
Unified Evaluation Script for Audio-CALM.
VERSION: FULL_MODAL (Supports both TTS and ASR)
"""

import os
import sys
import json
import csv
import logging
import random
import torch
import types
import torchaudio
import soundfile as sf
import hydra
import wandb
import numpy as np
import gradio as gr
from tqdm import tqdm
from typing import List, Dict, Optional
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import PeftModel
import evaluate
from rich.logging import RichHandler
from rich.console import Console

# --- Environment Patches ---
if not hasattr(torchaudio, "list_audio_backends"):
    try:
        import torchaudio.backend
        torchaudio.list_audio_backends = getattr(torchaudio.backend, "list_audio_backends", lambda: ["soundfile"])
    except ImportError:
        torchaudio.list_audio_backends = lambda: []

sys.path.append(os.getcwd())
from models.modeling_calm import QwenCALM, QwenCALMConfig

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, show_path=False)])
logger = logging.getLogger("eval")
console = Console()

try:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()
except ImportError:
    normalizer = lambda x: x.lower()

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
        except:
            logger.warning("⚠️ HiFi-GAN not found. Will use GL.")

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

        # Fix Chipmunk effect
        mel = torch.nn.functional.interpolate(
            mel, scale_factor=2.0, mode='linear', align_corners=False
        )

        if self.hifi is not None:
            mel_log10 = mel * 0.43429
            try: return self.hifi.decode_batch(mel_log10.transpose(1, 2)).squeeze(1)
            except: 
                try: return self.hifi.decode_batch(mel_log10).squeeze(1)
                except: pass

        energy_mel = torch.exp(mel) 
        linear_energy = torch.matmul(energy_mel.transpose(1, 2), self.inverse_mel_basis).transpose(1, 2)
        linear_mag = torch.sqrt(torch.clamp(linear_energy, min=1e-8))
        wav = self.griffin_lim(linear_mag)
        peak = torch.max(torch.abs(wav))
        if peak > 1.0: wav = wav / peak
        return wav.squeeze(1)

# ==============================================================================
# 2. Monkey Patch (Fixes Position ID)
# ==============================================================================
def patched_projector_forward(self, x, offset=0):
    B, T, _ = x.shape
    device = x.device
    x = x.transpose(1, 2)
    x = self.conv_block(x)
    x = x.transpose(1, 2)
    start = offset
    end = offset + T
    pos_ids = torch.arange(start, end, device=device).clamp(max=self.max_audio_len - 1).unsqueeze(0).expand(B, -1)
    pos_emb_val = self.pos_emb(pos_ids)
    x = x + pos_emb_val
    for block in self.blocks:
        x = x + block(x)
    return self.post_norm(x)

# ==============================================================================
# 3. Model Loading
# ==============================================================================
def load_model(cfg, device):
    logger.info(f"🤖 Loading Model: {cfg.model.qwen_path}")
    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path, vae_path=cfg.model.vae_path, latent_dim=cfg.model.latent_dim,
        flow_hidden_dim=cfg.model.get("flow_hidden_dim", 2048), flow_num_layers=cfg.model.get("flow_num_layers", 4),
        use_precomputed_latents=False 
    )
    model = QwenCALM(config)
    
    ckpt = cfg.evaluation.checkpoint_path
    
    # Adapter Loading
    if os.path.exists(os.path.join(ckpt, "asr")) or os.path.exists(os.path.join(ckpt, "tts")):
        if os.path.exists(os.path.join(ckpt, "asr")):
            model.llm = PeftModel.from_pretrained(model.llm, os.path.join(ckpt, "asr"), adapter_name="asr")
        if os.path.exists(os.path.join(ckpt, "tts")):
            if isinstance(model.llm, PeftModel):
                try: model.llm.load_adapter(os.path.join(ckpt, "tts"), adapter_name="tts")
                except: pass
            else:
                model.llm = PeftModel.from_pretrained(model.llm, os.path.join(ckpt, "tts"), adapter_name="tts")
    else:
        if os.path.exists(os.path.join(ckpt, "adapter_config.json")):
            model.llm = PeftModel.from_pretrained(model.llm, ckpt)

    for component in ["input_proj", "output_head"]:
        bin_path = os.path.join(ckpt, f"{component}.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            getattr(model, component).load_state_dict(state_dict)

    logger.info("🩹 Applying Monkey Patch...")
    model.input_proj.forward = types.MethodType(patched_projector_forward, model.input_proj)

    model.to(device).eval()
    if cfg.training.get("bf16", False): 
        model.to(torch.bfloat16)
        model.vae.to(torch.float32) 
    return model

# ==============================================================================
# 4. ASR Inference Logic (NEW!)
# ==============================================================================
@torch.no_grad()
def run_asr_inference(model, tokenizer, latent_path, device):
    # 1. 切换 ASR Adapter
    if hasattr(model.llm, "set_adapter") and hasattr(model.llm, "peft_config"):
        if "asr" in model.llm.peft_config:
            model.llm.set_adapter("asr")

    # 2. 处理 Audio Latent
    if not os.path.exists(latent_path): return ""
    payload = torch.load(latent_path, map_location="cpu")
    audio = payload.get("latent", payload) if isinstance(payload, dict) else payload
    if audio.dim() == 2:
        if audio.shape[0] == 64: audio = audio.transpose(0, 1) # [T, 64]
        audio = audio.unsqueeze(0) # [1, T, 64]
    
    audio = audio.to(device).to(model.llm.dtype)
    
    # 编码 Audio (offset=0)
    audio_embeds = model.input_proj(audio, offset=0) 

    # 3. 构造 Prompt
    # 保持简单，与训练一致
    prompt = "Transcribe the audio content into text."
    prefix_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prefix_embeds = model.get_input_embeddings()(prefix_ids)

    # 4. 拼接: [Audio] + [Text]
    inputs_embeds = torch.cat([audio_embeds, prefix_embeds], dim=1)

    # 5. 生成 (使用 Beam Search !!!)
    outputs = model.llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=256,
        
        # === [关键修改] 开启 Beam Search ===
        num_beams=5,          # 刷榜标配，比 Greedy 准很多
        do_sample=False,      # 确定性生成
        
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0 # ASR 不建议惩罚重复，因为有时候真的会重复词
    )
    
    # 6. 解码
    transcription = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return transcription

def eval_task_asr(cfg, model, tokenizer, data):
    console.print("[bold green]>>> Running ASR Evaluation (Beam=5, HF Norm)[/bold green]")
    
    # 尝试加载官方 Normalizer
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        normalizer = BasicTextNormalizer() 
    except ImportError:
        logger.warning("Transformers Normalizer not found. Falling back to simple regex.")
        import re
        def normalizer(text):
            text = text.lower()
            text = re.sub(r"[^a-z0-9\s]", "", text) 
            return " ".join(text.split())

    csv_file = open(os.path.join(cfg.evaluation.output_dir, "asr_results.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "wer", "text_ref", "text_pred", "norm_ref", "norm_pred"])

    wers = []
    
    for i, item in enumerate(tqdm(data, desc="ASR Decoding")):
        text_ref = item.get("text", "")
        latent_path = item.get("audio") or item.get("latent_path") or item.get("file_path")
        
        if not latent_path: continue
        
        try:
            text_pred = run_asr_inference(model, tokenizer, latent_path, cfg.device)
            
            # === [HF 标准化] ===
            # 去除标点、转小写、标准化数字等
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
                console.print(f"[Sample {i+1}] Avg WER: {avg_wer:.2%}")
                # 打印一下看看对齐效果
                # console.print(f"   Ref: {norm_ref[:40]}")
                # console.print(f"   Pred:{norm_pred[:40]}")
                
        except Exception as e:
            logger.error(f"Error sample {i}: {e}")

    csv_file.close()
    if len(wers) > 0:
        final_wer = sum(wers) / len(wers)
        console.print(f"[bold blue]✅ Final WER (HF Standard): {final_wer:.2%}[/bold blue]")

# ==============================================================================
# 5. TTS Logic (Original)
# ==============================================================================
def generate_one_step_flow(model, condition, steps, cfg_scale, device):
    noise = torch.randn(1, 1, model.config.latent_dim, device=device, dtype=model.llm.dtype)
    dt = 1.0 / steps
    x = noise
    for i in range(steps):
        t = torch.full((1,), i/steps, device=device, dtype=x.dtype)
        v_cond = model.output_head(condition, x, t)
        v_uncond = model.output_head(torch.zeros_like(condition), x, t)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + v * dt
    return x

@torch.no_grad()
def run_tts_inference(model, tokenizer, vocoder, text, gt_latent_path=None, steps=10, cfg_scale=1.0, device="cuda"):
    if hasattr(model.llm, "set_adapter") and hasattr(model.llm, "peft_config"):
        if "tts" in model.llm.peft_config:
            model.llm.set_adapter("tts")

    prompt = f"Read this text:\n{text}"
    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    text_embeds = model.get_input_embeddings()(text_ids)
    
    out = model.llm(inputs_embeds=text_embeds, use_cache=True, past_key_values=None, output_hidden_states=True)
    past_kv = out.past_key_values
    condition = out.hidden_states[-1][:, -1:, :] 
    
    curr_latent = generate_one_step_flow(model, condition, steps, cfg_scale, device)
    history_latents = [curr_latent]

    pbar = tqdm(range(300), desc="Gen Audio", leave=False)
    for i in pbar:
        input_latent = curr_latent
        curr_embeds = model.input_proj(input_latent, offset=i)
        out = model.llm(inputs_embeds=curr_embeds, use_cache=True, past_key_values=past_kv, output_hidden_states=True)
        past_kv = out.past_key_values
        condition = out.hidden_states[-1][:, -1:, :]
        curr_latent = generate_one_step_flow(model, condition, steps, cfg_scale, device)
        history_latents.append(curr_latent)

    latents = torch.cat(history_latents, dim=1).transpose(1, 2).to(torch.float32)
    mel = model.vae.decode(latents)
    wav = vocoder.decode(mel)
    return wav.cpu()

def eval_task_tts(cfg, model, tokenizer, vocoder, data):
    wav_dir = os.path.join(cfg.evaluation.output_dir, "generated_wavs")
    os.makedirs(wav_dir, exist_ok=True)
    csv_file = open(os.path.join(cfg.evaluation.output_dir, "tts_results.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "text", "wav_path"])

    for i, item in enumerate(data):
        text = item.get("text", "")
        if not text: continue
        try:
            scale = cfg.evaluation.get("cfg_scale", 1.0)
            wav = run_tts_inference(
                model, tokenizer, vocoder, text, 
                steps=cfg.evaluation.flow_steps, 
                cfg_scale=scale, 
                device=cfg.device
            )
            wav_np = wav.squeeze().numpy()
            save_path = os.path.join(wav_dir, f"sample_{i}.wav")
            sf.write(save_path, wav_np, 16000)
            writer.writerow([i, text, save_path])
            if (i+1)%10==0: console.print(f"[Sample {i+1}] Done.")
        except Exception as e:
            logger.error(f"Error sample {i}: {e}")

    csv_file.close()

@hydra.main(version_base=None, config_path="../config", config_name="calm_config")
def main(cfg: DictConfig):
    set_seed(cfg.evaluation.get("seed", 42))
    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)
    with open_dict(cfg): cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not cfg.evaluation.get("web_demo", False):
        wandb.init(project=cfg.evaluation.get("wandb_project", "Audio-CALM-Eval"), config=OmegaConf.to_container(cfg))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    model = load_model(cfg, cfg.device)
    
    data = load_dataset_jsonl(cfg.evaluation.test_file, cfg.evaluation.max_samples)
    
    # 核心分流逻辑
    task = cfg.evaluation.task.lower()
    if task == "tts":
        vocoder = Vocoder(cfg.device)
        eval_task_tts(cfg, model, tokenizer, vocoder, data)
    elif task == "asr":
        eval_task_asr(cfg, model, tokenizer, data)
    elif task == "mix":
        vocoder = Vocoder(cfg.device)
        eval_task_tts(cfg, model, tokenizer, vocoder, data)
        eval_task_asr(cfg, model, tokenizer, data)

if __name__ == "__main__":
    main()