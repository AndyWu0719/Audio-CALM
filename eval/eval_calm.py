"""
Unified Evaluation Script for Audio-CALM (Flow Matching & GMM).
FIXED: 
1. Reverted Denormalization (since oracle_raw_norm is good).
2. Fixed Positional Embedding Bug in inference loop (The cause of robotic noise).
3. Added Monkey Patch for Torchaudio 2.1+ compatibility.
"""

import os
import sys
import json
import csv
import logging
import random
from typing import Optional, List, Dict

import torch
import torchaudio
import hydra
import wandb
import gradio as gr
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import PeftModel
import evaluate

# --- [CRITICAL FIX] Monkey Patch for Torchaudio 2.1+ vs SpeechBrain ---
if not hasattr(torchaudio, "list_audio_backends"):
    try:
        import torchaudio.backend
        if hasattr(torchaudio.backend, "list_audio_backends"):
             torchaudio.list_audio_backends = torchaudio.backend.list_audio_backends
        else:
             torchaudio.list_audio_backends = lambda: ["soundfile"]
    except Exception:
        torchaudio.list_audio_backends = lambda: []
# ----------------------------------------------------------------------

# Rich logger & progress
from rich.logging import RichHandler
from rich.progress import track
from rich.console import Console
from rich.traceback import install as _install_rich

_install_rich()
logging.basicConfig(
    level="INFO", 
    format="%(message)s", 
    datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("eval")
console = Console()

sys.path.append(os.getcwd())

# === Project Imports ===
# Ensure models/modeling_calm.py exists and is the one with Flow fixes
from models.modeling_calm import QwenCALM, QwenCALMConfig

# Optional HF datasets
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except Exception:
    HF_DATASETS_AVAILABLE = False

from transformers import pipeline, AutoTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_seed(seed: Optional[int]):
    if seed is None: return
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError: pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl(path: str, max_samples: int = -1, seed: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(path):
        logger.error(f"Dataset file not found: {path}")
        return []
        
    if HF_DATASETS_AVAILABLE:
        ds = load_dataset("json", data_files={"test": path}, split="test")
        if max_samples > 0:
            ds = ds.shuffle(seed or 42).select(range(min(len(ds), max_samples)))
        return [dict(x) for x in ds]
    
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line: data.append(json.loads(line))
    
    if max_samples > 0:
        rnd = random.Random(seed or 42)
        rnd.shuffle(data)
        data = data[:max_samples]
    return data

# ---------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------
class Vocoder:
    def __init__(self, source: str = "speechbrain/tts-hifigan-libritts-16kHz", device: str = "cuda"):
        self.device = device
        logger.info(f"Loading vocoder from {source}...")
        try:
            from speechbrain.inference.vocoders import HIFIGAN
            self.hifi_gan = HIFIGAN.from_hparams(
                source=source, 
                savedir="tmp_hifigan_checkpoints", 
                run_opts={"device": device}
            )
        except ImportError:
            logger.critical("SpeechBrain not installed. Run: pip install speechbrain")
            raise

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Input: Mel Spectrogram [B, 80, T]
        Output: Waveform [B, T_wav]
        """
        if mel.dim() == 3 and mel.shape[-1] == 80:
            mel = mel.transpose(1, 2)
        with torch.no_grad():
            wav = self.hifi_gan.decode_batch(mel)
        return wav.squeeze(1)

# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
def load_unified_model(cfg: DictConfig, device: str):
    logger.info(f"Loading base model from {cfg.model.qwen_path}...")
    
    flow_hidden = cfg.model.get("flow_hidden_dim", 2048) # 确保默认值匹配你的训练配置
    flow_layers = cfg.model.get("flow_num_layers", 4)

    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path,
        vae_path=cfg.model.vae_path,
        head_type=cfg.model.get("head_type", "flow"),
        num_mixtures=cfg.model.get("num_mixtures", 8),
        latent_dim=cfg.model.latent_dim,
        use_precomputed_latents=False, 
        flow_hidden_dim=flow_hidden,
        flow_num_layers=flow_layers,
    )
    
    # 1. 初始化基础模型 (Input Proj + LLM + Head)
    model = QwenCALM(config)

    ckpt_path = cfg.evaluation.checkpoint_path
    logger.info(f"Loading weights from {ckpt_path}...")

    # 2. [CRITICAL] 加载 MoA Adapters
    # 你的 checkpoint 目录下应该有 'asr' 和 'tts' (如果训练了的话) 的文件夹
    # 或者直接在根目录下有 adapter_model.bin
    
    from peft import PeftConfig
    
    # 检查是否是多 Adapter 结构
    asr_adapter_path = os.path.join(ckpt_path, "asr")
    tts_adapter_path = os.path.join(ckpt_path, "tts")
    
    # 默认加载逻辑 (单个 Adapter)
    if os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
        logger.info("Found single adapter configuration.")
        model.llm = PeftModel.from_pretrained(model.llm, ckpt_path)
    
    # MoA 加载逻辑
    elif os.path.exists(asr_adapter_path) or os.path.exists(tts_adapter_path):
        logger.info("Found multiple adapters (MoA).")
        first_adapter = True
        
        if os.path.exists(asr_adapter_path):
            logger.info(f"Loading ASR adapter from {asr_adapter_path}")
            if first_adapter:
                model.llm = PeftModel.from_pretrained(model.llm, asr_adapter_path, adapter_name="asr")
                first_adapter = False
            else:
                model.llm.load_adapter(asr_adapter_path, adapter_name="asr")
                
        if os.path.exists(tts_adapter_path):
            logger.info(f"Loading TTS adapter from {tts_adapter_path}")
            if first_adapter:
                model.llm = PeftModel.from_pretrained(model.llm, tts_adapter_path, adapter_name="tts")
                first_adapter = False
            else:
                model.llm.load_adapter(tts_adapter_path, adapter_name="tts")
    else:
        logger.warning("⚠️ No LoRA adapters found! Evaluating base model only.")

    # 3. 如果需要，Merge LoRA (通常 Eval 不 merge，以此保留切换能力)
    if cfg.model.get("merge_lora", False):
        try:
            model.llm = model.llm.merge_and_unload()
            logger.info("LoRA merged.")
        except Exception as e:
            logger.warning(f"Could not merge LoRA: {e}")

    # 4. 加载 Projector 和 Head 权重
    for name in ["input_proj", "output_head"]:
        bin_path = os.path.join(ckpt_path, f"{name}.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            # 处理可能的 DDP 里的 'module.' 前缀
            new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
            getattr(model, name).load_state_dict(new_sd)
            logger.info(f"Loaded {name}.")
        else:
            logger.warning(f"⚠️ {name}.bin not found in checkpoint!")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and cfg.training.get("bf16", False) else torch.float16
    model.llm.to(dtype)
    model.input_proj.to(dtype)
    model.output_head.to(dtype)
    model.vae.to(torch.float32) # VAE 通常保持 fp32
    model.to(device)
    model.eval()
    
    return model

# ---------------------------------------------------------------------
# Inference Methods (Flow & GMM)
# ---------------------------------------------------------------------
def ode_solve_euler(model, condition, initial_noise, n_steps=10, cfg_scale=1.5):
    dt = 1.0 / n_steps
    x = initial_noise
    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype
    
    uncond = torch.zeros_like(condition)

    for i in range(n_steps):
        t_value = i / n_steps
        t = torch.full((batch_size,), t_value, device=device, dtype=dtype)
        v_cond = model.output_head(condition, x, t)
        v_uncond = model.output_head(uncond, x, t)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + v * dt
    return x

def sample_from_gmm(pi, mu, log_sigma, temp=1.0):
    if temp != 1.0:
        pi = pi / temp
    k = torch.distributions.Categorical(logits=pi).sample()
    k_expanded = k.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mu.shape[-1])
    mu_k = torch.gather(mu, 2, k_expanded).squeeze(2)
    log_sigma_k = torch.gather(log_sigma, 2, k_expanded).squeeze(2)
    return torch.normal(mu_k, torch.exp(log_sigma_k))

@torch.no_grad()
def generate_speech(model, tokenizer, vocoder, text: str, device="cuda", max_len: int = 300, flow_steps: int = 10, cfg_scale: float = 3.0):
    # === [CRITICAL FIX] Activate TTS Adapter ===
    if hasattr(model.llm, "set_adapter"):
        if "tts" in model.llm.peft_config:
            model.llm.set_adapter("tts")
        elif "default" in model.llm.peft_config:
            model.llm.set_adapter("default")

    text_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    text_embeds = model.get_input_embeddings()(text_ids) # [1, L_text, H]

    # Start with zero latent
    curr_latent = torch.zeros((1, 1, model.config.latent_dim), device=device, dtype=model.llm.dtype)
    
    # [CRITICAL FIX]: Keep track of ALL latents for correct positional embedding
    all_latents_list = [curr_latent] 
    
    past_key_values = None
    
    # First step input
    curr_audio_embed = model.input_proj(curr_latent) 
    next_input = torch.cat([text_embeds, curr_audio_embed], dim=1)

    for _ in range(max_len):
        # 1. LLM Step
        out = model.llm(inputs_embeds=next_input, use_cache=True, past_key_values=past_key_values, output_hidden_states=True)
        past_key_values = out.past_key_values
        condition = out.hidden_states[-1][:, -1:, :]

        # 2. Audio Head Step
        if getattr(model, "head_type", "flow") == "flow":
            noise = torch.randn_like(curr_latent)
            next_latent = ode_solve_euler(model, condition, noise, n_steps=flow_steps, cfg_scale=cfg_scale)
        else:
            pi, mu, log_sigma = model.output_head(condition)
            next_latent = sample_from_gmm(pi, mu, log_sigma)

        all_latents_list.append(next_latent)
        
        # 3. [CRITICAL FIX] Re-project full sequence to get correct Positional Embeddings
        # Concatenate all history to shape [1, T_curr, D]
        full_latent_seq = torch.cat(all_latents_list, dim=1)
        
        # Pass full sequence to projector. Projector adds pos_emb[0...T] correctly.
        full_audio_embeds = model.input_proj(full_latent_seq)
        
        # Take ONLY the last frame for the next autoregressive step
        # shape [1, 1, H]
        next_input = full_audio_embeds[:, -1:, :]

    # Decode
    # Skip the initial zero latent when decoding
    latents_seq = torch.cat(all_latents_list[1:], dim=1).transpose(1, 2).float() 
    mel_or_wav = model.vae.decode(latents_seq)
    
    # Check VAE output type
    if mel_or_wav.shape[1] == 80:
        # Direct decode without extra normalization
        wav = vocoder.decode(mel_or_wav)
    else:
        wav = mel_or_wav.squeeze(1)
        
    return wav.cpu()

def transcribe_audio(model, tokenizer, latent: torch.Tensor, device="cuda", max_new_tokens: int = 256) -> str:
    """ASR Inference: Latent -> Input Projector -> LLM -> Text."""
    latent = latent.to(device)
    # Match dtype to projector weights
    proj_dtype = next(model.input_proj.parameters()).dtype
    latent = latent.to(proj_dtype)
    
    if latent.dim() == 2: latent = latent.unsqueeze(0) 
    if latent.shape[-1] != model.config.latent_dim: 
        if latent.shape[1] == model.config.latent_dim: latent = latent.transpose(1, 2)
        
    # === [CRITICAL FIX] Activate ASR Adapter ===
    if hasattr(model.llm, "set_adapter"):
        if "asr" in model.llm.peft_config:
            model.llm.set_adapter("asr")
        elif "default" in model.llm.peft_config: # Fallback for single adapter training
            model.llm.set_adapter("default")

    with torch.no_grad():
        audio_embeds = model.input_proj(latent)
        prompt = "Transcribe audio:"
        prompt_ids = tokenizer.encode(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", 
            return_tensors="pt", add_special_tokens=False
        ).to(device)
        prompt_embeds = model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([audio_embeds, prompt_embeds], dim=1)
        
        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds, 
            max_new_tokens=max_new_tokens, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
            do_sample=False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ---------------------------------------------------------------------
# Oracle Test (Gold Standard Check)
# ---------------------------------------------------------------------
def run_oracle_test(cfg, model, vocoder, item):
    """
    Checks if VAE+Vocoder can reconstruct ground truth latent correctly.
    If this fails, the TTS will never work.
    """
    logger.info("🧪 Running Oracle Reconstruction Test...")
    latent_path = item.get("latent_path", "") or item.get("file_path", "")
    if not os.path.exists(latent_path):
        logger.warning("Could not find latent for oracle test.")
        return

    try:
        payload = torch.load(latent_path, map_location="cpu")
        latent = payload.get("latent", payload) if isinstance(payload, dict) else payload
        
        # Ensure shape [1, D, T] for VAE decode
        if latent.shape[-1] == model.config.latent_dim: 
            latent = latent.transpose(0, 1) # [T, D] -> [D, T]
        latent = latent.unsqueeze(0).to(model.vae.dtype).to(cfg.device)

        with torch.no_grad():
            mel_out = model.vae.decode(latent)
            
            # Save raw VAE output (normalized)
            wav_raw = vocoder.decode(mel_out).cpu()
            
        out_dir = os.path.join(cfg.evaluation.output_dir, "oracle_test")
        os.makedirs(out_dir, exist_ok=True)
        
        torchaudio.save(os.path.join(out_dir, "oracle_raw_norm.wav"), wav_raw, 16000)
        
        logger.info(f"✅ Oracle test saved to {out_dir}. Please listen to 'oracle_raw_norm.wav'!")
        
    except Exception as e:
        logger.error(f"Oracle test failed: {e}")

# ---------------------------------------------------------------------
# Evaluation Routines
# ---------------------------------------------------------------------
def evaluate_tts(cfg: DictConfig, model, tokenizer, vocoder, data: List[Dict]):
    logger.info(">>> Starting TTS Evaluation")
    out_dir = os.path.join(cfg.evaluation.output_dir, "tts")
    wav_dir = os.path.join(out_dir, "generated_wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    # Run Oracle Test on the first sample
    if len(data) > 0:
        run_oracle_test(cfg, model, vocoder, data[0])

    asr_pipe = None
    try:
        asr_pipe = pipeline("automatic-speech-recognition", model=cfg.evaluation.eval_asr_model, device=0 if torch.cuda.is_available() else -1)
    except Exception:
        logger.warning("Could not load ASR pipeline for TTS WER metric.")

    wer_metric = evaluate.load("wer")
    results = []
    
    csv_file = open(os.path.join(out_dir, "tts_results.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "text_ref", "text_pred", "wer", "wav_path"])

    n_steps = cfg.evaluation.get("flow_steps", 32)

    for i, item in track(enumerate(data), total=len(data), description="TTS Generation"):
        text = item.get("text", "")
        if not text: continue
        
        try:
            wav = generate_speech(
                model, tokenizer, vocoder, text, 
                device=cfg.get("device", "cuda"),
                max_len=cfg.evaluation.max_latents,
                flow_steps=n_steps
            )
            
            save_path = os.path.join(wav_dir, f"{i}.wav")
            torchaudio.save(save_path, wav, 16000)
            
            wer = 0.0; text_pred = ""
            if asr_pipe:
                transcription = asr_pipe(save_path)["text"]
                text_pred = normalizer(transcription)
                text_ref = normalizer(text)
                if text_ref: wer = wer_metric.compute(predictions=[text_pred], references=[text_ref])
            
            writer.writerow([i, text, text_pred, wer, save_path])
            results.append(wer)
            if i < 5: 
                wandb.log({f"tts_sample_{i}/audio": wandb.Audio(save_path, caption=text[:100]), f"tts_sample_{i}/wer": wer})
        except Exception as e:
            logger.error(f"TTS Failed on sample {i}: {e}")

    csv_file.close()
    if results:
        avg_wer = sum(results) / len(results)
        logger.info(f"🔥 TTS Evaluation Complete. Average WER: {avg_wer:.4f}")
        wandb.log({"eval/tts_avg_wer": avg_wer})

def evaluate_asr(cfg: DictConfig, model, tokenizer, data: List[Dict]):
    logger.info(">>> Starting ASR Evaluation")
    out_dir = os.path.join(cfg.evaluation.output_dir, "asr")
    os.makedirs(out_dir, exist_ok=True)
    
    wer_metric = evaluate.load("wer")
    predictions, references = [], []
    
    csv_file = open(os.path.join(out_dir, "asr_results.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "text_ref", "text_pred", "latent_path"])

    for i, item in track(enumerate(data), total=len(data), description="ASR Transcription"):
        text_gt = item.get("text", "")
        latent_path = item.get("latent_path", "") or item.get("file_path", "")
        
        if not latent_path or not os.path.exists(latent_path): continue
        
        try:
            payload = torch.load(latent_path, map_location="cpu")
            latent = payload.get("latent", payload.get("mel", None)) if isinstance(payload, dict) else payload
            if latent is None: continue

            pred_text = transcribe_audio(model, tokenizer, latent, device=cfg.get("device", "cuda"))
            norm_pred = normalizer(pred_text)
            norm_gt = normalizer(text_gt)
            
            if norm_gt:
                predictions.append(norm_pred)
                references.append(norm_gt)
            
            writer.writerow([i, text_gt, pred_text, latent_path])
            
        except Exception as e:
            logger.error(f"ASR Failed on sample {i}: {e}")

    csv_file.close()
    if predictions:
        final_wer = wer_metric.compute(predictions=predictions, references=references)
        logger.info(f"🔥 ASR Evaluation Complete. Global WER: {final_wer:.4f}")
        wandb.log({"eval/asr_global_wer": final_wer})

# ---------------------------------------------------------------------
# Gradio Demo
# ---------------------------------------------------------------------
def launch_demo(cfg, model, tokenizer, vocoder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def fn_tts(text, steps):
        wav = generate_speech(
            model, tokenizer, vocoder, text, 
            device=device, 
            max_len=cfg.evaluation.max_latents, 
            flow_steps=int(steps)
        )
        return (16000, wav.numpy())
    
    with gr.Blocks(title="Audio-CALM Unified") as demo:
        gr.Markdown("## 🌊 Audio-CALM: Flow Matching / GMM")
        with gr.Tab("TTS"):
            gr.Interface(fn_tts, inputs=["text", gr.Slider(5, 50, value=32, label="Flow Steps")], outputs="audio")
    demo.launch(server_name="0.0.0.0", share=True)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../config", config_name="calm_config")
def main(cfg: DictConfig):
    console.rule("[bold magenta]Audio-CALM Evaluation[/bold magenta]")
    
    task_arg = cfg.evaluation.task.lower()
    run_tts = task_arg in ["tts", "mix"]
    run_asr = task_arg in ["asr", "mix"]
    
    logger.info(f"Mode: {task_arg} (TTS={run_tts}, ASR={run_asr})")
    
    set_seed(cfg.evaluation.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # [FIX] Use open_dict to inject keys safely
    with open_dict(cfg):
        cfg.device = device

    # [FIX] Use .get() for safety against missing config keys
    if not cfg.evaluation.get("web_demo", False):
        wandb.init(
            project=cfg.evaluation.get("wandb_project", "Audio-CALM-Eval"),
            name=f"eval-{task_arg}-{os.path.basename(cfg.evaluation.checkpoint_path)}",
            config=OmegaConf.to_container(cfg)
        )

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    model = load_unified_model(cfg, device)
    
    # Load Vocoder (Only needed for TTS)
    vocoder = None
    if run_tts or cfg.evaluation.get("web_demo", False):
        vocoder = Vocoder(device=device)

    # Run
    if cfg.evaluation.get("web_demo", False):
        launch_demo(cfg, model, tokenizer, vocoder)
    else:
        data = load_jsonl(cfg.evaluation.test_file, cfg.evaluation.max_samples)
        
        if run_tts:
            evaluate_tts(cfg, model, tokenizer, vocoder, data)
        
        if run_asr:
            evaluate_asr(cfg, model, tokenizer, data)

if __name__ == "__main__":
    main()