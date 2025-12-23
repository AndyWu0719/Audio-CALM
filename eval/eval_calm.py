"""
Unified Evaluation Script for Audio-CALM.
Fixed: 
1. Forced Float32 for VAE/Vocoder to prevent type mismatch errors.
2. Uses soundfile for stable WAV saving.
3. Corrected JSON keys and CSV logging.
"""

import os
import sys
import json
import csv
import logging
import random
import torch
import torchaudio
import soundfile as sf  # [FIX] Use soundfile for stable WAV saving
import hydra
import wandb
import numpy as np
import gradio as gr
from typing import List, Dict, Optional
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import PeftModel
import evaluate
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

# --- Environment Patches ---
if not hasattr(torchaudio, "list_audio_backends"):
    try:
        import torchaudio.backend
        torchaudio.list_audio_backends = getattr(torchaudio.backend, "list_audio_backends", lambda: ["soundfile"])
    except ImportError:
        torchaudio.list_audio_backends = lambda: []

sys.path.append(os.getcwd())
from models.modeling_calm import QwenCALM, QwenCALMConfig

# --- Logging Setup ---
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger("eval")
console = Console()

# --- Metrics & Text Norm ---
try:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()
except ImportError:
    normalizer = lambda x: x.lower()

wer_metric = evaluate.load("wer")

# ==============================================================================
# Utils
# ==============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset_jsonl(path: str, max_samples: int = -1) -> List[Dict]:
    """Loads dataset. Supports keys: 'audio', 'file_path', 'latent_path'."""
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
    else:
        logger.info(f"Loaded full dataset: {len(data)} samples.")
        
    return data

def save_summary(output_dir: str, task: str, global_metrics: Dict, examples: List[Dict]):
    path = os.path.join(output_dir, f"{task}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "global_metrics": global_metrics,
            "examples": examples
        }, f, indent=4, ensure_ascii=False)
    logger.info(f"📝 Summary saved to {path}")

# ==============================================================================
# Components: Vocoder & Models
# ==============================================================================

class Vocoder:
    def __init__(self, device="cuda"):
        from speechbrain.inference.vocoders import HIFIGAN
        self.hifi = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="tmp_hifigan",
            run_opts={"device": device}
        )
    
    def decode(self, mel):
        # Ensure Input is Float32
        mel = mel.to(torch.float32)
        if mel.dim() == 3 and mel.shape[-1] == 80: mel = mel.transpose(1, 2)
        with torch.no_grad(): return self.hifi.decode_batch(mel).squeeze(1)

def load_model(cfg, device):
    logger.info(f"🤖 Loading Model: {cfg.model.qwen_path}")
    
    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path,
        vae_path=cfg.model.vae_path,
        latent_dim=cfg.model.latent_dim,
        flow_hidden_dim=cfg.model.get("flow_hidden_dim", 2048),
        flow_num_layers=cfg.model.get("flow_num_layers", 4),
    )
    model = QwenCALM(config)
    
    ckpt = cfg.evaluation.checkpoint_path
    if os.path.exists(os.path.join(ckpt, "asr")) or os.path.exists(os.path.join(ckpt, "tts")):
        logger.info("Detected Mixture of Adapters (MoA).")
        if os.path.exists(os.path.join(ckpt, "asr")):
            model.llm = PeftModel.from_pretrained(model.llm, os.path.join(ckpt, "asr"), adapter_name="asr")
        if os.path.exists(os.path.join(ckpt, "tts")):
            try:
                model.llm.load_adapter(os.path.join(ckpt, "tts"), adapter_name="tts")
            except:
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
            logger.info(f"Loaded {component}.")

    model.to(device).eval()
    
    # [CRITICAL FIX] Mixed Precision Handling
    if cfg.training.get("bf16", False): 
        model.to(torch.bfloat16)
        # Force VAE to float32 for stable decoding
        model.vae.to(torch.float32)
        logger.info("Model set to BF16 (VAE kept in FP32).")
    
    return model

# ==============================================================================
# Inference Logic
# ==============================================================================

@torch.no_grad()
def run_tts_inference(model, tokenizer, vocoder, text, steps=10, cfg_scale=3.0, device="cuda"):
    if hasattr(model.llm, "set_adapter") and "tts" in model.llm.peft_config:
        model.llm.set_adapter("tts")

    text_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    text_embeds = model.get_input_embeddings()(text_ids)
    
    curr_latent = torch.zeros((1, 1, model.config.latent_dim), device=device, dtype=model.llm.dtype)
    history_latents = [curr_latent]
    past_kv = None
    
    next_input = torch.cat([text_embeds, model.input_proj(curr_latent)], dim=1)

    for _ in range(300): # Max Latents limit
        out = model.llm(inputs_embeds=next_input, use_cache=True, past_key_values=past_kv, output_hidden_states=True)
        past_kv = out.past_key_values
        condition = out.hidden_states[-1][:, -1:, :]

        # Flow Matching ODE Step
        noise = torch.randn_like(curr_latent)
        dt = 1.0 / steps
        x = noise
        for i in range(steps):
            t = torch.full((1,), i/steps, device=device, dtype=x.dtype)
            v_cond = model.output_head(condition, x, t)
            v_uncond = model.output_head(torch.zeros_like(condition), x, t)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            x = x + v * dt
        
        history_latents.append(x)
        full_seq = torch.cat(history_latents, dim=1)
        next_input = model.input_proj(full_seq)[:, -1:, :]

    # Decode
    # [CRITICAL FIX] Cast to float32 before VAE decoding
    latents = torch.cat(history_latents[1:], dim=1).transpose(1, 2).float()
    mel = model.vae.decode(latents)
    wav = vocoder.decode(mel)
    return wav.cpu()

@torch.no_grad()
def run_asr_inference(model, tokenizer, latent, device="cuda"):
    if hasattr(model.llm, "set_adapter") and "asr" in model.llm.peft_config:
        model.llm.set_adapter("asr")

    # ASR needs input to match LLM dtype (BF16/FP16)
    latent = latent.to(device).to(model.input_proj.weight.dtype)
    
    if latent.dim() == 2: latent = latent.unsqueeze(0)
    if latent.shape[-1] != model.config.latent_dim: latent = latent.transpose(1, 2)

    audio_embeds = model.input_proj(latent)
    prompt_ids = tokenizer.encode("<|im_start|>user\nTranscribe audio:<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(device)
    inputs = torch.cat([audio_embeds, model.get_input_embeddings()(prompt_ids)], dim=1)
    
    out = model.llm.generate(inputs_embeds=inputs, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ==============================================================================
# Oracle & Evaluation
# ==============================================================================

def run_oracle_test(cfg, model, vocoder, item):
    """Checks if VAE+Vocoder works by reconstructing GT latent."""
    logger.info("🧪 Running Oracle Reconstruction Test...")
    latent_path = item.get("audio") or item.get("latent_path") or item.get("file_path")
    
    if not latent_path or not os.path.exists(latent_path):
        logger.warning(f"Could not find latent for oracle test. Path key missing or file {latent_path} invalid.")
        return

    try:
        payload = torch.load(latent_path, map_location="cpu")
        latent = payload.get("latent", payload) if isinstance(payload, dict) else payload
        if latent.shape[-1] == model.config.latent_dim: latent = latent.transpose(0, 1)
        
        # [CRITICAL FIX] Cast to Float32 for VAE
        latent = latent.unsqueeze(0).to(torch.float32).to(cfg.device)

        with torch.no_grad():
            mel_out = model.vae.decode(latent)
            wav_raw = vocoder.decode(mel_out).cpu().numpy()
            
        out_dir = os.path.join(cfg.evaluation.output_dir, "oracle_test")
        os.makedirs(out_dir, exist_ok=True)
        
        sf.write(os.path.join(out_dir, "oracle_raw_norm.wav"), wav_raw.squeeze(), 16000)
        logger.info(f"✅ Oracle test saved to {out_dir}.")
        
    except Exception as e:
        logger.error(f"Oracle test failed: {e}")

def eval_task_tts(cfg, model, tokenizer, vocoder, data):
    console.print("[bold green]>>> Running TTS Evaluation[/bold green]")
    from transformers import pipeline
    
    asr_pipe = None
    try:
        logger.info(f"Loading ASR Metric Model: {cfg.evaluation.eval_asr_model}")
        asr_pipe = pipeline("automatic-speech-recognition", model=cfg.evaluation.eval_asr_model, device=0)
    except Exception as e:
        logger.warning(f"ASR pipeline load failed: {e}. WER will be 0.")

    wav_dir = os.path.join(cfg.evaluation.output_dir, "generated_wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    if len(data) > 0: run_oracle_test(cfg, model, vocoder, data[0])

    predictions, references = [], []
    saved_examples = []
    
    csv_path = os.path.join(cfg.evaluation.output_dir, "tts_results.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "wer", "text_ref", "text_pred", "wav_path"])

    for i, item in enumerate(data):
        text = item.get("text", "")
        if not text: continue
        
        try:
            # Inference
            wav = run_tts_inference(model, tokenizer, vocoder, text, steps=cfg.evaluation.flow_steps, device=cfg.device)
            
            save_path = os.path.join(wav_dir, f"sample_{i}.wav")
            sf.write(save_path, wav.squeeze().numpy(), 16000)
            
            # Metric
            pred_text = ""
            wer = 0.0
            if asr_pipe:
                transcription = asr_pipe(save_path)["text"]
                pred_text = normalizer(transcription)
                ref_text = normalizer(text)
                if len(ref_text) > 0:
                    wer = wer_metric.compute(predictions=[pred_text], references=[ref_text])
            
            predictions.append(pred_text)
            references.append(normalizer(text))
            
            writer.writerow([i, f"{wer:.4f}", text, pred_text, save_path])
            
            if (i + 1) % 10 == 0:
                console.print(f"[Sample {i+1}] WER: [bold yellow]{wer:.2f}[/bold yellow] | GT: {text[:20]}... | Pred: {pred_text[:20]}...")
            
            if len(saved_examples) < 10:
                saved_examples.append({"id": i, "wer": wer, "text": text, "pred": pred_text})
                wandb.log({f"tts_sample_{i}": wandb.Audio(save_path, caption=f"WER: {wer:.2f} | {text}")})

        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")

    csv_file.close()
    
    valid_refs = [r for r in references if len(r) > 0]
    valid_preds = [p for i, p in enumerate(predictions) if len(references[i]) > 0]
    global_wer = wer_metric.compute(predictions=valid_preds, references=valid_refs) if valid_refs else 0.0
    
    console.rule(f"[bold red]TTS Result (Average WER): {global_wer:.4f}[/bold red]")
    save_summary(cfg.evaluation.output_dir, "tts", {"avg_wer": global_wer}, saved_examples)
    wandb.log({"tts/avg_wer": global_wer})

def eval_task_asr(cfg, model, tokenizer, data):
    console.print("[bold green]>>> Running ASR Evaluation[/bold green]")
    out_dir = os.path.join(cfg.evaluation.output_dir, "asr")
    os.makedirs(out_dir, exist_ok=True)
    
    predictions, references = [], []
    saved_examples = []
    
    csv_file = open(os.path.join(out_dir, "asr_results.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "wer", "text_ref", "text_pred", "latent_path"])

    for i, item in enumerate(data):
        gt_text = normalizer(item.get("text", ""))
        latent_path = item.get("audio") or item.get("latent_path") 
        
        if not latent_path: continue

        try:
            latent = torch.load(latent_path, map_location="cpu")
            if isinstance(latent, dict): latent = latent.get("latent", latent.get("mel"))
            
            pred_raw = run_asr_inference(model, tokenizer, latent, cfg.device)
            pred_text = normalizer(pred_raw)
            
            current_wer = wer_metric.compute(predictions=[pred_text], references=[gt_text]) if gt_text else 1.0
            
            predictions.append(pred_text)
            references.append(gt_text)
            
            writer.writerow([i, f"{current_wer:.4f}", gt_text, pred_text, latent_path])
            
            if (i + 1) % 10 == 0:
                console.print(f"[Sample {i+1}] WER: {current_wer:.2f} | GT: {gt_text[:20]}... | Pred: {pred_text[:20]}...")

            if len(saved_examples) < 10:
                saved_examples.append({"id": i, "wer": current_wer, "gt": gt_text, "pred": pred_text})
                
        except Exception as e:
            logger.error(f"ASR Error sample {i}: {e}")

    csv_file.close()
    global_wer = wer_metric.compute(predictions=predictions, references=references) if references else 0.0
    console.rule(f"[bold red]ASR Result: {global_wer:.4f}[/bold red]")
    
    save_summary(cfg.evaluation.output_dir, "asr", {"wer": global_wer}, saved_examples)
    wandb.log({"asr/wer": global_wer})

# ==============================================================================
# Main
# ==============================================================================

@hydra.main(version_base=None, config_path="../config", config_name="calm_config")
def main(cfg: DictConfig):
    set_seed(cfg.evaluation.get("seed", 42))
    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)
    with open_dict(cfg): cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not cfg.evaluation.get("web_demo", False):
        wandb.init(
            project=cfg.evaluation.get("wandb_project", "Audio-CALM-Eval"),
            name=f"eval-{cfg.evaluation.task}",
            config=OmegaConf.to_container(cfg)
        )

    # Load Model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    
    model = load_model(cfg, cfg.device)
    
    # Run Web Demo
    if cfg.evaluation.get("web_demo", False):
        logger.info("Starting Gradio Demo...")
        vocoder = Vocoder(cfg.device)
        def fn_tts(text, steps):
            wav = run_tts_inference(model, tokenizer, vocoder, text, steps=int(steps), device=cfg.device)
            return (16000, wav.numpy())
        gr.Interface(fn_tts, inputs=["text", gr.Slider(10, 50, value=32)], outputs="audio").launch(server_name="0.0.0.0", share=True)
        return

    # Load Data
    data = load_dataset_jsonl(cfg.evaluation.test_file, cfg.evaluation.max_samples)
    task = cfg.evaluation.task.lower()

    if task in ["asr", "mix"]:
        eval_task_asr(cfg, model, tokenizer, data)

    if task in ["tts", "mix"]:
        vocoder = Vocoder(cfg.device)
        eval_task_tts(cfg, model, tokenizer, vocoder, data)

    console.print("[bold blue]Evaluation Finished![/bold blue]")

if __name__ == "__main__":
    main()