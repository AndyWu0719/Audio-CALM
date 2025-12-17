"""
Evaluation script for Audio-CALM GMM (TTS / ASR).
- Uses Hugging Face datasets/pipeline/evaluate where applicable.
- Produces per-sample outputs and global metrics (WER).
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
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
import evaluate

# Rich logger & progress
from rich.logging import RichHandler
from rich.progress import track
from rich.traceback import install as _install_rich
_install_rich()
logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("eval")

sys.path.append(os.getcwd())

# Optional HF datasets
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except Exception:
    HF_DATASETS_AVAILABLE = False

from transformers import pipeline, AutoTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from models.modeling_gmm import QwenCALM, QwenCALMConfig

normalizer = BasicTextNormalizer()

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: Optional[int]):
    """Set deterministic seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl(path: str, max_samples: int = -1, seed: Optional[int] = None) -> List[Dict]:
    """Load JSONL dataset; use HF 'datasets' if available for deterministic sampling."""
    if HF_DATASETS_AVAILABLE:
        ds = load_dataset("json", data_files={"test": path}, split="test")
        if max_samples > 0:
            ds = ds.shuffle(seed or 0).select(range(max_samples))
        return [dict(x) for x in ds]
    data = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    if max_samples > 0:
        rnd = random.Random(seed or 0)
        rnd.shuffle(data)
        data = data[:max_samples]
    return data

# ---------------------------
# Vocoder
# ---------------------------
class Vocoder:
    """Simple wrapper for SpeechBrain HiFi-GAN vocoder."""
    def __init__(self, source: str = "speechbrain/tts-hifigan-libritts-16kHz", device: str = "cuda"):
        self.device = device
        logger.info(f"Loading vocoder from {source}...")
        try:
            from speechbrain.inference.vocoders import HIFIGAN
            self.hifi_gan = HIFIGAN.from_hparams(source=source, savedir="tmp_hifigan_checkpoints", run_opts={"device": device})
        except ImportError as e:
            raise RuntimeError("speechbrain not installed; install via `pip install speechbrain`") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load vocoder: {e}") from e

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """Decode mel spectrogram to waveform. Accepts [B, 80, T] or [B, T, 80]."""
        if mel.shape[-1] == 80 and mel.shape[1] != 80:
            mel = mel.transpose(1, 2)
        with torch.no_grad():
            wav = self.hifi_gan.decode_batch(mel)
        if wav.dim() == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)
        return wav

# ---------------------------
# Model loading & helpers
# ---------------------------
def load_calm_model(cfg: DictConfig, device: str):
    """Instantiate QwenCALM and load optional LoRA/projector/head weights."""
    logger.info(f"Loading model base from {cfg.model.qwen_path}...")
    cfg_model = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path,
        vae_path=cfg.model.vae_path,
        num_mixtures=cfg.model.num_mixtures,
        latent_dim=cfg.model.latent_dim,
        downsample_rate=cfg.data.latent_downsample,
        use_precomputed_latents=False,
    )
    model = QwenCALM(cfg_model)

    # Load LoRA adapter if present
    adapter_dir = cfg.evaluation.checkpoint_path
    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_config):
        logger.info(f"Loading LoRA adapter from {adapter_dir}...")
        model.llm = PeftModel.from_pretrained(model.llm, adapter_dir)
        if cfg.model.merge_lora:
            model.llm = model.llm.merge_and_unload()

    # Load projector / head if present
    proj_path = os.path.join(adapter_dir, "input_proj.bin")
    head_path = os.path.join(adapter_dir, "output_head.bin")
    if os.path.exists(proj_path):
        model.input_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
    if os.path.exists(head_path):
        model.output_head.load_state_dict(torch.load(head_path, map_location="cpu"))

    # Move to appropriate dtype and device
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.llm.to(llm_dtype)
    model.input_proj.to(llm_dtype)
    model.output_head.to(llm_dtype)
    model.vae.to(torch.float32)
    model.to(device)
    model.eval()
    return model

# ---------------------------
# GMM sampling
# ---------------------------
def sample_from_gmm(pi: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, temp: float = 1.0):
    """Sample latent vector from diagonal GMM predicted by model."""
    if temp != 1.0:
        pi = pi / temp
    dist_k = torch.distributions.Categorical(logits=pi)
    k_idx = dist_k.sample()
    B, S, K, D = mu.shape
    k_idx_expanded = k_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)
    mu_k = torch.gather(mu, 2, k_idx_expanded).squeeze(2)
    log_sigma_k = torch.gather(log_sigma, 2, k_idx_expanded).squeeze(2)
    sigma_k = torch.exp(log_sigma_k)
    z = torch.normal(mu_k, sigma_k)
    return z

# ---------------------------
# Generation & Transcription
# ---------------------------
def generate_speech(model, tokenizer, vocoder, text: str, device="cuda", max_len: int = 300, temp: float = 0.8):
    """Generate audio waveform from text via autoregressive latent sampling."""
    text_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(text_ids)

    curr_latent = torch.zeros((1, 1, model.config.latent_dim), device=device, dtype=model.llm.dtype)
    curr_audio_embed = model.input_proj(curr_latent)

    generated_latents = []
    past_key_values = None
    next_input_embeds = torch.cat([text_embeds, curr_audio_embed], dim=1)

    with torch.no_grad():
        for _ in range(max_len):
            out = model.llm(inputs_embeds=next_input_embeds, use_cache=True, past_key_values=past_key_values, output_hidden_states=True)
            past_key_values = out.past_key_values
            hidden_state = out.hidden_states[-1][:, -1:, :]
            pi, mu, log_sigma = model.output_head(hidden_state)
            next_latent = sample_from_gmm(pi, mu, log_sigma, temp=temp).to(dtype=model.llm.dtype)
            generated_latents.append(next_latent)
            next_input_embeds = model.input_proj(next_latent)

    latents_seq = torch.cat(generated_latents, dim=1).transpose(1, 2).float()  # [1, D, T]
    with torch.no_grad():
        decoded_out = model.vae.decode(latents_seq)

    if decoded_out.shape[1] == 80:
        if vocoder is None:
            raise RuntimeError("VAE returned mel-spectrograms but no vocoder provided.")
        wav = vocoder.decode(decoded_out)
    else:
        wav = decoded_out.squeeze(1)
    return wav.cpu()

def transcribe_audio(model, tokenizer, latent: torch.Tensor, device="cuda", max_new_tokens: int = 256) -> str:
    """Transcribe latents by concatenating audio embeddings with a prompt and decoding."""
    latent = latent.to(device)
    proj_dtype = next(model.input_proj.parameters()).dtype
    latent = latent.to(proj_dtype)
    if latent.dim() == 2:
        latent = latent.unsqueeze(0)

    target_dim = model.config.latent_dim
    if latent.shape[1] == target_dim and latent.shape[2] != target_dim:
        latent = latent.transpose(1, 2)
    if latent.shape[-1] != target_dim:
        raise ValueError(f"Shape mismatch: expected last dim {target_dim}, got {latent.shape}")

    with torch.no_grad():
        audio_embeds = model.input_proj(latent)
        prompt = "Transcribe audio:"
        prompt_ids = tokenizer.encode(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt", add_special_tokens=False).to(device)
        prompt_embeds = model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([audio_embeds, prompt_embeds], dim=1)
        output_ids = model.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ---------------------------
# Evaluation routines
# ---------------------------
def evaluate_tts(cfg: DictConfig, model, tokenizer, vocoder, data: List[Dict]):
    """Batch TTS evaluation: generate audio, save wavs, compute WER via ASR pipeline."""
    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)
    wav_dir = os.path.join(cfg.evaluation.output_dir, "generated_wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # ASR pipeline for WER (optional)
    eval_asr_pipe = None
    try:
        eval_asr_pipe = pipeline("automatic-speech-recognition", model=cfg.evaluation.eval_asr_model, device=0 if torch.cuda.is_available() else -1, chunk_length_s=30)
    except Exception as e:
        logger.warning(f"Failed to load ASR pipeline for WER: {e}. Skipping WER computation.")

    wer_metric = evaluate.load("wer")
    predictions, references = [], []

    csv_path = os.path.join(cfg.evaluation.output_dir, "tts_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["index", "text_ref", "pred_text", "wav_path"])

        for i, item in track(enumerate(data), total=len(data), description="TTS Evaluation"):
            text_gt = item.get("text", "")
            if not text_gt:
                continue
            try:
                wav_tensor = generate_speech(model, tokenizer, vocoder, text_gt, device=cfg.get("device", "cuda"), max_len=cfg.evaluation.max_latents, temp=0.8)
                if wav_tensor.dim() > 2 or wav_tensor.shape[-1] < 50:
                    continue
                save_path = os.path.join(wav_dir, f"sample_{i}.wav")
                torchaudio.save(save_path, wav_tensor, 16000)
                # log first n
                if i < 10:
                    wandb.log({f"audio/sample_{i}": wandb.Audio(save_path, caption=text_gt[:200])})
                pred_text = ""
                if eval_asr_pipe:
                    asr_out = eval_asr_pipe({"raw": wav_tensor.squeeze().numpy(), "sampling_rate": 16000})
                    pred_text = normalizer(asr_out["text"])
                    norm_gt = normalizer(text_gt)
                    if norm_gt.strip():
                        predictions.append(pred_text)
                        references.append(norm_gt)
                writer.writerow([i, text_gt, pred_text, save_path])
            except Exception as e:
                logger.error(f"Error generating sample {i}: {e}")

    if predictions:
        final_wer = wer_metric.compute(predictions=predictions, references=references)
        logger.info(f"Global TTS WER: {final_wer:.4f}")
        wandb.log({"eval/tts_wer": final_wer})
        with open(os.path.join(cfg.evaluation.output_dir, "tts_results.txt"), "w") as fh:
            fh.write(f"WER: {final_wer:.4f}\n")

def evaluate_asr(cfg: DictConfig, model, tokenizer, data: List[Dict]):
    """Batch ASR evaluation: load latents, transcribe, compute WER."""
    wer_metric = evaluate.load("wer")
    predictions, references = [], []
    csv_path = os.path.join(cfg.evaluation.output_dir, "asr_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["index", "text_ref", "pred_text", "latent_path"])
        for i, item in track(enumerate(data), total=len(data), description="ASR Evaluation"):
            text_gt = item.get("text", "")
            latent_path = item.get("latent_path", "") or item.get("file_path", "")
            if not latent_path or not os.path.exists(latent_path):
                continue
            try:
                payload = torch.load(latent_path, map_location="cpu")
                latent = payload["latent"] if isinstance(payload, dict) else payload
                pred_text = transcribe_audio(model, tokenizer, latent, device=cfg.get("device", "cuda"))
                norm_pred = normalizer(pred_text)
                norm_gt = normalizer(text_gt)
                if norm_gt.strip():
                    predictions.append(norm_pred)
                    references.append(norm_gt)
                writer.writerow([i, text_gt, pred_text, latent_path])
            except Exception as e:
                logger.error(f"Error in ASR sample {i}: {e}")

    if predictions:
        final_wer = wer_metric.compute(predictions=predictions, references=references)
        logger.info(f"Global ASR WER: {final_wer:.4f}")
        wandb.log({"eval/asr_wer": final_wer})

# ---------------------------
# Gradio demo
# ---------------------------
def create_demo(model, tokenizer, vocoder, device, max_latents):
    """Simple Gradio interactive demo for TTS and placeholder ASR."""
    def tts_fn(text, temp):
        if not text:
            return None
        try:
            wav = generate_speech(model, tokenizer, vocoder, text, device=device, max_len=max_latents, temp=float(temp))
            return (16000, wav.squeeze().numpy())
        except Exception as e:
            logger.error(f"Demo generation error: {e}")
            return None

    def asr_fn(audio_path):
        return "ASR from raw audio requires MelExtractor integration."

    with gr.Blocks(title="Audio-CALM Demo") as demo:
        gr.Markdown("# 🎵 Audio-CALM Interactive Demo")
        with gr.Tab("Text-to-Speech (TTS)"):
            with gr.Row():
                t_input = gr.Textbox(label="Input Text", lines=2)
                t_temp = gr.Slider(0.1, 2.0, value=0.8)
            t_btn = gr.Button("Generate")
            t_out = gr.Audio()
            t_btn.click(tts_fn, inputs=[t_input, t_temp], outputs=t_out)
        with gr.Tab("ASR"):
            gr.Markdown("Placeholder: ASR needs MelExtractor.")
    return demo

# ---------------------------
# Main (Hydra entry)
# ---------------------------
@hydra.main(version_base=None, config_path="../config", config_name="gmm_config")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.evaluation.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)
    wandb_mode = "online" if not cfg.evaluation.web_demo else "disabled"
    wandb.init(project=cfg.evaluation.get("wandb_project", "Audio-CALM-Eval"),
               entity=cfg.evaluation.get("wandb_entity", None),
               config=OmegaConf.to_container(cfg, resolve=True),
               mode=wandb_mode,
               name=f"eval-{cfg.evaluation.task}-{os.path.basename(cfg.evaluation.checkpoint_path)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_calm_model(cfg, device)

    vocoder = None
    if cfg.evaluation.task == "tts" and cfg.evaluation.use_vocoder:
        vocoder = Vocoder(device=device)

    data = load_jsonl(cfg.evaluation.test_file, max_samples=cfg.evaluation.max_samples, seed=cfg.evaluation.seed)
    logger.info(f"Loaded {len(data)} examples for {cfg.evaluation.task.upper()}")

    if cfg.evaluation.web_demo:
        demo = create_demo(model, tokenizer, vocoder, device, cfg.evaluation.max_latents)
        demo.launch(server_name="0.0.0.0", share=True)
        return

    if cfg.evaluation.task == "tts":
        evaluate_tts(cfg, model, tokenizer, vocoder, data)
    elif cfg.evaluation.task == "asr":
        evaluate_asr(cfg, model, tokenizer, data)
    else:
        raise ValueError(f"Unknown evaluation task: {cfg.evaluation.task}")

if __name__ == "__main__":
    main()