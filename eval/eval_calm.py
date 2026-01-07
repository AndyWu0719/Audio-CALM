"""
Unified Evaluation Script for Omni-Flow Audio-CALM.
Tasks:
- ASR: NAR Flow head -> text embeddings -> nearest vocab
- TTS: NAR Flow head -> VAE decode -> Vocoder
Metrics:
- ASR: WER / CER (HF evaluate)
- TTS: WER / CER via Whisper ASR on synthesized audio
"""

import os
import sys
import json
import csv
import logging
import random
import math
import re
import torch
import torchaudio
import soundfile as sf
import hydra
import wandb
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import PeftModel
import evaluate
from rich.logging import RichHandler
from rich.console import Console
from transformers import pipeline
from glob import glob

# --- Environment Patches ---
if not hasattr(torchaudio, "list_audio_backends"):
    try:
        import torchaudio.backend
        torchaudio.list_audio_backends = getattr(torchaudio.backend, "list_audio_backends", lambda: ["soundfile"])
    except ImportError:
        torchaudio.list_audio_backends = lambda: []

sys.path.append(os.getcwd())
from models.modeling_calm import QwenCALM, QwenCALMConfig
from models.modeling_calm import build_alignment_from_durations, distribute_remainder_vectorized
from models.modeling_vae import AcousticVAE

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("eval")
console = Console()

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def _normalize(text: str):
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Data Loading Utils
# ==============================================================================
def scan_eval_data(root_dir, subsets, max_samples=-1):
    if not root_dir or not subsets:
        logger.warning(f"⚠️ Empty data config: dir={root_dir}, subsets={subsets}")
        return []

    files = []
    subset_list = subsets.split(",") if isinstance(subsets, str) else []
    for subset in subset_list:
        pattern = os.path.join(root_dir, subset.strip(), "**", "*.trans.txt")
        found = sorted(glob(pattern, recursive=True))
        files.extend(found)

    data_list = []
    for trans_file in files:
        folder = os.path.dirname(trans_file)
        try:
            with open(trans_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    fid, txt = parts
                    pt_path = os.path.join(folder, f"{fid}.pt")
                    if os.path.exists(pt_path):
                        data_list.append(
                            {"text": txt, "latent_path": pt_path, "file_id": fid}
                        )
        except Exception as e:
            logger.warning(f"Error reading {trans_file}: {e}")
            continue

    if max_samples > 0 and len(data_list) > max_samples:
        random.shuffle(data_list)
        data_list = data_list[:max_samples]
    logger.info(f"✅ Loaded {len(data_list)} samples from {root_dir} ({subsets})")
    return data_list


def load_eval_datasets(cfg):
    asr_data = []
    tts_data = []
    task = cfg.evaluation.task.lower()
    max_samples = cfg.evaluation.get("max_samples", -1)

    if task in ["asr", "mix", "tts"]:
        logger.info("📂 Loading ASR Data...")
        asr_data = scan_eval_data(
            cfg.evaluation.datasets.asr.latent_dir,
            cfg.evaluation.datasets.asr.subsets,
            max_samples,
        )
    if task in ["tts", "mix", "asr"]:
        logger.info("📂 Loading TTS Data...")
        tts_data = scan_eval_data(
            cfg.evaluation.datasets.tts.latent_dir,
            cfg.evaluation.datasets.tts.subsets,
            max_samples,
        )
    return asr_data, tts_data


# ==============================================================================
# Sway Sampling Solver (DiT Compatible)
# ==============================================================================
def ode_solve_sway(model_head, condition, x_start, steps, cfg_scale=1.0, device="cuda",
                   context=None, context_mask=None, x_mask=None):  # [FIX] 添加参数
    x = x_start
    dt = 1.0 / steps
    B = x.shape[0]

    for i in range(steps):
        t_curr = i / steps
        t_tensor = torch.full((B,), t_curr, device=device, dtype=x.dtype)

        # Conditional prediction
        v_cond = model_head(condition, x, t_tensor, context=context, context_mask=context_mask, x_mask=x_mask)

        if cfg_scale != 1.0 and cfg_scale > 0:
            null_condition = torch.zeros_like(condition)
            null_context = torch.zeros_like(context) if context is not None else None
            null_context_mask = context_mask
            v_uncond = model_head(null_condition, x, t_tensor,
                                  context=null_context, context_mask=null_context_mask, x_mask=x_mask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        x = x + v * dt
    return x

# ==============================================================================
# Vocoder
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
                run_opts={"device": device},
            )
        except:
            pass

        self.mel_fb = torchaudio.transforms.MelScale(
            n_mels=80, sample_rate=16000, n_stft=513
        ).to(device).fb
        self.inverse_mel_basis = torch.linalg.pinv(self.mel_fb).to(device)
        self.griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024).to(device)

    def decode(self, mel):
        # mel input: [B, C, T] (Standard VAE output)
        mel = mel.to(self.device).float()

        if self.hifi:
            try:
                return self.hifi.decode_batch(mel).squeeze(1)
            except:
                pass

        # Griffin-Lim fallback
        energy = torch.exp(mel)
        mag = torch.sqrt(
            torch.clamp(
                torch.matmul(energy.transpose(1, 2), self.inverse_mel_basis).transpose(1, 2),
                min=1e-8,
            )
        )
        return self.griffin_lim(mag).squeeze(1)


# ==============================================================================
# Model Loading
# ==============================================================================
def load_model(cfg, device):
    logger.info(f"🤖 Loading Model Base: {cfg.model.qwen_path}")

    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path,
        vae_path=cfg.model.vae_path,
        latent_dim=cfg.model.latent_dim,
        use_precomputed_latents=True,
        max_audio_len=cfg.data.max_audio_len,
        max_text_len=cfg.data.max_text_len,
        tts_flow_hidden_dim=cfg.model.tts_flow_hidden_dim,
        tts_flow_num_layers=cfg.model.tts_flow_num_layers,
        asr_flow_hidden_dim=cfg.model.asr_flow_hidden_dim,
        asr_flow_num_layers=cfg.model.asr_flow_num_layers,
        tts_loss_weight=cfg.model.get("tts_loss_weight", 1.0),
        asr_loss_weight=cfg.model.get("asr_loss_weight", 1.0),
        len_pred_loss_weight=cfg.model.get("len_pred_loss_weight", 0.1),
        dur_pred_loss_weight=cfg.model.get("dur_pred_loss_weight", 0.1),
        mel_mean=cfg.model.mel_mean,
        mel_std=cfg.model.mel_std,
        latent_mean=cfg.model.latent_mean,
        latent_std=cfg.model.latent_std,
    )

    model = QwenCALM(config)

    if not hasattr(model, "vae"):
        logger.info("  - Loading VAE for TTS decoding...")
        vae = AcousticVAE.from_pretrained(cfg.model.vae_path)
        vae.to(device)
        vae.eval()
        vae.requires_grad_(False)
        model.vae = vae

    # 1. Load LoRA
    ckpt_dir = cfg.evaluation.checkpoint_path
    if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
        logger.info("  - Loading LoRA Adapter...")
        model.llm = PeftModel.from_pretrained(model.llm, ckpt_dir)

    # 2. Load Custom Components
    logger.info(f"  - Loading Custom Components from {ckpt_dir}...")
    components = [
        "input_proj",
        "tts_flow_head",
        "asr_flow_head",
        "soa_embed",
        "tts_len_predictor",
        "tts_dur_predictor",
        "asr_query_embed",
        "asr_cross_attn",
    ]

    for comp in components:
        bin_path = os.path.join(ckpt_dir, f"{comp}.bin")
        if os.path.exists(bin_path):
            sd = torch.load(bin_path, map_location="cpu")
            if comp == "soa_embed":
                if "weight" in sd:
                    sd = sd["weight"]
                model.soa_embed.data = sd
            else:
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                getattr(model, comp).load_state_dict(sd, strict=True)
                logger.info(f"    ✅ Loaded {comp}")
        else:
            logger.warning(f"    ⚠️ {comp}.bin not found! Using random init.")

    model.to(device)

    use_bf16 = cfg.training.get("bf16", False) or model.llm.dtype == torch.bfloat16
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_bf16_supported()) else torch.float32

    model.llm.to(dtype)
    model.input_proj.to(dtype)
    model.tts_flow_head.to(dtype)
    model.asr_flow_head.to(dtype)
    if hasattr(model, "tts_len_predictor"):
        model.tts_len_predictor.to(dtype)
    if hasattr(model, "tts_dur_predictor"):
        model.tts_dur_predictor.to(dtype)
    if hasattr(model, "asr_query_embed"):
        model.asr_query_embed.to(dtype)
    model.soa_embed.data = model.soa_embed.data.to(dtype)
    model.vae.to(torch.float32)

    model.eval()
    return model


# ==============================================================================
# ASR Inference
# ==============================================================================
@torch.no_grad()
def run_asr_inference_flow(model, tokenizer, latent_path, device, steps=20):
    if not os.path.exists(latent_path):
        return ""

    payload = torch.load(latent_path, map_location="cpu")
    audio = payload.get("latent", payload) if isinstance(payload, dict) else payload

    if audio.dim() == 2:
        if audio.shape[0] in (64, 80, 128, 192):
            audio = audio.transpose(0, 1)
        audio = audio.unsqueeze(0)
    audio = audio.to(device).to(model.llm.dtype)

    audio_embeds = model.input_proj(audio)  # [1, T_aud, D_llm]
    T_aud = audio_embeds.shape[1]

    soa = model.soa_embed.expand(1, -1, -1)
    prompt_txt = "<|im_start|>user\nTranscribe audio to text embedding.<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(prompt_txt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_embeds = model.get_input_embeddings()(prompt_ids)

    inp = torch.cat([audio_embeds, soa, prompt_embeds], dim=1)
    out = model.llm(inputs_embeds=inp, output_hidden_states=True)

    audio_context = out.hidden_states[-1][:, :T_aud, :]

    # 合理的文本长度上限：音频帧数的 1/4，至少 10，且不超过模型上限
    max_infer_len = min(max(T_aud // 4, 10), model.config.max_text_len)

    pos_ids = torch.arange(max_infer_len, device=device).unsqueeze(0)
    pos_ids = pos_ids.clamp(max=model.asr_query_embed.num_embeddings - 1)
    query_embeds = model.asr_query_embed(pos_ids)

    audio_mask = torch.ones(1, T_aud, device=device, dtype=torch.bool)
    key_padding_mask = ~audio_mask

    attn_out, _ = model.asr_cross_attn(
        query=query_embeds,
        key=audio_context,
        value=audio_context,
        key_padding_mask=key_padding_mask,
    )
    condition = attn_out

    text_dim = model.llm.config.hidden_size
    x_init = torch.randn(1, max_infer_len, text_dim, device=device, dtype=model.llm.dtype)

    x_final = ode_solve_sway(
        model.asr_flow_head,
        condition,
        x_init,
        steps,
        cfg_scale=1.0,
        device=device,
    )

    token_ids = model.search_nearest_tokens(x_final)[0]

    eos_candidates = set()
    if tokenizer.eos_token_id is not None:
        eos_candidates.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, "eod_id") and tokenizer.eod_id is not None:
        eos_candidates.add(tokenizer.eod_id)
    eos_candidates.add(151643)
    eos_candidates.add(151645)

    trunc_idx = len(token_ids)
    token_list = token_ids.tolist()
    for i, tid in enumerate(token_list):
        if tid in eos_candidates:
            trunc_idx = i
            break
    final_ids = token_ids[:trunc_idx]

    return tokenizer.decode(final_ids, skip_special_tokens=True)

def eval_task_asr(cfg, model, tokenizer, data):
    console.print("[bold green]>>> Running ASR Eval[/bold green]")
    out_path = os.path.join(cfg.evaluation.output_dir, "asr_results.csv")
    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)

    csv_file = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "ref", "pred", "wer", "cer"])

    preds, refs = [], []
    for i, item in enumerate(tqdm(data)):
        try:
            latent_path = item.get("latent_path") or item.get("file_path")
            pred = run_asr_inference_flow(model, tokenizer, latent_path, cfg.device)

            ref_norm, pred_norm = _normalize(item["text"]), _normalize(pred)
            if len(ref_norm) == 0:
                ref_norm = "<empty>"

            wer = wer_metric.compute(predictions=[pred_norm], references=[ref_norm])
            cer = cer_metric.compute(predictions=[pred_norm], references=[ref_norm])

            preds.append(pred_norm)
            refs.append(ref_norm)
            writer.writerow([i, ref_norm, pred_norm, wer, cer])
        except Exception as e:
            logger.error(f"Err {i}: {e}")

    if preds:
        console.print(f"✅ ASR WER: {wer_metric.compute(predictions=preds, references=refs):.2%}")
        console.print(f"✅ ASR CER: {cer_metric.compute(predictions=preds, references=refs):.2%}")


# ==============================================================================
# TTS Logic (NAR + Sway)
# ==============================================================================
@torch.no_grad()
def run_tts_inference(model, tokenizer, vocoder, text, steps=50, cfg_scale=2.5, device="cuda", max_audio_len=384):
    """
    NAR TTS inference with Flow head + VAE + Vocoder.
    """
    prompt = f"<|im_start|>user\nRead this text:\n{text}<|im_end|>\n<|im_start|>assistant\n"
    out_tok = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    text_ids = out_tok.input_ids.to(device)
    attention_mask = out_tok.attention_mask.to(device)
    text_embeds = model.get_input_embeddings()(text_ids)
    soa = model.soa_embed.expand(1, -1, -1)

    full_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1)
    pos_ids = full_mask.long().cumsum(-1) - 1
    pos_ids.masked_fill_(full_mask == 0, 1)

    out = model.llm(
        inputs_embeds=torch.cat([text_embeds, soa], dim=1),
        attention_mask=full_mask,
        position_ids=pos_ids,
        output_hidden_states=True,
    )

    hidden = out.hidden_states[-1]
    condition_vec = hidden[:, -1:, :]    # Global SOA
    text_context = hidden[:, :-1, :]     # Local text features
    text_ctx_mask = (full_mask[:, :-1] == 0)
    valid_mask = ~text_ctx_mask

    # Length prediction
    valid_len = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(text_context.dtype)
    masked_text = text_context * valid_mask.unsqueeze(-1).to(text_context.dtype)
    text_mean = masked_text.sum(dim=1) / valid_len
    pred_dtype = model.tts_len_predictor[0].weight.dtype
    text_mean = text_mean.to(pred_dtype)
    len_pred = model.tts_len_predictor(text_mean).squeeze(-1)

    text_len = valid_mask.sum(dim=1).float()
    min_frames = max(10, int(text_len.item() * 2))
    max_frames = min(max_audio_len, int(text_len.item() * 12))
    len_pred = len_pred.clamp(min=float(min_frames), max=float(max_frames))
    num_frames = int(len_pred.item())

    # Duration prediction and alignment
    B, T_txt, D = text_context.shape
    T_aud = num_frames

    pred_dtype = model.tts_dur_predictor[0].weight.dtype
    dur_raw = model.tts_dur_predictor(text_context.to(pred_dtype)).squeeze(-1)
    dur_pred = F.softplus(dur_raw) + 1e-4
    dur_pred = dur_pred.masked_fill(text_ctx_mask, 0)

    dur_sum = dur_pred.sum(dim=1, keepdim=True).clamp(min=1e-4)
    dur_scaled = dur_pred * (T_aud / dur_sum)
    dur_int = torch.floor(dur_scaled).long()

    # 确保有效 token 至少为 1
    dur_int = torch.where(valid_mask, torch.clamp(dur_int, min=1), torch.zeros_like(dur_int))

    # 若总和超出预算，按比例缩减并重新保底
    current_sum = dur_int.sum(dim=1)
    if (current_sum > T_aud).any():
        scale_factor = T_aud / current_sum.float().clamp(min=1)
        dur_int = (dur_int.float() * scale_factor.unsqueeze(1)).long()
        dur_int = torch.where(valid_mask, torch.clamp(dur_int, min=1), torch.zeros_like(dur_int))

    # 重新计算 remain，保证非负
    remain = (T_aud - dur_int.sum(dim=1)).clamp(min=0)

    align = build_alignment_from_durations(dur_int, valid_mask, T_aud, device, text_context.dtype)

    aligned_text = torch.bmm(align.transpose(1, 2), text_context)
    condition = aligned_text + condition_vec.expand(-1, T_aud, -1)

    # Sampling with CFG
    x_init = torch.randn(1, T_aud, model.config.latent_dim,
                         device=device, dtype=model.llm.dtype)
    x_mask = torch.zeros(1, T_aud, device=device, dtype=torch.bool)
    latents = ode_solve_sway(
        model.tts_flow_head,
        condition=condition,
        x_start=x_init,
        steps=steps,
        cfg_scale=cfg_scale,
        device=device,
        context=text_context,
        context_mask=text_ctx_mask,
        x_mask=x_mask,
    )

    # Denormalize latent
    latent_mean = getattr(model.config, "latent_mean", 0.0)
    latent_std = getattr(model.config, "latent_std", 1.0)
    latent_mean = torch.as_tensor(latent_mean, device=latents.device, dtype=latents.dtype)
    latent_std = torch.as_tensor(latent_std, device=latents.device, dtype=latents.dtype)
    if latent_mean.ndim == 1:
        latent_mean = latent_mean.view(1, 1, -1)
    if latent_std.ndim == 1:
        latent_std = latent_std.view(1, 1, -1)
    latents_denorm = latents * latent_std + latent_mean

    # VAE decode
    latents_for_vae = latents_denorm.transpose(1, 2).float()  # [B, D, T]
    mel_hat = model.vae.decode(latents_for_vae)

    # Denormalize mel using global stats
    mel_hat = mel_hat * model.config.mel_std + model.config.mel_mean

    print(f"Latent stats - mean: {latents_denorm.mean():.4f}, std: {latents_denorm.std():.4f}")
    print(f"Mel stats - mean: {mel_hat.mean():.4f}, std: {mel_hat.std():.4f}")

    return vocoder.decode(mel_hat).cpu()


def eval_task_tts(cfg, model, tokenizer, vocoder, data):
    console.print("[bold green]>>> Running TTS Eval[/bold green]")
    wav_dir = os.path.join(cfg.evaluation.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    try:
        asr_pipe = pipeline(
           "automatic-speech-recognition",
           model="openai/whisper-tiny.en",
           device=cfg.device,
       )
    except Exception:
        asr_pipe = None
        console.print("[yellow]⚠️ Whisper not available, skipping TTS WER/CER.[/yellow]")

    max_audio_len = cfg.data.max_audio_len
    steps = cfg.evaluation.get("steps", 50)
    cfg_scale = cfg.evaluation.get("cfg_scale", 2.5)

    wers, cers = [], []
    for i, item in enumerate(tqdm(data)):
        try:
            wav = run_tts_inference(
                model, tokenizer, vocoder, item["text"],
                steps=steps, cfg_scale=cfg_scale, device=cfg.device,
                max_audio_len=max_audio_len
            )
            path = os.path.join(wav_dir, f"{i}.wav")
            wave = wav.unsqueeze(0) if wav.dim() == 1 else wav  # [1, T]
            torchaudio.save(path, wave, 16000)

            if asr_pipe:
                pred = asr_pipe(path)["text"]
                ref_norm = _normalize(item["text"])
                pred_norm = _normalize(pred)
                wers.append(wer_metric.compute(predictions=[pred_norm], references=[ref_norm]))
                cers.append(cer_metric.compute(predictions=[pred_norm], references=[ref_norm]))
        except Exception as e:
            logger.error(f"Err {i}: {e}")

    if wers:
        console.print(f"✅ TTS WER: {sum(wers)/len(wers):.2%}")
    if cers:
        console.print(f"✅ TTS CER: {sum(cers)/len(cers):.2%}")


# ==============================================================================
# Main
# ==============================================================================
@hydra.main(version_base=None, config_path="../config", config_name="calm_config")
def main(cfg: DictConfig):
    if not hasattr(cfg, "device"):
        dev = 0 if torch.cuda.is_available() else "cpu"
        with open_dict(cfg):
            cfg.device = dev

    set_seed(cfg.evaluation.get("seed", 42))
    tokenizer = model = None

    if cfg.evaluation.task in ["asr", "mix", "tts"]:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
        model = load_model(cfg, cfg.device)

    asr_data, tts_data = load_eval_datasets(cfg)

    if cfg.evaluation.task in ["asr", "mix"] and asr_data:
        eval_task_asr(cfg, model, tokenizer, asr_data)

    if cfg.evaluation.task in ["tts", "mix"] and tts_data:
        vocoder = Vocoder(cfg.device)
        eval_task_tts(cfg, model, tokenizer, vocoder, tts_data)


if __name__ == "__main__":
    main()