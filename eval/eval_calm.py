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
import matplotlib.pyplot as plt
from transformers import pipeline

# --- Environment Patches ---
# 修复部分环境中 torchaudio 后端检测的问题
if not hasattr(torchaudio, "list_audio_backends"):
    try:
        import torchaudio.backend
        torchaudio.list_audio_backends = getattr(torchaudio.backend, "list_audio_backends", lambda: ["soundfile"])
    except ImportError:
        torchaudio.list_audio_backends = lambda: []

sys.path.append(os.getcwd())
# 【对应关系】：导入 modeling_calm.py 中的模型定义
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
    """
    功能：加载测试集数据 (.jsonl 格式)。
    """
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
    """
    功能：声码器，负责将 Mel 频谱转换为波形。
    
    【对应关系】：
    - 输入：来自 QwenCALM.vae.decode() 输出的 Mel 频谱。
    - 关键逻辑：处理 Log-Mel (VAE输出) 到 Log10-Mel (HiFi-GAN输入) 的转换。
    """
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

        # Griffin-Lim 作为备选方案
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
        """
        功能：执行解码。
        """
        mel = mel.to(self.device).to(torch.float32)
        
        # 1. 维度统一: [B, 80, T]
        if mel.dim() == 2: mel = mel.unsqueeze(0)
        if mel.shape[-1] == 80: mel = mel.transpose(1, 2)

        # 2. HiFi-GAN 解码
        if self.hifi is not None:
            # [CRITICAL FIX] 恢复缩放逻辑
            # VAE 输出是 Log (ln) Mel，HiFiGAN 需要 Log10 Mel
            # 关系: ln(x) = ln(10) * log10(x) => log10(x) ≈ ln(x) * 0.43429
            # 如果不缩放，能量会过大导致破音
            mel_log10 = mel * 0.43429
            
            try: return self.hifi.decode_batch(mel_log10.transpose(1, 2)).squeeze(1)
            except: 
                try: return self.hifi.decode_batch(mel_log10).squeeze(1)
                except: pass

        # 3. Fallback: Griffin-Lim (需要 Linear Mel)
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
    """
    功能：加载完整的 QwenCALM 模型用于推理。
    
    【对应关系】：
    - 加载 Config: 对应 config/calm_config.yaml
    - 加载 Base Model: Qwen2
    - 加载 Adapter: 对应 train_calm.py 保存的 LoRA
    - 加载 Projector/Head/SOA: 对应 train_calm.py 手动保存的 .bin 文件
    """
    logger.info(f"🤖 Loading Model Base: {cfg.model.qwen_path}")
    
    config = QwenCALMConfig(
        qwen_path=cfg.model.qwen_path, 
        vae_path=cfg.model.vae_path, 
        latent_dim=cfg.model.latent_dim,
        flow_hidden_dim=cfg.model.get("flow_hidden_dim", 2048), 
        flow_num_layers=cfg.model.get("flow_num_layers", 4),
        use_precomputed_latents=False 
    )
    
    # 1. 初始化模型结构
    model = QwenCALM(config)
    
    ckpt_dir = cfg.evaluation.checkpoint_path
    logger.info(f"📂 Loading Checkpoints from: {ckpt_dir}")

    # 2. 加载 LLM Adapters (LoRA)
    # 尝试加载 ASR 或 TTS Adapter
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
        # Fallback: 根目录下单个 Adapter
        if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
            logger.info("  - Loading Single LoRA...")
            model.llm = PeftModel.from_pretrained(model.llm, ckpt_dir)

    # 3. 加载 Projectors (Input/Output)
    for component in ["input_proj", "output_head"]:
        bin_path = os.path.join(ckpt_dir, f"{component}.bin")
        if os.path.exists(bin_path):
            logger.info(f"  - Loading {component}...")
            state_dict = torch.load(bin_path, map_location="cpu")
            # 修复 DDP 保存时可能带有的 module. 前缀
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            getattr(model, component).load_state_dict(state_dict)
        else:
            logger.warning(f"  ⚠️  {component}.bin not found! Model may not work.")

    # 4. 加载 SOA Embed
    soa_path = os.path.join(ckpt_dir, "soa_embed.bin")
    if os.path.exists(soa_path):
        logger.info(f"  - Loading soa_embed...")
        soa_data = torch.load(soa_path, map_location="cpu")
        
        # 兼容处理：支持 dict 或直接 tensor
        tensor_data = soa_data
        if isinstance(soa_data, dict):
            key = next((k for k in ["weight", "soa_embed"] if k in soa_data), None)
            if key:
                tensor_data = soa_data[key]
            else:
                tensor_data = list(soa_data.values())[0]
        
        # 精度对齐
        if cfg.training.get("bf16", False) and tensor_data.dtype != torch.bfloat16:
             tensor_data = tensor_data.to(torch.bfloat16)
             
        model.soa_embed.data = tensor_data
    else:
        logger.warning(f"  ⚠️  soa_embed.bin not found! TTS will produce noise.")

    model.to(device).eval()
    
    # 5. 混合精度设置
    if cfg.training.get("bf16", False): 
        logger.info("  - Converting to bfloat16 (VAE remains fp32)")
        model.to(torch.bfloat16)
        model.vae.to(torch.float32) # VAE 保持 FP32 以保证音质
        
    return model

# ==============================================================================
# 3. ASR Inference Logic
# ==============================================================================
@torch.no_grad()
def run_asr_inference(model, tokenizer, latent_path, device):
    """
    功能：ASR 推理。
    """
    # 切换 Adapter
    if hasattr(model.llm, "set_adapter") and hasattr(model.llm, "peft_config"):
        if "asr" in model.llm.peft_config:
            model.llm.set_adapter("asr")

    # 1. 加载音频 Latent
    if not os.path.exists(latent_path): return ""
    payload = torch.load(latent_path, map_location="cpu")
    audio = payload.get("latent", payload) if isinstance(payload, dict) else payload
    
    # [T, D] -> [1, T, D]
    if audio.dim() == 2:
        if audio.shape[0] == 64: audio = audio.transpose(0, 1) 
        audio = audio.unsqueeze(0) 
    
    audio = audio.to(device).to(model.llm.dtype)
    
    # 2. 投影音频特征 (Projector)
    # 【对应关系】：调用 modeling_calm.py 中 AudioInputProjector
    # offset=0 表示从头开始编码
    audio_embeds = model.input_proj(audio, offset=0) 

    # 3. 构建 Prompt
    prompt = "Transcribe the audio content into text."
    prefix_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prefix_embeds = model.get_input_embeddings()(prefix_ids)

    # 4. 拼接并生成
    inputs_embeds = torch.cat([audio_embeds, prefix_embeds], dim=1)

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
    """
    功能：ASR 任务评估循环，计算 WER。
    """
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
    功能：执行一步流匹配 (Flow Matching) 生成。
    作用：从高斯噪声出发，根据 condition 预测速度场，通过 Euler 积分推进一步，生成一帧音频 Latent。
    
    【对应关系】：
    - 调用 `model.output_head` (modeling_calm.py)。
    """
    # 1. 初始化噪声 x0 ~ N(0, 1)
    noise = torch.randn(1, 1, model.config.latent_dim, device=device, dtype=model.llm.dtype)
    dt = 1.0 / steps
    x = noise
    
    # 2. ODE 积分循环
    for i in range(steps):
        t = torch.full((1,), i/steps, device=device, dtype=x.dtype)
        
        # [FIXED] 调用 Flow Head
        # v_cond: 有条件生成
        v_cond = model.output_head(condition, x, t)
        # v_uncond: 无条件生成 (输入全零 Condition)
        v_uncond = model.output_head(torch.zeros_like(condition), x, t)
        
        # 3. CFG (Classifier-Free Guidance) 引导
        # 公式: v = v_uncond + scale * (v_cond - v_uncond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # 4. 更新 x
        x = x + v * dt
        
    return x

@torch.no_grad()
def run_tts_inference(model, tokenizer, vocoder, text, steps=10, cfg_scale=1.0, device="cuda", save_plot_path=None):
    """
    功能：TTS 推理主函数 (自回归生成 + 智能停止)。
    """
    # 切换 Adapter
    if hasattr(model.llm, "set_adapter") and hasattr(model.llm, "peft_config"):
        if "tts" in model.llm.peft_config:
            model.llm.set_adapter("tts")

    # 1. 准备 Text Prompt
    prompt = f"Read this text:\n{text}"
    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    text_embeds = model.get_input_embeddings()(text_ids)
    
    # 2. 添加 SOA Token
    soa_token = model.soa_embed.expand(1, -1, -1) 
    inputs_embeds = torch.cat([text_embeds, soa_token], dim=1)
    
    # 3. 运行 LLM 预填充
    out = model.llm(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None, output_hidden_states=True)
    past_kv = out.past_key_values
    
    # 获取第一个 Condition
    condition = out.hidden_states[-1][:, -1:, :] 
    
    # 4. 生成第一帧
    curr_latent = generate_one_step_flow(model, condition, steps, cfg_scale, device)
    history_latents = [curr_latent]

    # 5. 自回归循环 (带刹车机制)
    # 理论最大长度: 假设 10 秒 = 156 帧, 250 帧足够了
    max_frames = 250 
    pbar = tqdm(range(max_frames), desc="Gen Audio", leave=False)
    
    # 获取 EOS Token ID
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None: 
        eos_token_id = 151645 # Qwen 默认 EOS

    stop_reason = "max_length" # 记录停止原因

    for i in pbar:
        # 输入: 当前生成的 Latent
        input_latent = curr_latent
        curr_embeds = model.input_proj(input_latent, offset=i)
        
        # LLM Step
        # 注意：这里我们同时需要 hidden_states (给 Flow 用) 和 logits (给停止检测用)
        out = model.llm(inputs_embeds=curr_embeds, use_cache=True, past_key_values=past_kv, output_hidden_states=True)
        past_kv = out.past_key_values
        
        # A. 获取 Condition 给 Flow Head
        condition = out.hidden_states[-1][:, -1:, :]
        
        # B. [新增] 停止检测逻辑 (Stop Token Detection)
        # 获取 LM Head 的预测结果
        logits = out.logits[:, -1, :] # [1, Vocab]
        pred_token_id = torch.argmax(logits, dim=-1).item()
        
        # 这里的逻辑是：如果 LLM 预测下一个词是 EOS，说明它觉得音频该结束了
        if pred_token_id == eos_token_id:
            stop_reason = "eos_token"
            # console.print(f"[yellow]🛑 Stop Token Detected at step {i}[/yellow]")
            break
            
        # C. [可选] 静音检测 (Silence Detection) 作为双保险
        # 如果 Latent 的能量极低且已经生成了一定长度，也可以停
        # (需要根据你的 Latent 统计特性调整阈值，比如 0.05)
        latent_energy = torch.mean(torch.abs(input_latent)).item()
        if i > 50 and latent_energy < 0.05:
            stop_reason = "silence"
            # console.print(f"[yellow]🛑 Silence Detected at step {i}[/yellow]")
            break

        # Flow 生成下一帧
        curr_latent = generate_one_step_flow(model, condition, steps, cfg_scale, device)
        history_latents.append(curr_latent)

    # console.print(f"Generated {len(history_latents)} frames. Reason: {stop_reason}")

    # 6. 解码为波形
    latents = torch.cat(history_latents, dim=1).transpose(1, 2).to(torch.float32) 
    mel = model.vae.decode(latents)
    
    if save_plot_path:
        mel_cpu = mel.squeeze().float().cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_cpu, aspect='auto', origin='lower', interpolation='none')
        plt.colorbar()
        plt.title(f"Generated Mel (Text: {text[:20]}...) [Stop: {stop_reason}]")
        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close()
    
    wav = vocoder.decode(mel)
    return wav.cpu()

def eval_task_tts(cfg, model, tokenizer, vocoder, data):
    """
    功能：TTS 任务评估循环，生成音频 -> ASR 转录 -> 计算 WER。
    """
    wav_dir = os.path.join(cfg.evaluation.output_dir, "generated_wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    # [修改] CSV Header 增加 metrics
    csv_file = open(os.path.join(cfg.evaluation.output_dir, "tts_results.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["id", "text_ref", "text_pred", "wer", "wav_path"])
    
    console.print(f"[bold green]>>> Running TTS Evaluation (Steps={cfg.evaluation.flow_steps})[/bold green]")
    img_dir = os.path.join(cfg.evaluation.output_dir, "mel_plots")
    os.makedirs(img_dir, exist_ok=True)

    # [新增] 加载评估用的 ASR 模型 (Whisper)
    # 建议使用 whisper-small.en 或 whisper-base.en，速度快且精度够用
    asr_model_id = cfg.evaluation.get("eval_asr_model", "openai/whisper-tiny.en")
    console.print(f"[bold yellow]Loading ASR Evaluator: {asr_model_id}...[/bold yellow]")
    asr_pipe = pipeline(
        "automatic-speech-recognition", 
        model=asr_model_id, 
        device=cfg.device
    )

    # 文本标准化器 (移除标点，统一小写)
    import re
    def normalize(text): return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    wers = []

    for i, item in enumerate(data):
        text_ref = item.get("text", "")
        if not text_ref: continue
        
        try:
            scale = cfg.evaluation.get("cfg_scale", 1.0)
            steps = cfg.evaluation.get("flow_steps", 10)
            
            # 1. 生成音频
            wav = run_tts_inference(
                model, tokenizer, vocoder, text_ref, 
                steps=steps, 
                cfg_scale=scale, 
                device=cfg.device,
                save_plot_path=os.path.join(img_dir, f"mel_{i}.png")
            )
            
            wav_np = wav.squeeze().numpy()
            save_path = os.path.join(wav_dir, f"sample_{i}.wav")
            sf.write(save_path, wav_np, 16000)
            
            # 2. [新增] ASR 转录 (把生成的音频转回文字)
            # Whisper 需要 numpy array
            transcription = asr_pipe(wav_np)["text"]
            
            # 3. [新增] 计算 WER
            norm_ref = normalize(text_ref)
            norm_pred = normalize(transcription)
            
            if len(norm_ref) > 0:
                wer = wer_metric.compute(predictions=[norm_pred], references=[norm_ref])
            else:
                wer = 1.0
                
            wers.append(wer)
            
            # 4. 写入 CSV
            writer.writerow([i, text_ref, transcription, f"{wer:.4f}", save_path])
            
            if (i+1) % 5 == 0: 
                avg_wer = sum(wers) / len(wers)
                console.print(f"[Sample {i+1}] Avg WER: {avg_wer:.2%} | Ref: {text_ref[:20]}... | Pred: {transcription[:20]}...")
                
        except Exception as e:
            logger.error(f"Error sample {i}: {e}")

    csv_file.close()
    
    if len(wers) > 0:
        final_wer = sum(wers) / len(wers)
        console.print(f"[bold blue]✅ Final TTS WER: {final_wer:.2%}[/bold blue]")

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
    
    # 1. 加载模型
    model = load_model(cfg, cfg.device)
    
    # 2. 加载数据
    data = load_dataset_jsonl(cfg.evaluation.test_file, cfg.evaluation.max_samples)
    
    # 3. 任务分发
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