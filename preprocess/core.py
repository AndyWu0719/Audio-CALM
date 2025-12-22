# preprocess/core.py
import torch
import torch.nn as nn
import torchaudio
import contextlib
import os
import sys

# ==========================================================
# [FIX] 路径修复逻辑
# 无论脚本在哪里运行，都强制把 "项目根目录" 加入系统路径
# ==========================================================
# 获取当前脚本所在目录 (preprocess/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (Audio-CALM-v2/)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以安全导入 models 了
from models.modeling_vae import AcousticVAE

class MelExtractor(nn.Module):
    """
    Standardized Mel Spectrogram Extractor.
    Ensures consistency between Training and Preprocessing.
    """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
            f_min=0,
            f_max=8000,
            norm="slaney",
            mel_scale="slaney"
        )

    def forward(self, wav):
        # wav: [B, T]
        mel = self.mel_transform(wav)
        # Log-Mel scaling with clamping
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

def load_vae(ckpt_path, device):
    # 使用 contextlib 屏蔽 stdout
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        try:
            vae = AcousticVAE.from_pretrained(ckpt_path)
        except Exception:
            # Fallback logic...
            from models.modeling_vae import AudioVAEConfig
            config = AudioVAEConfig()
            vae = AcousticVAE(config)
            if os.path.isdir(ckpt_path):
                ckpt_file = os.path.join(ckpt_path, "pytorch_model.bin")
            else:
                ckpt_file = ckpt_path
            state_dict = torch.load(ckpt_file, map_location='cpu')
            vae.load_state_dict(state_dict, strict=False)
    
    vae.to(device)
    vae.eval()
    return vae

def process_audio_chunk(wav, target_sr=16000):
    """
    Standardize audio: Mix to Mono -> Normalize -> Check Length
    """
    # 1. Mix to Mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # 2. Normalize volume (Crucial for VAE stability)
    peak = torch.max(torch.abs(wav))
    if peak > 0:
        wav = wav / (peak + 1e-8) * 0.95
        
    return wav