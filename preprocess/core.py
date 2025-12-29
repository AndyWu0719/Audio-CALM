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

# 【对应关系】：导入 models 文件夹下的 VAE 模型定义，用于加载权重
from models.modeling_vae import AcousticVAE

class MelExtractor(nn.Module):
    """
    标准化的 Mel 频谱提取器。
    确保训练（Training）和预处理（Preprocessing）阶段的特征提取逻辑一致。
    
    【文件间关系】：
    - 被 `process_dataset.py` 调用：将 wav 转换为 mel，以便输入 VAE。
    - 被 `train_vae.py` 隐式使用：VAE 训练时的数据加载器也应遵循此标准。
    - 被 `scripts/check_pt.py` 调用：用于诊断时的特征提取对比。
    """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        # 1. 定义 Mel 频谱变换
        # 关键参数: 采样率 16k, 80个 Mel 频段
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
        """
        功能：将波形转换为对数 Mel 频谱 (Log-Mel Spectrogram)。
        """
        # 1. 提取 Mel 频谱 [B, n_mels, T]
        mel = self.mel_transform(wav)
        
        # 2. 对数缩放 (Log-Scaling)
        # 采用自然对数 (ln)，并进行截断防止 log(0)。
        # 【重要】：这决定了 VAE 输入数据的数值分布（最小值约 -11.5）。
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

def load_vae(ckpt_path, device):
    """
    功能：加载预训练的 VAE 模型，用于特征提取。
    
    【文件间关系】：
    - 被 `process_dataset.py` 调用：获取 VAE 的编码器部分来生成 Latent。
    """
    # 使用 contextlib 屏蔽 VAE 加载时的冗余日志输出
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        try:
            # 1. 尝试使用 HuggingFace 风格的 `from_pretrained` 加载
            vae = AcousticVAE.from_pretrained(ckpt_path)
        except Exception:
            # 2. 兜底逻辑：手动加载 state_dict
            # 如果 checkpoint 只是一个 .bin 文件而没有 config.json，则走此逻辑
            from models.modeling_vae import AudioVAEConfig
            config = AudioVAEConfig()
            vae = AcousticVAE(config)
            if os.path.isdir(ckpt_path):
                ckpt_file = os.path.join(ckpt_path, "pytorch_model.bin")
            else:
                ckpt_file = ckpt_path
            state_dict = torch.load(ckpt_file, map_location='cpu')
            vae.load_state_dict(state_dict, strict=False)
    
    # 3. 设置为 eval 模式 (对 BatchNorm/Dropout 很重要)
    vae.to(device)
    vae.eval()
    return vae

def process_audio_chunk(wav, target_sr=16000):
    """
    功能：标准化原始音频张量（单声道混合 + 归一化）。
    
    【文件间关系】：
    - 被 `process_dataset.py` 在加载音频后立即调用。
    """
    # 1. 混音：立体声转单声道
    # VAE 期望输入是单声道的。
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # 2. 音量归一化
    # 将音频幅值缩放到 [-0.95, 0.95] 范围。
    # 这确保了整个数据集的 Mel 能量水平一致，对 VAE 稳定性至关重要。
    peak = torch.max(torch.abs(wav))
    if peak > 0:
        wav = wav / (peak + 1e-8) * 0.95
        
    return wav