import os
import glob
import argparse
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import sys
import math

# === 引入你的 VAE 模型 ===
sys.path.append(os.getcwd()) 
from models.modeling_vae import AcousticVAE 

# ==============================================================================
# 1. 复用你之前的 MelExtractor 配置
# ==============================================================================
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256 

class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
            normalized=False,
            f_min=0,
            f_max=8000,
            norm="slaney",
            mel_scale="slaney"
        )
    
    def forward(self, wav):
        # wav: [B, T] or [1, T]
        mel = self.mel_transform(wav)
        # Log-Mel Scaling (和你训练 VAE 时保持一致)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

# ==============================================================================
# 2. VAE 加载与处理逻辑
# ==============================================================================
def load_vae(ckpt_path, device):
    print(f"🔄 Loading VAE from: {ckpt_path}")
    try:
        # 尝试标准加载
        vae = AcousticVAE.from_pretrained(ckpt_path)
    except Exception as e:
        print(f"⚠️ Standard load failed: {e}. Trying state_dict load...")
        # 如果你存的是整个对象或者只是权重，这里提供一个 fallback
        # 假设你有一个 config.json 在同目录下，或者硬编码 Config
        from models.modeling_vae import AudioVAEConfig
        config = AudioVAEConfig() # 使用默认配置，或者你需要根据你的训练修改参数
        vae = AcousticVAE(config)
        
        # 尝试加载权重 (如果是 .bin 或 .pt)
        if os.path.isdir(ckpt_path):
            ckpt_file = os.path.join(ckpt_path, "pytorch_model.bin")
        else:
            ckpt_file = ckpt_path
            
        state_dict = torch.load(ckpt_file, map_location='cpu')
        vae.load_state_dict(state_dict, strict=False)

    vae.to(device)
    vae.eval()
    return vae

def process_file(vae, mel_extractor, wav_path, out_root, in_root, device):
    try:
        # 1. 路径计算
        rel_path = os.path.relpath(os.path.dirname(wav_path), in_root)
        save_dir = os.path.join(out_root, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        file_id = os.path.splitext(os.path.basename(wav_path))[0]
        save_path = os.path.join(save_dir, f"{file_id}.pt")
        
        if os.path.exists(save_path):
            return None, None

        # 2. 加载音频
        wav, sr = torchaudio.load(wav_path)
        
        # 转单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # 重采样
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)

        # 3. [关键] 波形归一化 (复用你之前的逻辑)
        # wav = wav / (torch.max(torch.abs(wav)) + 1e-8) * 0.95
        peak = torch.max(torch.abs(wav))
        if peak > 0:
            wav = wav / (peak + 1e-8) * 0.95

        wav = wav.to(device) # [1, T]

        # 4. 提取 Mel 频谱
        with torch.no_grad():
            mel = mel_extractor(wav) # 输出 [1, 80, T_mel]
            
            # 5. VAE Encode
            # VAE encode 通常返回 (mu, logvar) 或 dist
            # 你的代码: mu, logvar = self.encode(mel_padded)
            # 我们只需要 mu (Latent)
            
            # 你的 VAE 可能需要 pad 到 stride 的倍数，这里简单处理一下
            # 如果你的 VAE forward 里有 pad 逻辑，这里直接调 encode 可能会报错尺寸不匹配
            # 为了安全，我们手动 Pad 一下 (假设 total_stride 是 4 或 8)
            pad_to = 4 
            if mel.shape[-1] % pad_to != 0:
                pad_len = pad_to - (mel.shape[-1] % pad_to)
                mel = torch.nn.functional.pad(mel, (0, pad_len), mode='reflect')

            mu, _ = vae.encode(mel)
            latent = mu # [1, D, T_latent]

        # [1, D, T] -> [D, T] (64, T)
        latent = latent.squeeze(0).cpu()

        # 6. 保存
        torch.save(latent, save_path)

        # 7. 处理文本
        txt_path = wav_path.replace(".wav", ".normalized.txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            trans_file = os.path.join(save_dir, f"{os.path.basename(save_dir)}.trans.txt")
            return trans_file, f"{file_id} {text}"
            
    except Exception as e:
        print(f"\n❌ Error processing {wav_path}: {e}")
        return None, None
    
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="Input root (e.g. LibriTTS_R/train-clean-100)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # 初始化模型
    vae = load_vae(args.vae_ckpt, device)
    mel_extractor = MelExtractor().to(device)
    mel_extractor.eval()

    # 扫描文件
    print(f"🔍 Scanning .wav files in {args.in_dir}...")
    wav_files = glob.glob(os.path.join(args.in_dir, "**", "*.wav"), recursive=True)
    print(f"📂 Found {len(wav_files)} files.")

    # 处理循环
    trans_buffer = {} 

    for wav_path in tqdm(wav_files):
        trans_file, line = process_file(vae, mel_extractor, wav_path, args.out_dir, args.in_dir, device)
        if trans_file and line:
            if trans_file not in trans_buffer:
                trans_buffer[trans_file] = []
            trans_buffer[trans_file].append(line)

    # 写入 trans.txt
    print("📝 Writing transcription files...")
    for path, lines in trans_buffer.items():
        with open(path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")

    print(f"✅ Done! Latents saved to {args.out_dir}")

if __name__ == "__main__":
    main()