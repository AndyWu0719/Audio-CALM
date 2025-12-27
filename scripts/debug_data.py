import os
import torch
import soundfile as sf
import torchaudio
import glob
import random
from models.modeling_vae import AcousticVAE, AudioVAEConfig

# ================= 配置区 =================
# 1. 指向你的 VAE Checkpoint
VAE_PATH = "outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim/checkpoint-6900" 
# 2. 指向你的 Latent 文件夹 (训练数据)
LATENT_DIR = "data/latents/train/LibriTTS_R" 
# ==========================================

class SimpleVocoder:
    def __init__(self, device="cuda"):
        self.device = device
        # 严格匹配你的预处理参数
        self.n_fft = 1024
        self.n_mels = 80
        self.sr = 16000
        
        # 1. Mel Basis [80, 513] -> Pinv -> [513, 80]
        mel_fb = torchaudio.transforms.MelScale(
            n_mels=80, sample_rate=16000, f_min=0, f_max=8000, 
            n_stft=513, norm="slaney", mel_scale="slaney"
        ).to(device).fb
        self.inv_mel_basis = torch.linalg.pinv(mel_fb)
        
        # 2. GL
        self.gl = torchaudio.transforms.GriffinLim(
            n_fft=1024, n_iter=60, win_length=1024, hop_length=256, power=1.0
        ).to(device)

    def decode(self, mel):
        # mel: [B, 80, T]
        mel = mel.to(self.device).float()
        energy = torch.exp(mel)
        # [B, 80, T] -> [B, T, 80] @ [80, 513] -> [B, T, 513]
        linear = torch.matmul(energy.transpose(1, 2), self.inv_mel_basis)
        mag = torch.sqrt(torch.clamp(linear.transpose(1, 2), min=1e-8))
        wav = self.gl(mag)
        return wav.squeeze().cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load VAE
    print(f"Loading VAE from {VAE_PATH}...")
    try:
        vae = AcousticVAE.from_pretrained(VAE_PATH).to(device)
    except:
        print("Loading via config/state_dict...")
        config = AudioVAEConfig()
        vae = AcousticVAE(config).to(device)
        sd = torch.load(os.path.join(VAE_PATH, "pytorch_model.bin"), map_location='cpu')
        vae.load_state_dict(sd)
    vae.eval()

    # 2. Find a Latent File
    files = glob.glob(os.path.join(LATENT_DIR, "**", "*.pt"), recursive=True)
    if not files:
        print(f"❌ No .pt files found in {LATENT_DIR}")
        return
    
    # 随机选一个文件
    target_file = random.choice(files)
    print(f"🕵️ Inspecting File: {target_file}")

    # 3. Load & Analyze Latent
    data = torch.load(target_file, map_location='cpu')
    # 兼容不同的保存格式 (有些可能是 dict, 有些直接是 tensor)
    latent = data.get("latent", data) if isinstance(data, dict) else data
    
    print(f"📊 Latent Raw Shape: {latent.shape}")
    print(f"   Stats: Mean={latent.mean():.4f}, Std={latent.std():.4f}, Min={latent.min():.4f}, Max={latent.max():.4f}")

    # 4. Shape Correction (关键步骤: 维度猜谜)
    # VAE Decode 需要 [B, 64, T]
    latent = latent.to(device)
    if latent.dim() == 2:
        # 可能是 [T, 64] 或 [64, T]
        if latent.shape[0] == 64: 
            print("   -> Detected [64, T], unsqueezing...")
            latent = latent.unsqueeze(0) # [1, 64, T]
        elif latent.shape[1] == 64:
            print("   -> Detected [T, 64], transposing...")
            latent = latent.transpose(0, 1).unsqueeze(0) # [1, 64, T]
    
    # 5. Decode
    vocoder = SimpleVocoder(device)
    with torch.no_grad():
        mel = vae.decode(latent)
        print(f"📊 Reconstructed Mel Stats: Min={mel.min():.4f}, Max={mel.max():.4f}")
        wav = vocoder.decode(mel)

    # 6. Save
    out_path = "debug_reconstruction.wav"
    sf.write(out_path, wav, 16000)
    print(f"✅ Saved reconstruction to: {out_path}")
    print("👉 Please listen to this file strictly!")

if __name__ == "__main__":
    main()