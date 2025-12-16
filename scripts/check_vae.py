# 快速检查：VAE encode->decode 的 mel MSE
import torch
import sys
import os
sys.path.append(os.getcwd())
from models.modeling_vae import AcousticVAE

vae = AcousticVAE.from_pretrained("outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim/checkpoint-6900")
vae.eval()

mel = torch.load("/data0/determined/users/andywu/Audio-CALM-v2/data/mel_features/dev-clean/84-121123-0007.pt").unsqueeze(0)  # [1, 80, T]
with torch.no_grad():
    out = vae(mel)
    print(f"Reconstruction MSE: {out['rec_loss'].item():.4f}")
    print(f"KL Loss: {out['kl_loss'].item():.4f}")