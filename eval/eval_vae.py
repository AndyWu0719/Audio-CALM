import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

try:
    from speechbrain.inference.vocoders import HifiGAN
except ImportError:
    try:
        from speechbrain.inference.vocoders import HIFIGAN as HifiGAN
    except ImportError:
        try:
            from speechbrain.inference import HifiGAN
        except ImportError:
            import speechbrain.inference.vocoders as mod
            available = [x for x in dir(mod) if 'GAN' in x and 'Base' not in x]
            if available:
                HifiGAN = getattr(mod, available[0])
                print(f"[DEBUG] Auto-detected HifiGAN class: {available[0]}")
            else:
                raise ImportError("Could not find HifiGAN class in speechbrain.inference.vocoders!")

sys.path.append(os.getcwd())

from models.modeling_vae import AcousticVAE
from preprocess.prepare_mel import MelExtractor

def save_plot(orig_mel, recon_mel, save_path):
    orig_mel = orig_mel.cpu().numpy()
    recon_mel = recon_mel.cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    vmin = min(orig_mel.min(), recon_mel.min())
    vmax = max(orig_mel.max(), recon_mel.max())

    im1 = axes[0].imshow(orig_mel, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Mel Spectrogram")
    axes[0].set_ylabel("Mel Channels")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(recon_mel, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstructed Mel Spectrogram (VAE)")
    axes[1].set_ylabel("Mel Channels")
    axes[1].set_xlabel("Time Frames")
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VAE checkpoint folder")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to a raw .flac/.wav file for testing")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_results_hifigan", help="Where to save results")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading VAE from {args.checkpoint}...")
    vae_model = AcousticVAE.from_pretrained(args.checkpoint).to(device)
    vae_model.eval()

    print("Loading HiFi-GAN from SpeechBrain (pretrained on LibriTTS)...")
    hifi_gan = HifiGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz", 
        savedir="./outputs/eval/tmpdir_vocoder",
        run_opts={"device": device}
    )

    print(f"Processing audio: {args.audio_path}")
    wav, sr = torchaudio.load(args.audio_path)
    
    if sr != 16000:
        print(f"Resampling from {sr} to 16000 Hz...")
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    print("Applying Peak Normalization (0.95)...")
    wav = wav / (torch.max(torch.abs(wav)) + 1e-8) * 0.95
    wav = wav.to(device)
    
    extractor = MelExtractor().to(device)
    with torch.no_grad():
        gt_mel = extractor(wav) # [1, 80, T]
    
    print("Running VAE Inference...")
    with torch.no_grad():
        outputs = vae_model(gt_mel)
        recon_mel = outputs['recon_mel'] # [1, 80, T]

    mse = torch.nn.functional.mse_loss(recon_mel, gt_mel).item()
    
    # Cosine Similarity
    gt_flat = gt_mel.reshape(gt_mel.size(0), -1) 
    recon_flat = recon_mel.reshape(recon_mel.size(0), -1)
    cosine_sim = torch.nn.functional.cosine_similarity(gt_flat, recon_flat, dim=1).mean().item()

    print("-" * 30)
    print(f"MSE Loss:              {mse:.6f}")
    print(f"Cosine Similarity:     {cosine_sim:.4f}")
    print("-" * 30)

    plot_path = os.path.join(args.output_dir, "mel_comparison.png")
    save_plot(gt_mel.squeeze(0), recon_mel.squeeze(0), plot_path)
    print(f"Saved plot to {plot_path}")

    print("Reconstructing waveform using HiFi-GAN...")
    
    mel_input_recon = recon_mel
    mel_input_gt = gt_mel

    with torch.no_grad():
        wav_recon = hifi_gan.decode_batch(mel_input_recon)
        
        wav_oracle = hifi_gan.decode_batch(mel_input_gt)

    path_orig = os.path.join(args.output_dir, "1_original.wav")
    torchaudio.save(path_orig, wav.cpu(), 16000)
    
    path_recon = os.path.join(args.output_dir, "2_vae_reconstructed_hifigan.wav")
    torchaudio.save(path_recon, wav_recon.squeeze(0).cpu(), 16000)
    
    path_oracle = os.path.join(args.output_dir, "3_oracle_mel_hifigan.wav")
    torchaudio.save(path_oracle, wav_oracle.squeeze(0).cpu(), 16000)

    print("\n=== 听觉测试指南 ===")
    print(f"1. 听 '{path_oracle}' (Oracle):")
    print("   - 如果这个文件也有杂音/变声，说明你的 MelExtractor 参数(n_fft, hop_len)与 SpeechBrain 的 HiFi-GAN 不匹配。")
    print("   - 此时需要调整你的 prepare_mel.py 参数以匹配 HiFi-GAN，或者自己训练一个 Vocoder。")
    print(f"2. 听 '{path_recon}' (VAE Result):")
    print("   - 如果 Oracle 很清晰，但这个文件很闷，说明 VAE 丢失了高频。")
    print("   - 如果 Oracle 很清晰，但这个文件有底噪，说明 VAE 没学好静音部分的分布。")

if __name__ == "__main__":
    main()