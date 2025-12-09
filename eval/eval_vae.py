import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import argparse
from glob import glob

# 引用项目根目录
sys.path.append(os.getcwd())

from models.modeling_vae import AcousticVAE
from preprocess.prepare_mel import MelExtractor # 复用之前的提取器逻辑

def save_plot(orig_mel, recon_mel, save_path):
    """画出对比图"""
    orig_mel = orig_mel.cpu().numpy()
    recon_mel = recon_mel.cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # 原始
    axes[0].imshow(orig_mel, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("Original Mel Spectrogram")
    axes[0].set_ylabel("Mel Channels")
    
    # 重建
    axes[1].imshow(recon_mel, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("Reconstructed Mel Spectrogram (VAE)")
    axes[1].set_ylabel("Mel Channels")
    axes[1].set_xlabel("Time Frames")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def mel_to_audio_griffinlim(mel, device):
    """
    使用 Griffin-Lim 算法将 Mel 频谱转回音频。
    注意：这是一个估计算法，声音会有'机械感'，但足以验证内容是否清晰。
    最好的做法是使用 HiFi-GAN，但需要额外的权重。
    """
    # 反归一化 (Log -> Linear)
    # 假设 prepare_mel 做的是 log(x + 1e-5)
    mel = torch.exp(mel) - 1e-5
    
    # 你的音频参数
    n_fft = 1024
    n_mels = 80
    sample_rate = 16000
    hop_length = 256
    
    # 1. Mel -> Linear Spectrogram (Inverse Mel Scale)
    inverse_mel_transform = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, 
        n_mels=n_mels, 
        sample_rate=sample_rate,
        norm='slaney' # 这里的 norm 需要尝试，通常 torchaudio 默认是 None
    ).to(device)
    
    linear_spec = inverse_mel_transform(mel)
    
    # 2. Linear Spectrogram -> Waveform (Griffin-Lim)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, 
        hop_length=hop_length,
        power=2.0,
    ).to(device)
    
    waveform = griffin_lim(linear_spec)
    return waveform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VAE checkpoint folder")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to a raw .flac/.wav file for testing")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_results", help="Where to save results")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载模型
    print(f"Loading model from {args.checkpoint}...")
    model = AcousticVAE.from_pretrained(args.checkpoint).to(device)
    model.eval()
    
    # 2. 处理音频
    print(f"Processing audio: {args.audio_path}")
    wav, sr = torchaudio.load(args.audio_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    wav = wav.to(device)
    
    # 3. 提取特征 (GT)
    # 临时初始化一个 extractor，保证处理逻辑一致
    extractor = MelExtractor().to(device)
    with torch.no_grad():
        gt_mel = extractor(wav) # [1, 80, T]
        
    # 4. VAE 推理
    with torch.no_grad():
        outputs = model(gt_mel)
        recon_mel = outputs['recon_mel'] # [1, 80, T]
        
    # 5. 计算指标
    mse = torch.nn.functional.mse_loss(recon_mel, gt_mel).item()
    
    # --- [新增] 1. Cosine Similarity (形状相似度) ---
    # 将 [1, 80, T] 展平为 [1, 80*T] 计算整体相似度
    gt_flat = gt_mel.reshape(gt_mel.size(0), -1) 
    recon_flat = recon_mel.reshape(recon_mel.size(0), -1)
    cosine_sim = torch.nn.functional.cosine_similarity(gt_flat, recon_flat, dim=1).mean().item()
    
    # --- [新增] 2. Explained Variance (能量还原度) ---
    # 类似于 R-squared: 1 - (误差能量 / 原始能量)
    error_energy = torch.sum((gt_mel - recon_mel) ** 2)
    gt_energy = torch.sum(gt_mel ** 2)
    # 避免分母为 0
    reconstruction_accuracy = (1.0 - error_energy / (gt_energy + 1e-8)).item()
    # 限制在 0-1 之间 (如果重建极差可能为负，但这不应该发生)
    reconstruction_accuracy = max(0.0, min(1.0, reconstruction_accuracy))

    print("-" * 30)
    print(f"MSE Loss (Lower is better):       {mse:.6f}")
    print(f"Cosine Similarity (Max 1.0):      {cosine_sim:.4f}  (-> {cosine_sim*100:.2f}%)")
    print(f"Energy Fidelity (Max 1.0):        {reconstruction_accuracy:.4f}  (-> {reconstruction_accuracy*100:.2f}%)")
    print("-" * 30)

    # 自动评判
    if cosine_sim > 0.98:
        print(">> 结论: SOTA 级还原精度 (Excellent, >98%)")
    elif cosine_sim > 0.95:
        print(">> 结论: 高保真还原 (Good, >95%)")
    else:
        print(">> 结论: 仍有损耗 (Fair, <95%)")
    print(f"Reconstruction MSE: {mse:.6f}")
    if mse < 0.2:
        print(">> Quality Check: Good (Low MSE)")
    elif mse < 0.5:
        print(">> Quality Check: Acceptable")
    else:
        print(">> Quality Check: Poor (High MSE - Training might not be converged)")

    # 6. 保存可视化对比图
    plot_path = os.path.join(args.output_dir, "comparison.png")
    save_plot(gt_mel.squeeze(0), recon_mel.squeeze(0), plot_path)
    print(f"Saved comparison plot to {plot_path}")
    
    # 7. 还原音频 (Griffin-Lim)
    print("Reconstructing waveform (Griffin-Lim)...")
    recon_wav = mel_to_audio_griffinlim(recon_mel.squeeze(0), device)
    
    # 保存音频
    out_wav_path = os.path.join(args.output_dir, "reconstructed.wav")
    orig_wav_path = os.path.join(args.output_dir, "original.wav")
    
    torchaudio.save(out_wav_path, recon_wav.cpu().unsqueeze(0), 16000)
    torchaudio.save(orig_wav_path, wav.cpu(), 16000)
    
    print(f"Saved reconstructed audio to {out_wav_path}")
    print("注意：Griffin-Lim 算法生成的音频会有机械音/金属音，这是正常的。")
    print("主要检查：语音内容是否清晰、语调是否保留、有没有严重的断裂或噪音。")

if __name__ == "__main__":
    main()