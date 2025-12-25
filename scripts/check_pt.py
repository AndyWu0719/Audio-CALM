import os
import sys
import torch
import torchaudio
import soundfile as sf
import argparse
import logging
from rich.console import Console
from rich.table import Table

# 添加项目根目录到路径
sys.path.append(os.getcwd())

# 导入你的模型定义
from models.modeling_vae import AcousticVAE, AudioVAEConfig
from preprocess.core import MelExtractor

# 配置日志
logging.basicConfig(level="ERROR") # 屏蔽底层杂乱日志
console = Console()

def load_vae(ckpt_path, device):
    console.print(f"[bold blue]Loading VAE from: {ckpt_path}[/bold blue]")
    try:
        # 尝试直接加载
        vae = AcousticVAE.from_pretrained(ckpt_path)
    except Exception as e:
        console.print(f"[yellow]直接加载失败，尝试加载 state_dict: {e}[/yellow]")
        config = AudioVAEConfig() # 使用默认配置，如果你的配置改过，这里可能需要调整
        vae = AcousticVAE(config)
        state_dict = torch.load(os.path.join(ckpt_path, "pytorch_model.bin"), map_location="cpu")
        vae.load_state_dict(state_dict, strict=False)
    
    vae.to(device).eval()
    return vae

def load_vocoder(device):
    console.print("[bold blue]Loading HiFi-GAN Vocoder...[/bold blue]")
    from speechbrain.inference.vocoders import HIFIGAN
    hifi = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz",
        savedir="tmp_hifigan",
        run_opts={"device": device}
    )
    return hifi

def run_diagnostic(pt_path, wav_path, vae, vocoder, device):
    console.rule("[bold]开始诊断[/bold]")
    
    # 1. 加载硬盘上的 .pt 文件 (Old Latent)
    console.print(f"📂 读取 PT 文件: {pt_path}")
    payload = torch.load(pt_path, map_location="cpu")
    # 兼容处理：有些 pt 是 tensor，有些是 dict
    latent_disk = payload.get("latent", payload) if isinstance(payload, dict) else payload
    
    # 维度调整 [C, T] -> [1, C, T]
    if latent_disk.dim() == 2: 
        latent_disk = latent_disk.unsqueeze(0)
    
    latent_disk = latent_disk.to(device).float()
    
    # 2. 解码硬盘 Latent (还原声音)
    with torch.no_grad():
        mel_disk = vae.decode(latent_disk)
        wav_disk = vocoder.decode_batch(mel_disk).cpu().squeeze()
        
    path_disk = "debug_output_disk_latent.wav"
    sf.write(path_disk, wav_disk.numpy(), 16000)
    console.print(f"💾 [Old] 硬盘Latent解码保存为: [bold red]{path_disk}[/bold red] (听听这个，如果全是噪音，说明pt失效)")

    # 3. 现场处理 Wav (如果提供了)
    if wav_path and os.path.exists(wav_path):
        console.print(f"🎵 读取原始 WAV: {wav_path}")
        
        # 模拟预处理逻辑
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000: wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        # 关键：归一化 (和你 preprocess/core.py 保持一致)
        wav = wav / (torch.max(torch.abs(wav)) + 1e-8) * 0.95
        wav = wav.to(device)
        
        # 提取 Mel
        mel_extractor = MelExtractor().to(device)
        with torch.no_grad():
            mel_gt = mel_extractor(wav)
            
            # 现场编码 (Fresh Latent)
            # 注意：你的模型返回 (mu, logvar)
            mu, logvar = vae.encode(mel_gt)
            latent_fresh = mu # 在 eval 模式下通常使用均值
            
            # 现场解码
            mel_fresh = vae.decode(latent_fresh)
            wav_fresh = vocoder.decode_batch(mel_fresh).cpu().squeeze()
            
        path_fresh = "debug_output_fresh_encode.wav"
        sf.write(path_fresh, wav_fresh.numpy(), 16000)
        console.print(f"💾 [New] 现场重新编码保存为: [bold green]{path_fresh}[/bold green] (听听这个，这代表VAE的真实水平)")
        
        # 4. 数值对比 (真相时刻)
        # 确保维度对齐 (有时候 pt 里可能是转置过的)
        if latent_disk.shape != latent_fresh.shape:
             # 尝试转置匹配
             if latent_disk.shape[-1] == latent_fresh.shape[1]:
                 latent_disk = latent_disk.transpose(1, 2)
        
        # 截取相同长度对比
        min_len = min(latent_disk.shape[-1], latent_fresh.shape[-1])
        diff = torch.abs(latent_disk[..., :min_len] - latent_fresh[..., :min_len]).mean().item()
        
        table = Table(title="Latent 数值对比")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("硬盘 Latent 均值", f"{latent_disk.mean().item():.4f}")
        table.add_row("现场 Latent 均值", f"{latent_fresh.mean().item():.4f}")
        table.add_row("两者平均差异 (L1)", f"{diff:.4f}")
        
        console.print(table)
        
        if diff > 0.5:
            console.print("[bold red]⚠️ 警告：差异巨大！[/bold red]")
            console.print("这说明【硬盘里的 .pt】和【当前 VAE 算出来的】完全不是一回事。")
            console.print("可能原因：")
            console.print("1. 你虽然用了'老VAE'，但可能是不小心用了不同的 checkpoint (比如 step 5k 和 step 10k)。")
            console.print("2. 预处理参数变了 (比如之前是 Power 1.0, 现在的 core.py 是 Power 2.0)。")
        else:
            console.print("[bold green]✅ 差异很小，Latent 是一致的。[/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, required=True, help="Path to a .pt file")
    parser.add_argument("--wav", type=str, required=True, help="Path to the corresponding .wav file")
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE checkpoint folder")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vae = load_vae(args.vae, device)
    vocoder = load_vocoder(device)
    
    run_diagnostic(args.pt, args.wav, vae, vocoder, device)