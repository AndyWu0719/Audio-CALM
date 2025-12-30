import os
import sys
import torch
import torchaudio
import soundfile as sf
import argparse
import logging
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# 添加项目根目录到路径
sys.path.append(os.getcwd())

# 导入你的模型定义
try:
    from models.modeling_vae import AcousticVAE, AudioVAEConfig
    from preprocess.core import MelExtractor
except ImportError:
    print("❌ 无法导入模型，请确保你在项目根目录下运行此脚本 (例如: python scripts/diagnose_full.py ...)")
    sys.exit(1)

# 配置日志
logging.basicConfig(level="ERROR") # 屏蔽底层杂乱日志
console = Console()

# ==============================================================================
# 1. 核心功能模块
# ==============================================================================

def analyze_distribution(latent_tensor, name="Latent"):
    """
    移植自 scripts/check_latents.py 的核心统计逻辑
    """
    # 确保是 float 并且在 CPU 上计算统计量
    data = latent_tensor.detach().cpu().float()
    
    # 1. 基础检查
    has_nan = torch.isnan(data).any().item()
    has_inf = torch.isinf(data).any().item()
    
    l_min = data.min().item()
    l_max = data.max().item()
    l_mean = data.mean().item()
    l_std = data.std().item()
    
    # 2. 打印表格
    table = Table(title=f"📊 分布统计: {name}", border_style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="bold yellow")
    table.add_column("Health Check", style="bold")

    # NaN / Inf Check
    status_nan = "[bold red]FAIL[/bold red]" if has_nan else "[green]PASS[/green]"
    status_inf = "[bold red]FAIL[/bold red]" if has_inf else "[green]PASS[/green]"
    table.add_row("Contains NaN", str(has_nan), status_nan)
    table.add_row("Contains Inf", str(has_inf), status_inf)
    
    # Stats Check
    table.add_row("Min", f"{l_min:.4f}", "")
    table.add_row("Max", f"{l_max:.4f}", "")
    
    # Mean Check (Should be close to 0)
    mean_status = "[green]OK[/green]" if abs(l_mean) < 0.5 else "[yellow]SHIFTED[/yellow]"
    table.add_row("Mean", f"{l_mean:.4f}", mean_status)
    
    # Std Check (Should be close to 1, or at least > 0.1)
    if l_std < 0.1: std_status = "[bold red]COLLAPSED (Too Small)[/bold red]"
    elif l_std > 5.0: std_status = "[bold red]EXPLODED (Too Large)[/bold red]"
    else: std_status = "[green]OK[/green]"
    table.add_row("Std Dev", f"{l_std:.4f}", std_status)
    
    console.print(table)
    
    # 3. 诊断建议
    if l_std < 0.5:
        scale_factor = 1.0 / (l_std + 1e-8)
        console.print(f"[yellow]💡 建议: 方差过小。如果这是 Flow Matching 的目标，建议训练时乘以 {scale_factor:.2f}[/yellow]")
    elif l_std > 2.0:
        scale_factor = 1.0 / (l_std + 1e-8)
        console.print(f"[yellow]💡 建议: 方差过大。建议训练时乘以 {scale_factor:.2f}[/yellow]")
    
    return l_mean, l_std

def load_vae(ckpt_path, device):
    console.print(f"[bold blue]Loading VAE from: {ckpt_path}[/bold blue]")
    try:
        # 尝试直接加载
        vae = AcousticVAE.from_pretrained(ckpt_path)
    except Exception as e:
        console.print(f"[yellow]直接加载失败，尝试加载 state_dict: {e}[/yellow]")
        config = AudioVAEConfig() # 使用默认配置
        vae = AcousticVAE(config)
        
        # 兼容 pytorch_model.bin 或 model.safetensors
        bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
        if not os.path.exists(bin_path):
            # 简单尝试递归查找
            import glob
            files = glob.glob(os.path.join(ckpt_path, "**/*.bin"), recursive=True)
            if files: bin_path = files[0]
            
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            vae.load_state_dict(state_dict, strict=False)
        else:
            console.print("[bold red]❌ 找不到 VAE 权重文件！[/bold red]")
            sys.exit(1)
    
    vae.to(device).eval()
    return vae

def load_vocoder(device):
    console.print("[bold blue]Loading HiFi-GAN Vocoder...[/bold blue]")
    try:
        from speechbrain.inference.vocoders import HIFIGAN
        hifi = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="tmp_hifigan",
            run_opts={"device": device}
        )
        return hifi
    except ImportError:
        console.print("[red]❌ 需要安装 speechbrain: pip install speechbrain[/red]")
        sys.exit(1)

# ==============================================================================
# 2. 主流程
# ==============================================================================

def run_diagnostic(pt_path, wav_path, vae, vocoder, device):
    console.rule("[bold]开始全能诊断[/bold]")
    
    # --- 阶段 1: 硬盘文件 (.pt) 分析 ---
    console.print(Panel(f"📂 阶段 1: 分析硬盘文件\n{pt_path}", style="bold cyan"))
    
    if not os.path.exists(pt_path):
        console.print("[red]❌ .pt 文件不存在[/red]")
        return

    payload = torch.load(pt_path, map_location="cpu")
    # 兼容处理：有些 pt 是 tensor，有些是 dict
    if isinstance(payload, dict):
        latent_disk = payload.get("latent", payload.get("mel", None))
        if latent_disk is None:
            console.print(f"[red]❌ 字典中找不到 'latent' 或 'mel' 键。Keys: {list(payload.keys())}[/red]")
            return
    else:
        latent_disk = payload
    
    # 维度调整 [C, T] -> [1, C, T]
    if latent_disk.dim() == 2: 
        latent_disk = latent_disk.unsqueeze(0)
    
    latent_disk = latent_disk.to(device).float()
    
    # [新增功能] 统计分布
    analyze_distribution(latent_disk, "硬盘 Latent (.pt)")
    
    # 解码硬盘 Latent (还原声音)
    console.print("🔄 正在解码硬盘 Latent...")
    with torch.no_grad():
        mel_disk = vae.decode(latent_disk)
        # 兼容 HiFi-GAN 的 Log 预处理
        # 假设 VAE 输出是 Log Mel (ln), HiFi-GAN 需要 Log10 Mel
        # 如果你的 VAE 输出是 Linear，这里可能会炸，这正是我们要测的
        mel_for_vocoder = mel_disk * 0.43429
        wav_disk = vocoder.decode_batch(mel_for_vocoder).cpu().squeeze()
        
    path_disk = "debug_output_disk_latent.wav"
    sf.write(path_disk, wav_disk.numpy(), 16000)
    console.print(f"💾 音频已保存: [bold red]{path_disk}[/bold red] (听听是否正常)")


    # --- 阶段 2: 原始 Wav 对比分析 ---
    if wav_path and os.path.exists(wav_path):
        console.print(Panel(f"🎵 阶段 2: 对比原始 Wav\n{wav_path}", style="bold magenta"))
        
        # 模拟预处理逻辑 (与 preprocess/core.py 保持一致)
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000: wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        # 关键：归一化
        wav = wav / (torch.max(torch.abs(wav)) + 1e-8) * 0.95
        wav = wav.to(device)
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        
        # 提取 Mel
        mel_extractor = MelExtractor().to(device)
        with torch.no_grad():
            mel_gt = mel_extractor(wav)
            
            # 现场编码 (Fresh Latent)
            mu, logvar = vae.encode(mel_gt)
            latent_fresh = mu # 在 eval 模式下通常使用均值
            
            # [新增功能] 统计现场分布
            analyze_distribution(latent_fresh, "现场编码 Latent (Fresh)")
            
            # 现场解码
            mel_fresh = vae.decode(latent_fresh)
            mel_fresh_vocoder = mel_fresh * 0.43429
            wav_fresh = vocoder.decode_batch(mel_fresh_vocoder).cpu().squeeze()
            
        path_fresh = "debug_output_fresh_encode.wav"
        sf.write(path_fresh, wav_fresh.numpy(), 16000)
        console.print(f"💾 音频已保存: [bold green]{path_fresh}[/bold green] (代表当前 VAE 的能力上限)")
        
        # --- 阶段 3: 最终对比 ---
        console.print(Panel("🔍 阶段 3: 新旧一致性检查", style="bold white"))
        
        # 维度对齐
        t_disk = latent_disk
        t_fresh = latent_fresh
        
        if t_disk.shape != t_fresh.shape:
             console.print(f"[yellow]⚠️ 形状不匹配: Disk{t_disk.shape} vs Fresh{t_fresh.shape}[/yellow]")
             # 尝试转置
             if t_disk.shape[-1] == t_fresh.shape[1]:
                 t_disk = t_disk.transpose(1, 2)
                 console.print("   -> 已转置 Disk Latent 以匹配")
        
        # 截取相同长度
        min_len = min(t_disk.shape[-1], t_fresh.shape[-1])
        t_disk = t_disk[..., :min_len]
        t_fresh = t_fresh[..., :min_len]
        
        diff = torch.abs(t_disk - t_fresh).mean().item()
        
        table = Table(title="一致性对比")
        table.add_column("Metric")
        table.add_column("Result")
        table.add_row("平均 L1 差异", f"{diff:.4f}")
        
        if diff > 0.5:
            res_style = "[bold red]FAIL[/bold red]"
            msg = "差异巨大！预处理流程或 VAE 版本不一致！"
        elif diff > 0.1:
            res_style = "[yellow]WARNING[/yellow]"
            msg = "存在明显差异，可能是 Padding 或 归一化参数微调导致。"
        else:
            res_style = "[bold green]PASS[/bold green]"
            msg = "两者基本一致。"
            
        table.add_row("结论", res_style)
        console.print(table)
        console.print(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-CALM VAE & Data Diagnostic Tool")
    parser.add_argument("--pt", type=str, required=True, help="Path to a .pt file (latent)")
    parser.add_argument("--wav", type=str, required=True, help="Path to the corresponding source .wav file")
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE checkpoint folder")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vae = load_vae(args.vae, device)
    vocoder = load_vocoder(device)
    
    run_diagnostic(args.pt, args.wav, vae, vocoder, device)
    
# python ./scripts/check_pt.py --pt "/data0/determined/users/andywu/Audio-CALM-v2/data/latents/dev/LibriTTS_R/dev-clean/84/121123/84_121123_000008_000001.pt" --wav "/data0/determined/users/andywu/Audio-CALM-v2/data/raw/LibriTTS_R/dev/dev-clean/84/121123/84_121123_000008_000001.wav" --vae "/data0/determined/users/andywu/Audio-CALM-v2/outputs/checkpoints/vae_4x_64_5e-4/checkpoint-8700"