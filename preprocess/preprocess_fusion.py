import os
import glob
import argparse
import torch
import torch.nn as nn
import torchaudio
import torch.multiprocessing as mp
from tqdm import tqdm
import sys
import math

# === 引入你的 VAE 模型 ===
sys.path.append(os.getcwd()) 
from models.modeling_vae import AcousticVAE 

# ==============================================================================
# 1. 你的 MelExtractor (完全保留参数)
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
            normalized=False, # 你之前的设置
            f_min=0,
            f_max=8000,
            norm="slaney",
            mel_scale="slaney"
        )
    
    def forward(self, wav):
        # wav: [1, T]
        mel = self.mel_transform(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

# ==============================================================================
# 2. 核心处理逻辑 (Latent + Text)
# ==============================================================================
def process_chunk(rank, gpu_id, file_list, args):
    device = torch.device(f"cuda:{gpu_id}")
    
    # 初始化组件
    try:
        # 加载 VAE
        vae = AcousticVAE.from_pretrained(args.vae_ckpt).to(device)
        vae.eval()
        # 加载 Mel 提取器
        mel_extractor = MelExtractor().to(device)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load models: {e}")
        return

    # 进度条
    iterator = tqdm(file_list, desc=f"GPU {gpu_id}", position=rank) if rank < 8 else file_list
    
    # 用于缓存 text 写入，避免频繁 IO
    # Key: trans.txt 的绝对路径, Value: list of lines
    trans_buffer = {} 

    for wav_path in iterator:
        try:
            # --- A. 路径计算 ---
            # 保持目录结构: output_dir/subset/reader/chapter/xxx.pt
            rel_path = os.path.relpath(os.path.dirname(wav_path), args.in_dir)
            save_dir = os.path.join(args.out_dir, rel_path)
            
            file_id = os.path.splitext(os.path.basename(wav_path))[0]
            save_path = os.path.join(save_dir, f"{file_id}.pt")
            
            # 如果已存在，跳过 Latent 计算，但不能跳过文本收集！
            # 为了简单，这里建议全量跑。如果必须断点续传，需要额外逻辑处理文本。
            if os.path.exists(save_path) and not args.force_overwrite:
                pass 
            else:
                os.makedirs(save_dir, exist_ok=True)

                # --- B. 音频处理 (完全复用你的逻辑) ---
                wav, sr = torchaudio.load(wav_path)
                
                # 重采样
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    wav = resampler(wav)

                # 转单声道
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                
                # [关键] 你的归一化逻辑
                peak = torch.max(torch.abs(wav))
                if peak > 0:
                    wav = wav / (peak + 1e-8) * 0.95

                wav = wav.to(device) # [1, T]

                # Mel + VAE
                with torch.no_grad():
                    mel = mel_extractor(wav) # [1, 80, T_mel]
                    
                    # 简单的 Pad 逻辑防止 VAE 尺寸报错
                    pad_to = 4
                    if mel.shape[-1] % pad_to != 0:
                        pad_len = pad_to - (mel.shape[-1] % pad_to)
                        mel = torch.nn.functional.pad(mel, (0, pad_len), mode='reflect')

                    # Encode
                    mu, logvar = vae.encode(mel)
                    
                    # 你偏好保存 mu
                    latent = mu.squeeze(0).cpu() # [64, T_lat]

                # --- C. 保存 Latent (复用你的 Dict 格式) ---
                payload = {
                    "latent": latent,
                    "latent_type": "mu",
                    "vae_path": args.vae_ckpt,
                    # 可以顺便把 mel 存进去，如果硬盘够大的话，方便 debug
                    # "mel": mel.squeeze(0).cpu() 
                }
                torch.save(payload, save_path)

            # --- D. [关键新增] 文本处理 ---
            # 读取同目录下的 .normalized.txt
            txt_src = wav_path.replace(".wav", ".normalized.txt")
            if os.path.exists(txt_src):
                with open(txt_src, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                # 确定 trans.txt 的位置 (LibriSpeech 格式: chapter目录下的 trans.txt)
                trans_file_path = os.path.join(save_dir, f"{os.path.basename(save_dir)}.trans.txt")
                
                if trans_file_path not in trans_buffer:
                    trans_buffer[trans_file_path] = []
                
                # 格式: ID TEXT
                trans_buffer[trans_file_path].append(f"{file_id} {text_content}")

        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {wav_path} - {e}")

    # --- E. 批量写入文本 ---
    for trans_path, lines in trans_buffer.items():
        # 追加模式，防止多进程覆盖（虽然理论上不同进程处理不同文件，但同一目录可能被分到不同 chunk? 
        # 最好是按目录分 chunk，但为了简单，这里用追加模式 + 锁稍微不安全但通常没事，
        # 或者直接覆盖，因为不同 chunk 不会重叠文件）
        # 稳妥起见：
        mode = 'a' if os.path.exists(trans_path) else 'w'
        with open(trans_path, mode, encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="Raw LibriTTS-R root")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    parser.add_argument("--force_overwrite", action="store_true")
    args = parser.parse_args()

    print(f"🚀 Scanning wav files in {args.in_dir}...")
    # 递归查找所有 wav
    wav_files = glob.glob(os.path.join(args.in_dir, "**", "*.wav"), recursive=True)
    total_files = len(wav_files)
    print(f"📂 Found {total_files} files.")

    if total_files == 0:
        return

    # 多进程分配
    num_procs = args.num_gpus * args.workers_per_gpu
    chunk_size = math.ceil(total_files / num_procs)
    file_chunks = [wav_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    print(f"🔥 Starting {num_procs} processes on {args.num_gpus} GPUs...")
    
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank in range(len(file_chunks)):
        gpu_id = rank % args.num_gpus
        p = mp.Process(
            target=process_chunk,
            args=(rank, gpu_id, file_chunks[rank], args)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print("✅ All done! Latents and transcripts are ready.")

if __name__ == "__main__":
    main()