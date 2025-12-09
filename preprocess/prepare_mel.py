import os
import torch
import torchaudio
import argparse
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import torch.multiprocessing as mp
import math

# === 配置 ===
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
            normalized=True
        )
    
    def forward(self, wav):
        # wav: [1, T]
        mel = self.mel_transform(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

def worker_process(rank, file_subset, data_root, output_dir, device_id):
    """
    工作进程函数：每个进程绑定一个 GPU 处理一部分文件
    """
    # 设置当前进程使用的 GPU
    device = torch.device(f"cuda:{device_id}")
    
    # 初始化提取器 (每个进程独立初始化)
    extractor = MelExtractor().to(device)
    
    # 只需要在 rank 0 显示进度条，避免混乱
    iterator = tqdm(file_subset, desc=f"GPU {device_id}", position=rank) if rank < 8 else file_subset
    
    for wav_path in iterator:
        try:
            # === 路径计算逻辑修改 ===
            # 1. 获取相对于 data_root 的路径
            # 例如: wav_path = .../LibriSpeech/dev-clean/84/121123/84-121123-0000.flac
            # rel_path = dev-clean/84/121123/84-121123-0000.flac
            rel_path = os.path.relpath(wav_path, data_root)
            
            # 2. 拆分路径组件
            parts = rel_path.split(os.sep)
            
            if len(parts) >= 2:
                # 第一级目录 (如 dev-clean)
                subset_folder = parts[0]
                # 文件名 (如 84-121123-0000.flac)
                filename = parts[-1]
                
                # 3. 构造新的扁平化路径
                # output_dir/dev-clean/84-121123-0000.pt
                save_filename = os.path.splitext(filename)[0] + ".pt"
                save_path = os.path.join(output_dir, subset_folder, save_filename)
            else:
                # 如果文件直接在根目录下，或者路径解析异常，保持原样或放入 root
                filename = os.path.basename(wav_path)
                save_filename = os.path.splitext(filename)[0] + ".pt"
                save_path = os.path.join(output_dir, save_filename)
            
            # =======================
            
            # 检查是否存在
            if os.path.exists(save_path):
                continue
                
            # 创建目录 (只创建一级子目录)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 加载与处理
            wav, sr = torchaudio.load(wav_path)
            
            if sr != SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
                
            wav = wav.to(device)
            with torch.no_grad():
                mel = extractor(wav) 
            
            # 保存
            torch.save(mel.squeeze(0).cpu(), save_path)
            
        except Exception as e:
            print(f"[GPU {device_id}] Error processing {wav_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="LibriSpeech 根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="Mel 特征输出目录")
    parser.add_argument("--n_gpus", type=int, default=torch.cuda.device_count(), help="使用的 GPU 数量")
    parser.add_argument("--workers_per_gpu", type=int, default=2, help="每个 GPU 启动多少个进程 (建议 2-4)")
    args = parser.parse_args()
    
    # 1. 扫描文件
    print(f"Scanning files in {args.data_root}...")
    files = glob(os.path.join(args.data_root, "**/*.flac"), recursive=True)
    files += glob(os.path.join(args.data_root, "**/*.wav"), recursive=True)
    total_files = len(files)
    print(f"Found {total_files} audio files.")
    
    if total_files == 0:
        return

    # 2. 规划任务
    num_processes = args.n_gpus * args.workers_per_gpu
    chunk_size = math.ceil(total_files / num_processes)
    
    print(f"Starting {num_processes} processes on {args.n_gpus} GPUs...")
    
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_files)
        file_subset = files[start_idx:end_idx]
        
        gpu_id = i % args.n_gpus
        
        p = mp.Process(
            target=worker_process,
            args=(i, file_subset, args.data_root, args.output_dir, gpu_id)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print("All done!")

if __name__ == "__main__":
    main()