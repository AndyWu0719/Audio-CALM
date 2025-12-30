# preprocess/process_dataset.py
import sys
import time
import os
import argparse
import math
import csv
import warnings
import logging
import multiprocessing as mp

# [优化] 1. 启动即打印 PID，方便确认进程存活
print(f"[Process] Initializing... (PID: {os.getpid()})", flush=True)

import torch
import torchaudio

# [配置] 路径修复：确保能导入项目根目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
if current_dir not in sys.path: sys.path.insert(0, current_dir)

# 【对应关系】：从 core.py 导入核心处理工具
from core import MelExtractor, load_vae, process_audio_chunk

# [配置] 屏蔽警告
warnings.filterwarnings("ignore")
logging.getLogger("torchvision").setLevel(logging.ERROR)

def get_common_voice_map(tsv_path):
    """
    功能：读取 CommonVoice 的元数据文件 (.tsv)，建立 文件名->文本 的映射。
    """
    mapping = {}
    if not os.path.exists(tsv_path): return mapping
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            mapping[row['path']] = row['sentence']
    return mapping

def _get_librispeech_text(wav_path):
    """
    功能：解析 LibriSpeech 数据集目录下的 .trans.txt 文件获取文本。
    """
    folder = os.path.dirname(wav_path)
    file_id = os.path.splitext(os.path.basename(wav_path))[0]
    try:
        files = os.listdir(folder)
        trans_file = next((f for f in files if f.endswith(".trans.txt")), None)
        if trans_file:
            with open(os.path.join(folder, trans_file), 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(file_id):
                        return line.strip().split(" ", 1)[1]
    except: pass
    return None

def scan_files(root_dir):
    """
    功能：递归扫描目录下的所有音频文件。
    """
    sys.stdout.write(f"🔎 Scanning files in {root_dir} ... ")
    sys.stdout.flush()
    files = []
    extensions = {'.wav', '.flac', '.mp3'}
    for root, _, filenames in os.walk(root_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in extensions:
                files.append(os.path.join(root, f))
    print(f"Found {len(files)} files.", flush=True)
    return files

def worker_process(rank, gpu_id, file_list, args, cv_mapping, queue):
    """
    功能：单个工作进程的主逻辑。负责加载模型并处理分配给它的文件列表。
    
    参数：
    - rank: 进程编号
    - gpu_id: 指定使用的 GPU ID
    - file_list: 该进程需要处理的文件路径列表
    - queue: 用于向主进程汇报进度的队列
    """
    # [关键优化] 1. 限制单核，防止 CPU 争抢 (因为 PyTorch 多进程下默认会争抢 CPU)
    torch.set_num_threads(1)
    
    try:
        # 1. 加载模型到指定 GPU
        device = torch.device(f"cuda:{gpu_id}")
        # 根据模式决定是否加载 VAE
        if not args.mel_only:
            vae = load_vae(args.vae_ckpt, device)
            vae.eval()
        # 【对应关系】：调用 core.MelExtractor 初始化特征提取器
        mel_extractor = MelExtractor().to(device)
        mel_extractor.eval()
    except Exception:
        queue.put(None)
        return

    trans_buffer = {}
    # [关键优化] 2. 批量汇报阈值：每处理 100 个文件向主进程汇报一次，减少 IPC 开销
    REPORT_BATCH = 100 
    processed_count = 0

    # [关键优化] 3. 使用 inference_mode 加速并禁用梯度计算
    with torch.inference_mode():
        for wav_path in file_list:
            try:
                # --- 2. 路径计算 ---
                # 确定输出文件的保存路径
                if args.dataset_name == "commonvoice":
                    file_id = os.path.splitext(os.path.basename(wav_path))[0]
                    save_dir = args.out_dir
                else:
                    # 保持原始目录结构
                    rel_path = os.path.relpath(os.path.dirname(wav_path), args.in_dir)
                    save_dir = os.path.join(args.out_dir, rel_path)
                    file_id = os.path.splitext(os.path.basename(wav_path))[0]

                save_path = os.path.join(save_dir, f"{file_id}.pt")
                
                # --- 3. 跳过已存在的文件 ---
                if os.path.exists(save_path) and not args.force:
                    processed_count += 1
                    if processed_count >= REPORT_BATCH:
                        queue.put(processed_count)
                        processed_count = 0
                    continue

                os.makedirs(save_dir, exist_ok=True)

                # --- 4. 音频加载与标准化 ---
                wav, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                # 【对应关系】：调用 core.process_audio_chunk 进行归一化
                # non_blocking=True 尝试加速 CPU->GPU 数据传输
                wav = process_audio_chunk(wav).to(device, non_blocking=True)

                # --- 5. 特征提取与编码 ---
                # 提取 Log-Mel
                mel = mel_extractor(wav)
                
                # 处理 VAE 的下采样填充问题 (padding)
                pad_to = 4 # VAE 通常下采样 4 倍，所以长度要是 4 的倍数
                if mel.shape[-1] % pad_to != 0:
                    pad_len = pad_to - (mel.shape[-1] % pad_to)
                    mel = torch.nn.functional.pad(mel, (0, pad_len), mode='reflect')
                    
                if args.mel_only:
                    # === 分支 A: 仅保存 Mel (用于训练 VAE) ===
                    # 必须保存为 "mel" key，以便 train_vae.py 识别
                    payload = {"mel": mel.squeeze(0).cpu()} 
                    torch.save(payload, save_path)
                else:
                    # === 分支 B: 保存 Latent (用于训练 CALM) ===
                    # 必须有 VAE 才能运行
                    with torch.no_grad():
                        mu, _ = vae.encode(mel)
                        latent = mu.squeeze(0).cpu() # [Dim, Time]
                    payload = {
                        "latent": latent, 
                        "vae_path": args.vae_ckpt,
                        # "mel": mel.squeeze(0).cpu() # 可选：如果硬盘空间够，建议加上
                    }
                    torch.save(payload, save_path)

                # --- 7. 处理文本 (Transcript) ---
                # 根据不同数据集类型获取文本
                text = None
                if args.dataset_name == "libritts":
                    txt_path = wav_path.replace(".wav", ".normalized.txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f: text = f.read().strip()
                elif args.dataset_name == "librispeech":
                    text = _get_librispeech_text(wav_path)
                elif args.dataset_name == "commonvoice":
                    text = cv_mapping.get(os.path.basename(wav_path), None)

                # 缓存文本，稍后批量写入 .trans.txt
                if text:
                    fname = f"{os.path.basename(save_dir)}.trans.txt"
                    if args.dataset_name == "commonvoice": fname = "commonvoice.trans.txt"
                    tpath = os.path.join(save_dir, fname)
                    if tpath not in trans_buffer: trans_buffer[tpath] = []
                    # 格式: file_id text
                    trans_buffer[tpath].append(f"{file_id} {text}")
                
                # 进度更新
                processed_count += 1
                if processed_count >= REPORT_BATCH:
                    queue.put(processed_count)
                    processed_count = 0

            except Exception:
                # 出错也计数，防止进度条卡死，但通常应记录错误日志
                processed_count += 1
                if processed_count >= REPORT_BATCH:
                    queue.put(processed_count)
                    processed_count = 0

    # 循环结束后，汇报剩余进度
    if processed_count > 0:
        queue.put(processed_count)

    # 8. 写入文本文件缓存
    for path, lines in trans_buffer.items():
        try:
            with open(path, 'a', encoding='utf-8') as f:
                for line in lines: f.write(line + "\n")
        except: pass
    
    # 发送结束信号
    queue.put(None)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    功能：在终端打印进度条。
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称 (libritts, librispeech, commonvoice)")
    parser.add_argument("--in_dir", type=str, required=True, help="原始音频输入目录")
    parser.add_argument("--out_dir", type=str, required=True, help="Latent 输出目录")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="VAE 模型检查点路径 (mel_only模式下可忽略)")
    parser.add_argument("--mel_only", action="store_true", help="仅提取 Mel 频谱(用于训练 VAE), 不需要加载 VAE 模型")
    parser.add_argument("--cv_tsv", type=str, default=None, help="CommonVoice 的 TSV 元数据路径")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="使用的 GPU 数量")
    parser.add_argument("--workers_per_gpu", type=int, default=4, help="每个 GPU 启动的进程数")
    parser.add_argument("--force", action="store_true", help="是否强制覆盖已存在的文件")
    args = parser.parse_args()
    
    if not args.mel_only and args.vae_ckpt is None:
        print("❌ Error: 提取 Latent (非 --mel_only 模式) 必须指定 --vae_ckpt")
        return

    # 1. 扫描文件
    files = scan_files(args.in_dir)
    total_files = len(files)
    if total_files == 0: return

    # 2. 准备元数据 (仅针对 CommonVoice)
    cv_mapping = {}
    if args.dataset_name == "commonvoice" and args.cv_tsv:
        print("📖 Loading CV metadata...", flush=True)
        cv_mapping = get_common_voice_map(args.cv_tsv)

    # 3. 任务分片 (Sharding)
    num_procs = args.num_gpus * args.workers_per_gpu
    chunk_size = math.ceil(total_files / num_procs)
    chunks = [files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    # 4. 初始化多进程管理器
    manager = mp.Manager()
    queue = manager.Queue()
    
    print(f"🔥 Launching {len(chunks)} workers...", flush=True)
    
    mp.set_start_method('spawn', force=True)
    processes = []
    active_workers = 0
    
    # 5. 启动工作进程
    for rank, chunk in enumerate(chunks):
        if len(chunk) == 0: continue
        gpu_id = rank % args.num_gpus
        p = mp.Process(target=worker_process, args=(rank, gpu_id, chunk, args, cv_mapping, queue))
        p.start()
        processes.append(p)
        active_workers += 1

    # 6. 监控进度
    processed_total = 0
    finished_workers = 0
    subset_name = os.path.basename(args.in_dir.rstrip('/'))
    if subset_name == "clips": subset_name = "CV_Full"
    
    print_progress_bar(0, total_files, prefix=f'Processing {subset_name}', length=40)

    start_time = time.time()
    
    # 进度条循环：直到所有 worker 发送结束信号 (None)
    while finished_workers < active_workers:
        try:
            msg = queue.get(timeout=0.5)
            if msg is None:
                finished_workers += 1
            elif isinstance(msg, int):
                processed_total += msg
                elapsed = time.time() - start_time
                speed = processed_total / (elapsed + 1e-5)
                suffix = f"({processed_total}/{total_files}) [{speed:.1f} file/s]"
                print_progress_bar(processed_total, total_files, prefix=f'Processing {subset_name}', suffix=suffix, length=40)
        except:
            if not any(p.is_alive() for p in processes) and queue.empty():
                break
            continue

    print_progress_bar(total_files, total_files, prefix=f'Processing {subset_name}', suffix='Done!          ', length=40)
    print()
    
    for p in processes: p.join()

if __name__ == "__main__":
    main()