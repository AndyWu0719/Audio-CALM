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

# [配置] 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
if current_dir not in sys.path: sys.path.insert(0, current_dir)

from core import MelExtractor, load_vae, process_audio_chunk

# [配置] 屏蔽警告
warnings.filterwarnings("ignore")
logging.getLogger("torchvision").setLevel(logging.ERROR)

def get_common_voice_map(tsv_path):
    mapping = {}
    if not os.path.exists(tsv_path): return mapping
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            mapping[row['path']] = row['sentence']
    return mapping

def _get_librispeech_text(wav_path):
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
    # [关键优化] 1. 限制单核，防止 CPU 争抢
    torch.set_num_threads(1)
    
    try:
        device = torch.device(f"cuda:{gpu_id}")
        vae = load_vae(args.vae_ckpt, device)
        mel_extractor = MelExtractor().to(device)
        vae.eval()
        mel_extractor.eval()
    except Exception:
        queue.put(None)
        return

    trans_buffer = {}
    # [关键优化] 2. 批量汇报阈值
    REPORT_BATCH = 100 
    processed_count = 0

    # [关键优化] 3. 使用 torch.inference_mode() (修复了这里的 import 错误)
    with torch.inference_mode():
        for wav_path in file_list:
            try:
                # --- 路径计算 ---
                if args.dataset_name == "commonvoice":
                    file_id = os.path.splitext(os.path.basename(wav_path))[0]
                    save_dir = args.out_dir
                else:
                    rel_path = os.path.relpath(os.path.dirname(wav_path), args.in_dir)
                    save_dir = os.path.join(args.out_dir, rel_path)
                    file_id = os.path.splitext(os.path.basename(wav_path))[0]

                save_path = os.path.join(save_dir, f"{file_id}.pt")
                
                # --- Skip ---
                if os.path.exists(save_path) and not args.force:
                    processed_count += 1
                    if processed_count >= REPORT_BATCH:
                        queue.put(processed_count)
                        processed_count = 0
                    continue

                os.makedirs(save_dir, exist_ok=True)

                # --- Process ---
                wav, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                # non_blocking=True 尝试加速数据传输
                wav = process_audio_chunk(wav).to(device, non_blocking=True)

                # --- Compute ---
                mel = mel_extractor(wav)
                pad_to = 4
                if mel.shape[-1] % pad_to != 0:
                    pad_len = pad_to - (mel.shape[-1] % pad_to)
                    mel = torch.nn.functional.pad(mel, (0, pad_len), mode='reflect')
                
                mu, _ = vae.encode(mel)
                latent = mu.squeeze(0).cpu()

                # --- Save ---
                payload = {"latent": latent, "vae_path": args.vae_ckpt}
                torch.save(payload, save_path)

                # --- Text ---
                text = None
                if args.dataset_name == "libritts":
                    txt_path = wav_path.replace(".wav", ".normalized.txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f: text = f.read().strip()
                elif args.dataset_name == "librispeech":
                    text = _get_librispeech_text(wav_path)
                elif args.dataset_name == "commonvoice":
                    text = cv_mapping.get(os.path.basename(wav_path), None)

                if text:
                    fname = f"{os.path.basename(save_dir)}.trans.txt"
                    if args.dataset_name == "commonvoice": fname = "commonvoice.trans.txt"
                    tpath = os.path.join(save_dir, fname)
                    if tpath not in trans_buffer: trans_buffer[tpath] = []
                    trans_buffer[tpath].append(f"{file_id} {text}")
                
                processed_count += 1
                if processed_count >= REPORT_BATCH:
                    queue.put(processed_count)
                    processed_count = 0

            except Exception:
                processed_count += 1
                if processed_count >= REPORT_BATCH:
                    queue.put(processed_count)
                    processed_count = 0

    if processed_count > 0:
        queue.put(processed_count)

    for path, lines in trans_buffer.items():
        try:
            with open(path, 'a', encoding='utf-8') as f:
                for line in lines: f.write(line + "\n")
        except: pass
    
    queue.put(None)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--cv_tsv", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--workers_per_gpu", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    files = scan_files(args.in_dir)
    total_files = len(files)
    if total_files == 0: return

    cv_mapping = {}
    if args.dataset_name == "commonvoice" and args.cv_tsv:
        print("📖 Loading CV metadata...", flush=True)
        cv_mapping = get_common_voice_map(args.cv_tsv)

    num_procs = args.num_gpus * args.workers_per_gpu
    chunk_size = math.ceil(total_files / num_procs)
    chunks = [files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    manager = mp.Manager()
    queue = manager.Queue()
    
    print(f"🔥 Launching {len(chunks)} workers...", flush=True)
    
    mp.set_start_method('spawn', force=True)
    processes = []
    active_workers = 0
    
    for rank, chunk in enumerate(chunks):
        if len(chunk) == 0: continue
        gpu_id = rank % args.num_gpus
        p = mp.Process(target=worker_process, args=(rank, gpu_id, chunk, args, cv_mapping, queue))
        p.start()
        processes.append(p)
        active_workers += 1

    processed_total = 0
    finished_workers = 0
    subset_name = os.path.basename(args.in_dir.rstrip('/'))
    if subset_name == "clips": subset_name = "CV_Full"
    
    print_progress_bar(0, total_files, prefix=f'Processing {subset_name}', length=40)

    start_time = time.time()
    
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