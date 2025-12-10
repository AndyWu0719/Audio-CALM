import os
import torch
import argparse
from tqdm import tqdm
from glob import glob
import sys
import torch.multiprocessing as mp
import math

sys.path.append(os.getcwd())
from models.modeling_vae import AcousticVAE

def process_files(rank, gpu_id, file_list, args):
    device = torch.device(f"cuda:{gpu_id}")
    
    try:
        vae = AcousticVAE.from_pretrained(args.vae_path).to(device)
        vae.eval()
    except Exception as e:
        print(f"[Rank {rank}] Failed to load model: {e}")
        return

    iterator = tqdm(file_list, desc=f"Rank {rank} (GPU {gpu_id})", position=rank)

    for f_path in iterator:
        try:
            rel_path = os.path.relpath(f_path, args.mel_dir)
            out_path = os.path.join(args.output_dir, rel_path)
            
            if os.path.exists(out_path):
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Load Mel [80, T]
            mel = torch.load(f_path, map_location=device)
            
            # Encode -> Latent [64, T/16]
            with torch.no_grad():
                mu, _ = vae.encode(mel.unsqueeze(0)) # Add batch dim
                # mu shape: [1, 64, T_latent]
                latent = mu.squeeze(0).cpu() # [64, T_latent]
            
            # Save
            torch.save(latent, out_path)
            
        except Exception as e:
            print(f"[Rank {rank}] Error processing {f_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel_dir", type=str, required=True, help="Input Mel directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output Latent directory")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--subsets", nargs="+", default=None, help="Specific subfolders to process")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--workers_per_gpu", type=int, default=2, help="Number of processes per GPU")
    args = parser.parse_args()

    if args.subsets:
        subsets = args.subsets
    else:
        subsets = [d for d in os.listdir(args.mel_dir) if os.path.isdir(os.path.join(args.mel_dir, d))]
    
    print(f"Target subsets: {subsets}")
    
    all_files = []
    for subset in subsets:
        subset_dir = os.path.join(args.mel_dir, subset)
        files = glob(os.path.join(subset_dir, "*.pt"))
        all_files.extend(files)
    
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")
    
    if total_files == 0:
        return

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    num_processes = num_gpus * args.workers_per_gpu
    
    print(f"Launching {num_processes} processes on {num_gpus} GPUs ({args.workers_per_gpu} workers/GPU)...")

    chunk_size = math.ceil(total_files / num_processes)
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank in range(num_processes):
        gpu_id = rank % num_gpus
        if rank < len(file_chunks):
            p = mp.Process(
                target=process_files,
                args=(rank, gpu_id, file_chunks[rank], args)
            )
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()

    print("All done!")

if __name__ == "__main__":
    main()