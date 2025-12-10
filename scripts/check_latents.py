import os
import torch
import glob
import argparse
import numpy as np
from tqdm import tqdm

def check_latents(data_dir, num_files=None):
    """
    检查指定目录下 Latent (.pt) 文件的分布情况
    """
    # 支持递归查找，或者是单层目录
    search_path = os.path.join(data_dir, "**/*.pt")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        # 尝试非递归搜索
        search_path = os.path.join(data_dir, "*.pt")
        files = glob.glob(search_path, recursive=False)

    if not files:
        print(f"[Error] No .pt files found in {data_dir}")
        return

    print(f"Found {len(files)} files. Starting check...")
    
    # 如果文件太多，可以限制检查的数量用于快速验证
    if num_files is not None and num_files > 0:
        files = files[:num_files]
        print(f"Checking first {num_files} files...")

    total_mean = 0.0
    total_var = 0.0
    global_min = float('inf')
    global_max = float('-inf')
    total_elements = 0
    
    nan_count = 0
    inf_count = 0

    # 为了避免爆内存，使用 Welford 算法或简单的累加统计（这里用简单的 Batch 统计近似）
    # 或者收集所有文件的 min/max/mean 做分布概览
    
    file_means = []
    file_stds = []

    for fpath in tqdm(files):
        try:
            # 加载 Latent
            latent = torch.load(fpath, map_location="cpu") # Shape expected: [Dim, Time]
            
            # 转换为 float 以防精度溢出
            latent = latent.float()
            
            # 1. 检查 NaN / Inf
            if torch.isnan(latent).any():
                print(f"[WARNING] NaN found in {fpath}")
                nan_count += 1
            if torch.isinf(latent).any():
                print(f"[WARNING] Inf found in {fpath}")
                inf_count += 1
                
            # 2. 统计数值
            l_min = latent.min().item()
            l_max = latent.max().item()
            l_mean = latent.mean().item()
            l_std = latent.std().item()
            
            if l_min < global_min: global_min = l_min
            if l_max > global_max: global_max = l_max
            
            file_means.append(l_mean)
            file_stds.append(l_std)

        except Exception as e:
            print(f"[Error] Failed to load {fpath}: {e}")

    # 汇总结果
    print("\n" + "="*40)
    print("Latent Distribution Summary")
    print("="*40)
    print(f"Total Files Checked: {len(files)}")
    print(f"Global Min: {global_min:.4f}")
    print(f"Global Max: {global_max:.4f}")
    print(f"Average Mean: {np.mean(file_means):.4f} (across files)")
    print(f"Average Std : {np.mean(file_stds):.4f} (across files)")
    print("-" * 40)
    print(f"Files with NaN: {nan_count}")
    print(f"Files with Inf: {inf_count}")
    print("="*40)

    # 简单的直方图建议
    if np.abs(np.mean(file_means)) > 1.0:
        print("[Suggestion] Latents are not centered at 0. Consider normalization if using GMM/Diffusion.")
    if np.abs(global_max) > 20.0 or np.abs(global_min) > 20.0:
        print("[Suggestion] Latent values are quite large. Check if this is expected for your VAE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .pt latent files")
    parser.add_argument("--num_files", type=int, default=None, help="Number of files to check (default: all)")
    
    args = parser.parse_args()
    
    check_latents(args.data_dir, args.num_files)