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
    
    if num_files is not None and num_files > 0:
        files = files[:num_files]
        print(f"Checking first {num_files} files...")

    global_min = float('inf')
    global_max = float('-inf')
    
    nan_count = 0
    inf_count = 0
    
    file_means = []
    file_stds = []

    for fpath in tqdm(files):
        try:
            # 1. 加载数据
            payload = torch.load(fpath, map_location="cpu")
            
            # 2. [关键修复] 兼容字典格式
            if isinstance(payload, dict):
                # 尝试获取 'latent' 或 'mel'
                if "latent" in payload:
                    latent = payload["latent"]
                elif "mel" in payload:
                    latent = payload["mel"]
                else:
                    # 如果都不在，打印所有的 key 看看是什么
                    print(f"[Skip] File {os.path.basename(fpath)} is a dict but has unknown keys: {list(payload.keys())}")
                    continue
            else:
                latent = payload

            # 3. 确保是 Tensor 并且转换为 float
            if not isinstance(latent, torch.Tensor):
                print(f"[Skip] Content in {os.path.basename(fpath)} is not a Tensor (got {type(latent)})")
                continue
                
            latent = latent.float()
            
            # 4. 检查 NaN / Inf
            if torch.isnan(latent).any():
                print(f"[WARNING] NaN found in {fpath}")
                nan_count += 1
                continue # 跳过坏数据，不计入统计
            if torch.isinf(latent).any():
                print(f"[WARNING] Inf found in {fpath}")
                inf_count += 1
                continue

            # 5. 统计数值
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
    if len(file_means) == 0:
        print("\n[Error] No valid latents were processed.")
        return

    avg_mean = np.mean(file_means)
    avg_std = np.mean(file_stds)

    print("\n" + "="*40)
    print("Latent Distribution Summary")
    print("="*40)
    print(f"Total Valid Files : {len(file_means)}")
    print(f"Global Min        : {global_min:.4f}")
    print(f"Global Max        : {global_max:.4f}")
    print(f"Average Mean      : {avg_mean:.4f} (Should be close to 0)")
    print(f"Average Std       : {avg_std:.4f}  (Should be close to 1)")
    print("-" * 40)
    print(f"Files with NaN    : {nan_count}")
    print(f"Files with Inf    : {inf_count}")
    print("="*40)

    # 诊断建议
    if avg_std < 0.5:
        scale_factor = 1.0 / avg_std
        print(f"\n⚠️  [DIAGNOSIS] Variance is too SMALL ({avg_std:.4f}).")
        print(f"👉 Suggestion: Multiply latents by {scale_factor:.4f} during training.")
    elif avg_std > 2.0:
        scale_factor = 1.0 / avg_std
        print(f"\n⚠️  [DIAGNOSIS] Variance is too LARGE ({avg_std:.4f}).")
        print(f"👉 Suggestion: Multiply latents by {scale_factor:.4f} during training.")
    elif abs(avg_mean) > 0.5:
        print(f"\n⚠️  [DIAGNOSIS] Data is not centered (Mean={avg_mean:.4f}).")
        print(f"👉 Suggestion: Subtract {avg_mean:.4f} during training.")
    else:
        print(f"\n✅ [DIAGNOSIS] Data distribution looks healthy!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .pt latent files")
    parser.add_argument("--num_files", type=int, default=1000, help="Number of files to check (default: 1000)")
    
    args = parser.parse_args()
    
    check_latents(args.data_dir, args.num_files)