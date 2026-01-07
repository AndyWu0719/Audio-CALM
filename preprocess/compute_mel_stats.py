import os
import argparse
import math
import torch
from tqdm import tqdm

def iter_mel_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".pt"):
                yield os.path.join(dirpath, name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="mel_only 生成的根目录，例如 data/mels/train/LibriSpeech")
    args = parser.parse_args()

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for path in tqdm(list(iter_mel_files(args.root)), desc="Scanning mels"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mel = payload["mel"].float()  # [80, T]
        total_sum += mel.sum().item()
        total_sq_sum += (mel ** 2).sum().item()
        total_count += mel.numel()

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean * mean
    var = max(var, 1e-8)
    std = math.sqrt(var)

    print(f"Global mel_mean: {mean:.6f}")
    print(f"Global mel_std:  {std:.6f}")

if __name__ == "__main__":
    main()

# python preprocess/compute_mel_stats.py --root data/mels/train
# mel_mean: -6.589515
# mel_std: 3.860679