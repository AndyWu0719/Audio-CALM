import os
import json
import argparse
from glob import glob
from tqdm import tqdm
import torch

def save_jsonl(data, path):
    with open(path, "w") as f:
        for s in data:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(data)} samples to {path}")

def is_valid_pt_file(path):
    """
    深度检查：尝试加载文件
    """
    try:
        # 只加载不返回，验证文件完整性
        torch.load(path, map_location="cpu")
        return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel_dir", type=str, required=True, help="Path to VAE training .pt files")
    parser.add_argument("--librispeech_root", type=str, required=True, help="Original LibriSpeech root for text")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train/dev/test jsonl files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Scanning Mel files in {args.mel_dir}...")
    mel_map = {}
    
    files = glob(os.path.join(args.mel_dir, "**", "*.pt"), recursive=True)
    
    # 使用 tqdm 显示进度
    for f in tqdm(files, desc="Indexing & Verifying"):
        # [修改] 增加深度校验，确保写入 JSONL 的文件一定能用
        if not is_valid_pt_file(f):
            print(f"Skipping corrupt file: {f}")
            continue
            
        key = os.path.splitext(os.path.basename(f))[0]
        mel_map[key] = f
    
    print(f"Found {len(mel_map)} valid Mel files.")
    
    train_samples = []
    dev_samples = []
    test_samples = []
    
    print(f"Scanning Transcripts in {args.librispeech_root}...")
    for root, dirs, files in os.walk(args.librispeech_root):
        is_train = "train-" in root
        is_dev = "dev-" in root
        is_test = "test-" in root
        
        if not (is_train or is_dev or is_test):
            continue

        for f in files:
            if f.endswith(".trans.txt"):
                try:
                    with open(os.path.join(root, f), "r") as trans_f:
                        for line in trans_f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) != 2: continue
                            file_id, text = parts
                            
                            if file_id in mel_map:
                                item = {
                                    "text": text,
                                    "mel_path": mel_map[file_id]
                                }
                                
                                if is_train:
                                    train_samples.append(item)
                                elif is_dev:
                                    dev_samples.append(item)
                                elif is_test:
                                    test_samples.append(item)
                except Exception as e:
                    print(f"Error reading transcript {f}: {e}")
    
    total_matched = len(train_samples) + len(dev_samples) + len(test_samples)
    print(f"Total matched pairs: {total_matched}")
    
    if train_samples:
        save_jsonl(train_samples, os.path.join(args.output_dir, "calm_train.jsonl"))
    if dev_samples:
        save_jsonl(dev_samples, os.path.join(args.output_dir, "calm_dev.jsonl"))
    if test_samples:
        save_jsonl(test_samples, os.path.join(args.output_dir, "calm_test.jsonl"))

if __name__ == "__main__":
    main()