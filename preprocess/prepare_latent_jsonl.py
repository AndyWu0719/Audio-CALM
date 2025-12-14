"""
生成 eval 用的 test_latent.jsonl
从 LibriSpeech 的 .trans.txt 读取 transcript，匹配预计算的 latent 文件

用法:
    python preprocess/prepare_test_jsonl.py \
        --latent_dir /data0/determined/users/andywu/Audio-CALM-v2/data/latents/dev \
        --librispeech_root /data0/determined/users/andywu/speechcalm/data/full_librispeech/LibriSpeech \
        --output_file /data0/determined/users/andywu/Audio-CALM-v2/data/test_latent.jsonl \
        --subsets dev-clean dev-other
"""

import os
import json
import argparse
from glob import glob
from tqdm import tqdm


def load_transcripts(librispeech_root: str, subsets: list) -> dict:
    """
    从 LibriSpeech 的 .trans.txt 文件加载所有 transcript
    返回: {utterance_id: transcript_text}
    """
    transcripts = {}
    
    for subset in subsets:
        subset_dir = os.path.join(librispeech_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"[WARN] Subset dir not found: {subset_dir}")
            continue
        
        # LibriSpeech 结构: subset/speaker_id/chapter_id/*.trans.txt
        trans_files = glob(os.path.join(subset_dir, "*", "*", "*.trans.txt"))
        
        for trans_file in trans_files:
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 格式: "utterance_id transcript text..."
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        transcripts[utt_id] = text
    
    print(f"Loaded {len(transcripts)} transcripts from {subsets}")
    return transcripts


def extract_utt_id_from_path(latent_path: str) -> str:
    """
    从 latent 文件路径提取 utterance_id
    例如: .../dev-clean/1272-128104-0000.pt -> 1272-128104-0000
    """
    filename = os.path.basename(latent_path)
    utt_id = filename.replace(".pt", "")
    return utt_id


def main():
    parser = argparse.ArgumentParser(description="Generate test_latent.jsonl for eval")
    parser.add_argument("--latent_dir", type=str, required=True,
                        help="Directory containing latent .pt files")
    parser.add_argument("--librispeech_root", type=str, required=True,
                        help="Root directory of LibriSpeech dataset")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output jsonl file path")
    parser.add_argument("--subsets", nargs="+", default=["dev-clean", "dev-other"],
                        help="LibriSpeech subsets to use for transcripts")
    args = parser.parse_args()

    # 1. 加载所有 transcript
    transcripts = load_transcripts(args.librispeech_root, args.subsets)
    
    # 2. 扫描所有 latent 文件
    latent_files = []
    for subset in args.subsets:
        subset_dir = os.path.join(args.latent_dir, subset)
        if os.path.isdir(subset_dir):
            files = glob(os.path.join(subset_dir, "*.pt"))
            latent_files.extend(files)
    
    # 如果子目录不存在，尝试直接扫描 latent_dir
    if not latent_files:
        latent_files = glob(os.path.join(args.latent_dir, "**", "*.pt"), recursive=True)
    
    print(f"Found {len(latent_files)} latent files")
    
    # 3. 匹配并生成 jsonl
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    matched = 0
    unmatched = 0
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        for latent_path in tqdm(latent_files, desc="Generating jsonl"):
            utt_id = extract_utt_id_from_path(latent_path)
            
            if utt_id in transcripts:
                entry = {
                    "text": transcripts[utt_id],
                    "latent_path": latent_path,
                    "utt_id": utt_id,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                matched += 1
            else:
                unmatched += 1
    
    print(f"Done! Matched: {matched}, Unmatched: {unmatched}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()