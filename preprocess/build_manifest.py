# preprocess/build_manifest.py
import os
import glob
import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str, required=True, help="Root directory of processed latents")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl path")
    args = parser.parse_args()

    print(f"🔨 Building manifest from {args.latent_dir}...")
    
    # 1. Find all trans.txt files (because they contain the text & ID)
    trans_files = glob.glob(os.path.join(args.latent_dir, "**", "*.trans.txt"), recursive=True)
    
    entries = []
    for trans_path in tqdm(trans_files):
        folder = os.path.dirname(trans_path)
        with open(trans_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2: continue
                
                file_id, text = parts
                # Assuming latent file is named {file_id}.pt
                latent_path = os.path.join(folder, f"{file_id}.pt")
                
                if os.path.exists(latent_path):
                    # Create a clean relative entry or absolute path
                    entries.append({
                        "id": file_id,
                        "audio": latent_path,
                        "text": text,
                        # "dataset": "libritts" # Optional
                    })

    print(f"📝 Writing {len(entries)} items to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    print("✅ Manifest generation done.")

if __name__ == "__main__":
    main()