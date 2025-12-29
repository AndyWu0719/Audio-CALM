# preprocess/build_manifest.py
import os
import glob
import json
import argparse
from tqdm import tqdm

def main():
    """
    功能：生成用于训练的 JSONL 清单文件。
    
    【文件间关系】：
    - 输入依赖：依赖 `process_dataset.py` 生成的目录结构（包含 .trans.txt 和 .pt 文件）。
    - 输出流向：生成的 .jsonl 文件将被 `train_calm.py` 中的 `CalmDataset` 类读取。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str, required=True, help="处理好的潜变量根目录")
    parser.add_argument("--output_file", type=str, required=True, help="输出的 .jsonl 路径")
    args = parser.parse_args()

    print(f"🔨 正在从 {args.latent_dir} 构建清单...")
    
    # 1. 查找所有的转录文件
    # 这些 .trans.txt 文件是由 `process_dataset.py` 在处理音频时生成或复制的。
    # 它们包含了文件名 ID 和对应的文本内容。
    trans_files = glob.glob(os.path.join(args.latent_dir, "**", "*.trans.txt"), recursive=True)
    
    entries = []
    for trans_path in tqdm(trans_files):
        folder = os.path.dirname(trans_path)
        with open(trans_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 2. 解析每一行转录文本
                # 预期格式: "文件ID 文本内容"
                parts = line.strip().split(" ", 1)
                if len(parts) != 2: continue
                
                file_id, text = parts
                
                # 3. 定位对应的潜变量文件 (.pt)
                # 【对应关系】：这里匹配 `process_dataset.py` 中的保存命名规则：
                # save_path = os.path.join(save_dir, f"{file_id}.pt")
                latent_path = os.path.join(folder, f"{file_id}.pt")
                
                # 4. 验证文件存在性并添加到列表
                if os.path.exists(latent_path):
                    # 创建符合 CalmDataset.__getitem__ 读取格式的条目
                    entries.append({
                        "id": file_id,
                        "audio": latent_path, # 训练时将通过 torch.load() 加载此路径
                        "text": text,
                        # "dataset": "libritts" # 可选元数据
                    })

    # 5. 将结果写入 JSONL 文件
    print(f"📝 正在写入 {len(entries)} 条数据到 {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    print("✅ 清单生成完成。")

if __name__ == "__main__":
    main()