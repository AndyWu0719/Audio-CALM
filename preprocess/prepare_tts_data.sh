#!/bin/bash

# === 配置路径 ===
# 你的 Python 解释器路径 (如果是 conda 环境)
PYTHON_BIN="python" 

# 原始数据存放位置
RAW_DATA_ROOT="/data0/determined/users/andywu/Audio-CALM-v2/data/LibriTTS-R"

# VAE Latent 输出位置
OUTPUT_ROOT="/data0/determined/users/andywu/Audio-CALM-v2/data/latents/LibriTTS-R_FULL"

# 你的 VAE Checkpoint 路径
VAE_CKPT="/data0/determined/users/andywu/Audio-CALM-v2/outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim/checkpoint-6900"

# 需要处理的子集 (通常我们只处理 Train 和 Dev，Test 可以不处理除非你要跑测试)
SUBSETS=(
    "train-clean-100"
    "train-clean-360"
    "train-other-500"
    "dev-clean"
    "dev-other"
)

echo "🔥 开始批量处理 VAE Latents..."

for subset in "${SUBSETS[@]}"; do
    IN_DIR="$RAW_DATA_ROOT/$subset"
    OUT_DIR="$OUTPUT_ROOT/$subset"
    
    echo "--------------------------------------------------------"
    echo "Processing subset: $subset"
    echo "Input:  $IN_DIR"
    echo "Output: $OUT_DIR"
    echo "--------------------------------------------------------"
    
    $PYTHON_BIN preprocess/prepare_tts_data.py \
        --in_dir "$IN_DIR" \
        --out_dir "$OUT_DIR" \
        --vae_ckpt "$VAE_CKPT"
        
    if [ $? -ne 0 ]; then
        echo "❌ 处理 $subset 时出错，脚本终止！"
        exit 1
    fi
done

echo "✅ 全量数据处理完成！输出目录: $OUTPUT_ROOT"