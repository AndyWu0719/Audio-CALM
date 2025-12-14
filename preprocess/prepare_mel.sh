#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

WORK_PATH=$(pwd)

DATA_ROOT="/data0/determined/users/andywu/speechcalm/data/full_librispeech/LibriSpeech"
OUTPUT_DIR="${WORK_PATH}/data/mel_features"
N_GPUS=4
WORKERS_PER_GPU=2 

echo "开始提取 Mel 频谱图..."
echo "数据根目录: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "GPU 数量: $N_GPUS"
echo "每个 GPU 的进程数: $WORKERS_PER_GPU"

python preprocess/prepare_mel.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --n_gpus "$N_GPUS" \
    --workers_per_gpu "$WORKERS_PER_GPU"