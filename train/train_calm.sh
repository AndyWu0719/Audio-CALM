#!/bin/bash

# === 环境变量配置 ===
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_PORT=29505

# === 路径配置 ===
WORK_PATH=$(pwd)
QWEN_PATH="${WORK_PATH}/qwen_audio_pretrained"
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_16x/checkpoint-54000" 

TRAIN_DATA_DIR="${WORK_PATH}/data/latents/train"
EVAL_DATA_DIR="${WORK_PATH}/data/latents/dev"

LIBRISPEECH_ROOT="/data0/determined/users/andywu/speechcalm/data/full_librispeech/LibriSpeech"
OUTPUT_DIR="${WORK_PATH}/outputs/checkpoints/calm_latent_v1"

# === 训练参数 ===
PER_DEVICE_BATCH_SIZE=2  # 使用 Latent 省显存，可以尝试增大 Batch Size
GRAD_ACCUM=16             # 保持总 Batch Size 约为 128 (4 * 4 * 8 = 128)
LR=1e-4                  # 基础 LR，Projector 会是 1e-3

echo "=== Starting CALM Joint Training (Latent Mode) ==="
echo "Data Dir: $DATA_DIR"

# 运行训练
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_calm.py \
    --qwen_path "$QWEN_PATH" \
    --vae_path "$VAE_PATH" \
    --mel_dir "$TRAIN_DATA_DIR" \
    --eval_mel_dir "$EVAL_DATA_DIR" \
    --librispeech_root "$LIBRISPEECH_ROOT" \
    --train_subsets "train-clean-100,train-clean-360,train-other-500" \
    --eval_subsets "dev-clean" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --num_train_epochs 10 \
    --save_steps 500 \
    --logging_steps 10 \
    --max_text_len 256 \
    --max_audio_len 2048 \
    --bf16 True \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --report_to "tensorboard" \
    --ddp_find_unused_parameters False \
    --run_name "calm-latent-v1" \
    --use_lora True \
    --use_precomputed_latents True \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --metric_for_best_model "loss" \
    --save_total_limit 2 \
    --remove_unused_columns False \
    --do_train

echo "Training finished."