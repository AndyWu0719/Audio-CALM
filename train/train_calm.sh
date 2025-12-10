#!/bin/bash

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1
MASTER_PORT=29505

WORK_PATH=$(pwd)
QWEN_PATH="${WORK_PATH}/qwen_audio_pretrained"
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_16x/checkpoint-54000" 

TRAIN_DATA_DIR="${WORK_PATH}/data/latents/train"
EVAL_DATA_DIR="${WORK_PATH}/data/latents/dev"

LIBRISPEECH_ROOT="/data0/determined/users/andywu/speechcalm/data/full_librispeech/LibriSpeech"

PER_DEVICE_BATCH_SIZE=2
GRAD_ACCUM=16

LATENT_DOWN=16
NUM_MIX=8
LATENT_DIM=64
LR=5e-5
LR_TAG=${LR//./p}   # 5e-5 -> 5e-5 或 5p00000e-05，根据需要可自定义
RUN_NAME="${LATENT_DOWN}-${NUM_MIX}-${LATENT_DIM}-${LR_TAG}"

OUTPUT_DIR_BASE="${WORK_PATH}/outputs/checkpoints/calm_latent_v1"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${RUN_NAME}"

echo "=== Starting CALM Joint Training (Latent Mode) ==="
echo "Train Data Dir: $TRAIN_DATA_DIR"
echo "Eval Data Dir: $EVAL_DATA_DIR"

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_calm.py \
    --do_train \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --report_to "tensorboard" \
    \
    --num_mixtures $NUM_MIX \
    --latent_dim $LATENT_DIM \
    --latent_downsample 16 \
    \
    --qwen_path "$QWEN_PATH" \
    --vae_path "$VAE_PATH" \
    --mel_dir "$TRAIN_DATA_DIR" \
    --eval_mel_dir "$EVAL_DATA_DIR" \
    --librispeech_root "$LIBRISPEECH_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    \
    --train_subsets "train-clean-100,train-clean-360,train-other-500" \
    --eval_subsets "dev-clean" \
    --max_text_len 256 \
    --max_audio_len 2048 \
    \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --num_train_epochs 3 \
    --optim "adamw_torch_fused" \
    \
    --use_lora True \
    --use_precomputed_latents True \
    --bf16 True \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters True \
    \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False

echo "Training finished."