#!/bin/bash

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1
MASTER_PORT=29505

WORK_PATH=$(pwd)
QWEN_PATH="${WORK_PATH}/qwen_audio_pretrained"
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_4x/checkpoint-54000" 

TRAIN_DATA_DIR="${WORK_PATH}/data/latents/train"
EVAL_DATA_DIR="${WORK_PATH}/data/latents/dev"
LIBRISPEECH_ROOT="/data0/determined/users/andywu/speechcalm/data/full_librispeech/LibriSpeech"

# 4-bit 模式下显存充足，单卡 Batch Size 设为 4，总 Batch Size 64
PER_DEVICE_BATCH_SIZE=2
GRAD_ACCUM=8 

# 开启 4-bit 量化
USE_QLORA=True

LATENT_DOWN=4
LATENT_DIM=64
NOISE_SIZE=64
MLP_LAYERS=2
NUM_SAMPLES=4
BETA=0.25
TEMPERATURE=1.0
LR=5e-5
LR_TAG=${LR//./p}

RUN_NAME="qlora-${LATENT_DOWN}-${NUM_SAMPLES}-${LATENT_DIM}-${LR_TAG}"
OUTPUT_DIR_BASE="${WORK_PATH}/outputs/checkpoints/calm_latent_energy"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${RUN_NAME}"

echo "=== Starting CALM Joint Training (Latent Mode) ==="
echo "Train Data Dir: $TRAIN_DATA_DIR"
echo "Eval Data Dir: $EVAL_DATA_DIR"

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_calm.py \
    --do_train \
    --do_eval \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --report_to "tensorboard" \
    \
    --latent_dim $LATENT_DIM \
    --latent_downsample $LATENT_DOWN \
    --noise_size $NOISE_SIZE \
    --num_mlp_layers $MLP_LAYERS \
    --num_samples $NUM_SAMPLES \
    --beta $BETA \
    --temperature $TEMPERATURE \
    --learning_rate $LR \
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
    --max_audio_len 512 \
    \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs 3 \
    --optim "adamw_torch" \
    \
    --use_lora True \
    --use_qlora $USE_QLORA \
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