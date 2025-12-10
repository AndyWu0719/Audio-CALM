#!/bin/bash

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_PORT=29505
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

WORK_PATH=$(pwd)
QWEN_PATH="${WORK_PATH}/qwen_audio_pretrained"
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_16x/checkpoint-54000" 

TRAIN_DATA_DIR="${WORK_PATH}/data/latents/train"
EVAL_DATA_DIR="${WORK_PATH}/data/latents/dev"

LIBRISPEECH_ROOT="/data0/determined/users/andywu/speechcalm/data/full_librispeech/LibriSpeech"
OUTPUT_DIR="${WORK_PATH}/outputs/checkpoints/calm_latent_v1"

PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32
LR=5e-5                  # Projector: 1e-3

echo "=== Starting CALM Joint Training (Latent Mode) ==="
echo "Train Data Dir: $TRAIN_DATA_DIR"
echo "Eval Data Dir: $EVAL_DATA_DIR"

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
    --ddp_find_unused_parameters True \
    --run_name "calm-latent-v1" \
    --use_lora True \
    --use_precomputed_latents True \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_total_limit 2 \
    --remove_unused_columns False \
    --latent_downsample 16 \
    --save_strategy "eval" \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --do_train

echo "Training finished."