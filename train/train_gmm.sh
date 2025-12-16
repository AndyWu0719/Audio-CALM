#!/bin/bash
# TTS
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1
MASTER_PORT=29505

WORK_PATH=$(pwd)
QWEN_PATH="${WORK_PATH}/qwen2_7B_Instruct" 
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim/checkpoint-6900" 

ASR_CHECKPOINT="${WORK_PATH}/outputs/checkpoints/calm_latent_gmm/4-mix8-dim64-1e-4-1/checkpoint-10990"

TRAIN_DATA_DIR="${WORK_PATH}/data/latents/train"
EVAL_DATA_DIR="${WORK_PATH}/data/latents/dev"

LIBRISPEECH_ROOT="/data0/determined/users/andywu/speechcalm/data/full_librispeech/LibriSpeech"

PER_DEVICE_BATCH_SIZE=4
GRAD_ACCUM=8
LR=1e-4

LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

NUM_MIX=8
LATENT_DOWN=4
LATENT_DIM=64

LR_TAG=${LR//./p}

RUN_NAME="tts_${LATENT_DOWN}-mix${NUM_MIX}-dim${LATENT_DIM}-${LR_TAG}-1"
OUTPUT_DIR_BASE="${WORK_PATH}/outputs/checkpoints/calm_latent_gmm"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${RUN_NAME}"

echo "=== Starting CALM Joint Training (Latent Mode) ==="
echo "Train Data Dir: $TRAIN_DATA_DIR"
echo "Eval Data Dir: $EVAL_DATA_DIR"

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_gmm.py \
    --do_train \
    --do_eval \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --report_to "tensorboard" \
    --task_mode tts \
    \
    --num_mixtures $NUM_MIX \
    --latent_dim $LATENT_DIM \
    --latent_downsample $LATENT_DOWN \
    \
    --qwen_path "$QWEN_PATH" \
    --vae_path "$VAE_PATH" \
    --mel_dir "$TRAIN_DATA_DIR" \
    --eval_mel_dir "$EVAL_DATA_DIR" \
    --librispeech_root "$LIBRISPEECH_ROOT" \
    \
    --train_subsets "train-clean-100,train-clean-360,train-other-500" \
    --eval_subsets "dev-clean" \
    --max_text_len 256 \
    --max_audio_len 512 \
    \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --freeze_projector True \
    --learning_rate $LR \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --num_train_epochs 5 \
    --optim "adamw_torch" \
    \
    --use_lora True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    \
    --use_precomputed_latents True \
    --bf16 True \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --deepspeed "train/ds_config.json" \
    \
    --dataloader_num_workers 8 \
    --remove_unused_columns False \
    \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --eval_strategy "steps" \
    --eval_steps 1000 \
    --load_best_model_at_end True \
    --metric_for_best_model "loss"