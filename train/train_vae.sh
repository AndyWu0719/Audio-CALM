#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
MASTER_PORT=29500 

WORK_PATH=$(pwd)
DATA_DIR="${WORK_PATH}/data/mel_features" 
OUTPUT_DIR="${WORK_PATH}/outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim"

STRIDES="2 2" 
LATENT_DIM=64
HIDDEN_DIM=512

BATCH_SIZE=512
CROP_SIZE=512
GRAD_ACCUM=1
LR=5e-4

echo "Starting VAE Training..."
echo "Strides: $STRIDES"
echo "Output: $OUTPUT_DIR"

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_vae.py \
    --data_dir "$DATA_DIR" \
    --train_subsets "train-clean-100,train-clean-360,train-other-500" \
    --eval_subsets "dev-clean" \
    --output_dir "$OUTPUT_DIR" \
    --strides $STRIDES \
    --latent_channels $LATENT_DIM \
    --hidden_channels $HIDDEN_DIM \
    --kl_weight 0.0001 \
    --kl_clamp 0.0 \
    --latent_dropout 0.0 \
    --crop_size $CROP_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --num_train_epochs 50 \
    --save_steps 2000 \
    --save_total_limit 3 \
    --logging_steps 50 \
    --eval_strategy "steps" \
    --eval_steps 2000 \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --run_name "audio_vae_4x_kl_annealing_l1_ssim" \
    --save_strategy "steps" 
    
echo "Training finished."