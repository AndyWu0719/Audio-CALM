#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1

export WANDB_PROJECT="Audio-CALM-VAE" 
export WANDB_NAME="gmm-mix-4-mix16-dim64-1e-4" 
MASTER_PORT=29505

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_gmm.py \
    data.task_mode="mix" \
    \
    model.freeze_projector=True \
    \
    training.learning_rate=1e-4 \
    training.run_name="$WANDB_NAME" \
    training.output_dir="$(pwd)/outputs/checkpoints/$WANDB_NAME"