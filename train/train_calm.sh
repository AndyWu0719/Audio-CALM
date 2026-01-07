#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1

export WANDB_PROJECT="Omni-Flow-Train" 
export WANDB_NAME="4-128-5e-5-2048-6" 
MASTER_PORT=29505

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_calm.py \
    --config-name="calm_config" \
    training.run_name="$WANDB_NAME" \
    training.output_dir="$(pwd)/outputs/checkpoints/omni_flow/$WANDB_NAME"