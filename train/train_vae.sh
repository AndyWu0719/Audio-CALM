#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export WANDB_PROJECT="Omni-Flow-VAE" 
export WANDB_NAME="vae_4x_128_5e-4"
MASTER_PORT=29500 

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_vae.py