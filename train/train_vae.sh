#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export WANDB_PROJECT="Audio-CALM-VAE" 
export WANDB_NAME="vae-run-2x_kl_annealing_l1_ssim"
MASTER_PORT=29500 

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train/train_vae.py