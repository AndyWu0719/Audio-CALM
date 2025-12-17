#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="Audio-CALM-VAE" 
export WANDB_NAME="vae-run-2x_kl_annealing_l1_ssim_eval"
WORK_PATH=$(pwd)
CHECKPOINT_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_2x_kl_annealing_l1_ssim/checkpoint-6900"

python eval/eval_vae.py --checkpoint ${CHECKPOINT_PATH} --web_demo --output_dir "${WORK_PATH}/outputs/eval_results_vae/audio_vae_2x_kl_annealing_l1_ssim"