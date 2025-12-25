#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="Audio-CALM-VAE" 
export WANDB_NAME="vae-run-4x_kl_annealing_l1_ssim_eval"
WORK_PATH=$(pwd)
CHECKPOINT_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim/checkpoint-6900"

python eval/eval_vae.py --checkpoint ${CHECKPOINT_PATH} --output_dir "${WORK_PATH}/outputs/eval_results_vae/audio_vae_4x_kl_annealing_l1_ssim" --audio_path "${WORK_PATH}/data/raw/LibriSpeech/dev/dev-clean/84/121123/84-121123-0001.flac"