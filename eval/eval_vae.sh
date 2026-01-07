#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="Omni-Flow-VAE" 
export WANDB_NAME="eval_vae_4x_128_5e-4"
WORK_PATH=$(pwd)
CHECKPOINT_PATH="${WORK_PATH}/outputs/checkpoints/vae_4x_128_5e-4/checkpoint-17350"

python eval/eval_vae.py --checkpoint ${CHECKPOINT_PATH} --output_dir "${WORK_PATH}/outputs/eval_results_vae/vae_4x_128_5e-4" --audio_path "${WORK_PATH}/data/raw/LibriSpeech/dev/dev-clean/84/121123/84-121123-0001.flac"