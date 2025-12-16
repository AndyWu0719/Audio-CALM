#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

WORK_PATH=$(pwd)
MEL_DIR="${WORK_PATH}/data/mel_features"
OUT_DIR="${WORK_PATH}/data/latents/train"
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim/checkpoint-6900"

SUBSETS="train-clean-100 train-clean-360 train-other-500"

python preprocess/prepare_latent.py \
  --mel_dir "$MEL_DIR" \
  --output_dir "$OUT_DIR" \
  --vae_path "$VAE_PATH" \
  --subsets $SUBSETS \
  --num_gpus 4 \
  --workers_per_gpu 2 \
  --latent_type mu