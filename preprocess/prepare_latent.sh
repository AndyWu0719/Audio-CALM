#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

WORK_PATH=$(pwd)
MEL_DIR="${WORK_PATH}/data/mel_features"
OUT_DIR="${WORK_PATH}/data/latents/test"
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_4x/checkpoint-54000"

SUBSETS="test-clean test-other"

python preprocess/prepare_latent.py \
  --mel_dir "$MEL_DIR" \
  --output_dir "$OUT_DIR" \
  --vae_path "$VAE_PATH" \
  --subsets $SUBSETS \
  --num_gpus 4 \
  --workers_per_gpu 2