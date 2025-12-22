#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
WORK_PATH=$(pwd)

python preprocess/preprocess_fusion.py \
  --in_dir /data0/determined/users/andywu/Audio-CALM-v2/data/LibriTTS-R \
  --out_dir /data0/determined/users/andywu/Audio-CALM-v2/data/latents/LibriTTS-R_FULL \
  --vae_ckpt /data0/determined/users/andywu/Audio-CALM-v2/outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim/checkpoint-6900 \
  --num_gpus 4 \
  --workers_per_gpu 4