#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

WORK_PATH=$(pwd)
CHECKPOINT_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_4x_kl_annealing"
AUDIO_PATH="/data0/determined/users/andywu/speechcalm/data/full_wavs/test-clean/61-70970-0025_000.wav"
OUTPUT_DIR="${WORK_PATH}/outputs/eval_results_vae"


python eval/eval_vae.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --audio_path ${AUDIO_PATH} \
    --output_dir ${OUTPUT_DIR}