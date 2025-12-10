#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

WORK_PATH=$(pwd)
QWEN_PATH="${WORK_PATH}/qwen_audio_pretrained"
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_16x/checkpoint-54000"
CHECKPOINT_PATH="${WORK_PATH}/outputs/checkpoints/calm_latent_v1/checkpoint-6500" 

TEST_FILE="${WORK_PATH}/data/calm_data/calm_dev.jsonl" 

OUTPUT_DIR="${WORK_PATH}/outputs/eval_results_gmm"

python eval/eval_gmm.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --qwen_path "$QWEN_PATH" \
    --vae_path "$VAE_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples 10