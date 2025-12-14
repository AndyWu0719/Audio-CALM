#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

WORK_PATH=$(pwd)
QWEN_PATH="${WORK_PATH}/qwen2_7B_Instruct"
VAE_PATH="${WORK_PATH}/outputs/checkpoints/audio_vae_4x_kl_annealing/checkpoint-109900"
CHECKPOINT_PATH="${WORK_PATH}/outputs/checkpoints/calm_latent_gmm/4-mix8-dim64-1p0e-04/checkpoint-500"

TEST_FILE="${WORK_PATH}/data/test_latent.jsonl"

OUTPUT_DIR="${WORK_PATH}/outputs/eval_results"

python eval/eval_gmm.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --qwen_path "$QWEN_PATH" \
    --vae_path "$VAE_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --task asr \
    --max_samples 50 \
    --max_new_tokens_asr 256 \
    --merge_lora \
    --num_mixtures 8 \
    --latent_dim 64 \
    --latent_downsample 4