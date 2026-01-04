#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1

RUN_NAME="asr-4-64-1e-4-2048-6"

BASE_DIR="$(pwd)/outputs/checkpoints"

CHECKPOINT_PATH="${BASE_DIR}/calm_moa_flow/asr-4-64-1e-4-2048-6/checkpoint-21980"
export WANDB_PROJECT="Audio-CALM-Eval"
export WANDB_NAME="${RUN_NAME}"

echo "=================================================="
echo "Running Evaluation on: ${CHECKPOINT_PATH}"
echo "=================================================="

python eval/eval_calm.py \
    --config-name="asr_config" \
    evaluation.task="asr" \
    evaluation.checkpoint_path="${CHECKPOINT_PATH}" \
    evaluation.output_dir="$(pwd)/outputs/eval_results_moa/${RUN_NAME}" \
    evaluation.max_samples=1000 \
    evaluation.flow_steps=50 \
    evaluation.cfg_scale=1.0