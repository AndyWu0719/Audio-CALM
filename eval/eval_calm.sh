#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1

RUN_NAME="eval-moa-flow-asr-mix-4-dim64-1e-4--2048-4-new"

BASE_DIR="$(pwd)/outputs/checkpoints"

CHECKPOINT_PATH="${BASE_DIR}/calm_moa_flow/flow-moa-asr-4-dim64-1e-4-2048-4/checkpoint-6594"

export WANDB_PROJECT="Audio-CALM-Eval"
export WANDB_NAME="${RUN_NAME}"

echo "=================================================="
echo "Running Evaluation on: ${CHECKPOINT_PATH}"
echo "=================================================="

python eval/eval_calm.py \
    evaluation.task="asr" \
    evaluation.checkpoint_path="${CHECKPOINT_PATH}" \
    evaluation.output_dir="$(pwd)/outputs/eval_results_moa/${RUN_NAME}" \
    evaluation.max_samples=50 \
    evaluation.flow_steps=1000 \
    +evaluation.cfg_scale=5.0