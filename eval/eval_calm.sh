#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1

CHECKPOINT_PATH="$(pwd)/outputs/checkpoints/omni_flow/tts_stage1_4-128-5e-5-2048-6/checkpoint-7000"
export WANDB_PROJECT="Omni-Flow-Eval"
export WANDB_NAME="tts_stage1_4-128-5e-5-2048-6"

echo "=================================================="
echo "Running Evaluation on: ${CHECKPOINT_PATH}"
echo "=================================================="

python eval/eval_calm.py \
    --config-name="tts_config" \
    evaluation.checkpoint_path="${CHECKPOINT_PATH}" \
    evaluation.output_dir="$(pwd)/outputs/eval_results/$WANDB_NAME" \
    evaluation.max_samples=10 \
    evaluation.steps=40 \
    evaluation.cfg_scale=1.0