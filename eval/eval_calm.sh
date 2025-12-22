#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1

RUN_NAME="eval-moa-flow-asr-mix-4-dim64-1e-4--2048-4"
BASE_DIR="$(pwd)/outputs/checkpoints" # 对应 config 中的 training.output_dir
CHECKPOINT_PATH="${BASE_DIR}/calm_moa_flow/flow-moa-asr-4-dim64-1e-4-2048-4/checkpoint-6594"

# WandB设置
export WANDB_PROJECT="Audio-CALM-Eval"
export WANDB_NAME="${RUN_NAME}"

echo "=================================================="
echo "Running Evaluation on: ${CHECKPOINT_PATH}"
echo "=================================================="

# 场景 1: 批量评估 TTS
python eval/eval_calm.py \
    evaluation.task="asr" \
    evaluation.checkpoint_path="${CHECKPOINT_PATH}" \
    evaluation.output_dir="$(pwd)/outputs/eval_results_moa/${RUN_NAME}" \
    evaluation.max_samples=50 \
    evaluation.flow_steps=50  # Flow Matching 推理步数，建议设为 32 或 50 以获得高质量音频

# 场景 2: 批量评估 ASR (如果需要)
# python eval/eval_calm.py \
#     evaluation.task="asr" \
#     evaluation.checkpoint_path="${CHECKPOINT_PATH}" \
#     evaluation.max_samples=100

# 场景 3: 启动 Web Demo
# python eval/eval_calm.py \
#     evaluation.web_demo=True \
#     evaluation.checkpoint_path="${CHECKPOINT_PATH}"