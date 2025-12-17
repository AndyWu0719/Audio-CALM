#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ENABLE_SDPA_FLASH_ATTENTION=1
export WANDB_PROJECT="Audio-CALM-VAE" 
export WANDB_NAME="eval_gmm-mix-4-mix8-dim64-1e-4" 

# 场景 1: 批量评估 TTS (默认)
# 你可以在这里使用 Hydra 的语法覆盖 config 文件里的参数
python eval/eval_gmm.py \
    evaluation.task="asr" \
    evaluation.max_samples=50

# 场景 2: 启动 Gradio Web Demo
# python eval/eval_gmm.py evaluation.web_demo=True