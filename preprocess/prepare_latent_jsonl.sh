#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python preprocess/prepare_latent_jsonl.py \
    --latent_dir /data0/determined/users/andywu/Audio-CALM-v2/data/latents/dev \
    --librispeech_root /data0/determined/users/andywu/speechcalm/data/full_librispeech/LibriSpeech \
    --output_file /data0/determined/users/andywu/Audio-CALM-v2/data/latents_jsonl/dev_clean_latent.jsonl \
    --subsets dev-clean