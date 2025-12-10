import os
import sys
import json
import torch
import torchaudio
import argparse
import numpy as np
import re
from tqdm import tqdm
from peft import PeftModel
import editdistance as ed

# 确保能导入项目根目录
sys.path.append(os.getcwd())

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.modeling_gmm import QwenCALM, QwenCALMConfig, sample_from_gmm
from models.modeling_vae import AcousticVAE

def load_calm_model(args, device):
    """
    重组训练好的 Audio-CALM 模型
    """
    print(f"Loading Base Qwen from {args.qwen_path}...")
    config = QwenCALMConfig(
        qwen_path=args.qwen_path,
        vae_path=args.vae_path,
        num_mixtures=8
    )
    model = QwenCALM(config)
    
    # 1. 加载 LoRA
    if os.path.exists(os.path.join(args.checkpoint, "adapter_model.bin")):
        print(f"Loading LoRA adapters from {args.checkpoint}...")
        model.llm = PeftModel.from_pretrained(model.llm, args.checkpoint)
        model.llm = model.llm.merge_and_unload()
    else:
        print("No LoRA adapter found.")

    # 2. 加载 Projector & Head
    print("Loading Projector & Head weights...")
    input_proj_path = os.path.join(args.checkpoint, "input_proj.bin")
    output_head_path = os.path.join(args.checkpoint, "output_head.bin")
    
    if os.path.exists(input_proj_path):
        model.input_proj.load_state_dict(torch.load(input_proj_path, map_location="cpu"))
    else:
        print(f"Warning: {input_proj_path} not found! Projector is random!")

    if os.path.exists(output_head_path):
        model.output_head.load_state_dict(torch.load(output_head_path, map_location="cpu"))
    else:
        print(f"Warning: {output_head_path} not found! Head is random!")

    # 3. [关键修复] 精度管理
    # LLM 部分使用 bf16/fp16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.llm.to(dtype)
    model.input_proj.to(dtype)
    model.output_head.to(dtype)
    
    # VAE 保持 float32 以确保音频生成的数值稳定性
    model.vae.to(torch.float32)

    model.to(device)
    model.eval()
    return model

def generate_audio_from_text(model, tokenizer, text, max_len=256, device="cuda"):
    """
    TTS 推理：文本 -> 音频 Latent
    """
    prompt = f"<|im_start|>user\nRead this: {text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 路径修复: transformer.wte
        text_embeds = model.llm.transformer.wte(text_ids) 
    
    curr_inputs = text_embeds
    generated_latents = []
    past_key_values = None
    
    with torch.no_grad():
        for _ in range(max_len):
            # Forward
            outputs = model.llm.transformer(
                inputs_embeds=curr_inputs, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            
            # 获取最后一个 token 的 hidden state
            last_hidden = outputs.last_hidden_state[:, -1:, :] 
            
            # GMM 预测
            pi, mu, log_sigma = model.output_head(last_hidden)
            
            # 采样
            z_next = sample_from_gmm(pi, mu, log_sigma) # [1, 1, 64]
            generated_latents.append(z_next)
            
            # 下一步输入 (注意：input_proj 需要 bf16/fp16，z_next 可能是 float32)
            z_next_casted = z_next.to(model.input_proj.weight.dtype)
            curr_inputs = model.input_proj(z_next_casted)
            
    latents = torch.cat(generated_latents, dim=1) # [1, T_gen, 64]
    return latents

def transcribe_audio(model, tokenizer, mel, device="cuda"):
    """
    ASR 推理：音频 -> 文本
    """
    TARGET_LEN = 2048
    if mel.shape[1] < TARGET_LEN:
        mel = torch.nn.functional.pad(mel, (0, TARGET_LEN - mel.shape[1]))
    mel = mel.to(device).unsqueeze(0).to(torch.float32) # VAE 需要 float32
    
    with torch.no_grad():
        mu, _ = model.vae.encode(mel)
        latents = mu.transpose(1, 2) # [1, T_lat, 64]
        latents = latents.to(model.input_proj.weight.dtype)
        audio_embeds = model.input_proj(latents)
        
    prompt = "<|im_start|>user\nTranscribe audio:<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        prompt_embeds = model.llm.transformer.wte(prompt_ids)
        
    inputs_embeds = torch.cat([audio_embeds, prompt_embeds], dim=1)
    
    B, T, _ = inputs_embeds.shape
    dummy_ids = torch.full((B, T), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((B, T), dtype=torch.long, device=device)
    
    eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    with torch.no_grad():
        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            input_ids=dummy_ids,       # 必须传
            attention_mask=attention_mask, # 必须传
            max_new_tokens=100,
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id
        )
        
    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return pred_text

def mel_to_wav(vae, latents, device):
    """Latent -> Mel -> Wav"""
    # 确保 latents 是 float32 且形状正确
    latents = latents.to(torch.float32)
    if latents.shape[-1] == 64:
        latents = latents.transpose(1, 2) # [1, 64, T]
        
    with torch.no_grad():
        recon_mel = vae.decode(latents) # [1, 80, T]
    
    # 反归一化 (假设训练时用了 log)
    mel = torch.exp(recon_mel) - 1e-5
    
    # [修复] Mel -> Linear Spectrogram -> Griffin-Lim
    # 1. 定义逆 Mel 变换
    inverse_mel_transform = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1,  # 513
        n_mels=80,
        sample_rate=16000,
        f_min=0.0,
        f_max=8000.0
    ).to(device)
    
    # 2. 定义 Griffin-Lim
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=1024, 
        hop_length=256, 
        power=1.0 # 通常幅度谱用 1.0，能量谱用 2.0，这里 mel 还原回来通常是幅度近似
    ).to(device)
    
    # 3. 执行转换
    try:
        linear_spec = inverse_mel_transform(mel)
        wav = griffin_lim(linear_spec)
    except Exception as e:
        print(f"Warning: Griffin-Lim failed: {e}")
        wav = torch.zeros(1, 16000).to(device) # 返回静音防止崩溃

    return wav

def compute_wer(ref, hyp):
    # [优化] 去除标点并转小写
    ref = re.sub(r'[^\w\s]', '', ref).lower().split()
    hyp = re.sub(r'[^\w\s]', '', hyp).lower().split()
    return ed.eval(ref, hyp) / len(ref) if len(ref) > 0 else 1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--qwen_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--max_samples", type=int, default=10)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eod_id
    
    model = load_calm_model(args, device)
    
    data = []
    with open(args.test_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    import random
    random.shuffle(data)
    data = data[:args.max_samples]
    
    print(f"Starting Evaluation on {len(data)} samples...")
    
    asr_wers = []
    tts_sims = []
    
    for i, item in enumerate(tqdm(data)):
        text_gt = item['text']
        # 加载 Mel 并转为 float32 (VAE 需要)
        mel_gt = torch.load(item['mel_path']).to(device).to(torch.float32)
        
        # === Task 1: TTS Eval ===
        vae_stride = getattr(model.vae, "total_stride", 16)
        target_len = mel_gt.shape[1] // vae_stride
        if target_len < 10:
            target_len = 50
        
        gen_latents = generate_audio_from_text(model, tokenizer, text_gt, max_len=target_len, device=device)
        
        wav_gen = mel_to_wav(model.vae, gen_latents, device)
        torchaudio.save(os.path.join(args.output_dir, f"sample_{i}_gen.wav"), wav_gen.cpu(), 16000)
        
        # Cosine Sim
        with torch.no_grad():
            mu_gt, _ = model.vae.encode(mel_gt.unsqueeze(0))
            gt_latents = mu_gt.transpose(1, 2)
            
        min_l = min(gen_latents.shape[1], gt_latents.shape[1])
        # 确保计算 Sim 时类型一致
        cos_sim = torch.nn.functional.cosine_similarity(
            gen_latents[:, :min_l, :].to(torch.float32).reshape(1, -1),
            gt_latents[:, :min_l, :].to(torch.float32).reshape(1, -1),
            dim=1
        ).item()
        tts_sims.append(cos_sim)
        
        # === Task 2: ASR Eval ===
        pred_text = transcribe_audio(model, tokenizer, mel_gt, device=device)
        wer = compute_wer(text_gt, pred_text)
        asr_wers.append(wer)
        
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"Sample {i}:\n")
            f.write(f"  GT Text: {text_gt}\n")
            f.write(f"  Pred Text: {pred_text} (WER: {wer:.2f})\n")
            f.write(f"  TTS CosSim: {cos_sim:.4f}\n\n")

    print("="*30)
    print(f"Average WER (ASR): {np.mean(asr_wers):.4f}")
    print(f"Average CosSim (TTS): {np.mean(tts_sims):.4f}")
    print(f"Samples saved to {args.output_dir}")

if __name__ == "__main__":
    main()