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

# [NEW] Import evaluate for standard metric calculation
import evaluate

sys.path.append(os.getcwd())

from transformers import AutoTokenizer
from models.modeling_gmm import QwenCALM, QwenCALMConfig
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Use Whisper's normalizer for standard ASR evaluation
normalizer = BasicTextNormalizer()

# =============================================================================
# 1. Utils
# =============================================================================
def sample_from_gmm(pi, mu, log_sigma):
    dist_k = torch.distributions.Categorical(logits=pi)
    k = dist_k.sample()  # [B, S]
    k_expanded = k.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mu.size(-1))
    mu_k = torch.gather(mu, 2, k_expanded).squeeze(2)
    log_sigma_k = torch.gather(log_sigma, 2, k_expanded).squeeze(2)
    sigma_k = torch.exp(log_sigma_k)
    z = torch.normal(mu_k, sigma_k)
    return z

def _get_single_token_id(tokenizer, s: str):
    tid = tokenizer.convert_tokens_to_ids(s)
    if tid is not None and tid != tokenizer.unk_token_id:
        return tid
    ids = tokenizer.encode(s, add_special_tokens=False)
    return ids[-1] if len(ids) > 0 else None

def _load_state_dict_any(path: str):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    st_path = path.replace(".bin", ".safetensors")
    if os.path.exists(st_path):
        try:
            from safetensors.torch import load_file
            return load_file(st_path)
        except Exception:
            return None
    return None

def _try_load_part_from_adapter(module, adapter_sd: dict, part_name: str):
    """
    Try to load specific parts (like input_proj) from adapter checkpoints
    if they were saved via modules_to_save.
    """
    if not isinstance(adapter_sd, dict):
        return False

    matched = {}
    key_pat = f"{part_name}."
    for k, v in adapter_sd.items():
        if key_pat in k:
            new_k = k.split(key_pat, 1)[1]
            matched[new_k] = v

    if not matched:
        return False

    missing, unexpected = module.load_state_dict(matched, strict=False)
    return len(matched) > 0

# =============================================================================
# 2. Load Model
# =============================================================================
def load_calm_model(args, device):
    print(f"Loading Base Qwen from {args.qwen_path}...")

    config = QwenCALMConfig(
        qwen_path=args.qwen_path,
        vae_path=args.vae_path,
        num_mixtures=args.num_mixtures,
        latent_dim=args.latent_dim,
        downsample_rate=args.latent_downsample,
        use_precomputed_latents=False,
    )
    model = QwenCALM(config)

    # 1) Load LoRA
    if os.path.exists(os.path.join(args.checkpoint, "adapter_config.json")):
        print(f"Loading LoRA adapters from {args.checkpoint}...")
        model.llm = PeftModel.from_pretrained(model.llm, args.checkpoint)
        if args.merge_lora:
            model.llm = model.llm.merge_and_unload()
    else:
        print("No LoRA adapter found (Running pure base model?).")

    # 2) Load projector/head
    print("Loading Projector & Head weights...")
    input_proj_path = os.path.join(args.checkpoint, "input_proj.bin")
    output_head_path = os.path.join(args.checkpoint, "output_head.bin")

    loaded_proj = False
    loaded_head = False

    if os.path.exists(input_proj_path):
        model.input_proj.load_state_dict(torch.load(input_proj_path, map_location="cpu"))
        loaded_proj = True
    if os.path.exists(output_head_path):
        model.output_head.load_state_dict(torch.load(output_head_path, map_location="cpu"))
        loaded_head = True

    # fallback: try read from adapter weights
    if (not loaded_proj) or (not loaded_head):
        adapter_sd = _load_state_dict_any(os.path.join(args.checkpoint, "adapter_model.bin"))
        if adapter_sd is None:
            adapter_sd = _load_state_dict_any(os.path.join(args.checkpoint, "adapter_model.safetensors"))

        if adapter_sd is not None:
            if not loaded_proj:
                loaded_proj = _try_load_part_from_adapter(model.input_proj, adapter_sd, "input_proj")
            if not loaded_head:
                loaded_head = _try_load_part_from_adapter(model.output_head, adapter_sd, "output_head")

    if not loaded_proj:
        print("[WARN] input_proj not found! Projector may be random.")
    if not loaded_head:
        print("[WARN] output_head not found! Head may be random.")

    # 3) dtype
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"LLM dtype: {llm_dtype}")

    model.llm.to(llm_dtype)
    model.input_proj.to(llm_dtype)
    model.output_head.to(llm_dtype)

    if hasattr(model, "vae"):
        model.vae.to(torch.float32)

    model.to(device)
    model.eval()
    return model

# =============================================================================
# 3. Inference
# =============================================================================
def transcribe_audio(model, tokenizer, latent, device="cuda", max_new_tokens=256, ablate_audio=False):
    """
    ASR Inference: Latent -> Text
    """
    latent = latent.to(device).to(torch.float32)

    # Standardize shape: [T, 64]
    if latent.dim() == 2:
        if latent.shape[0] == 64:
            latent = latent.transpose(0, 1)
    elif latent.dim() == 1:
        latent = latent.unsqueeze(0)

    latent = latent.unsqueeze(0)  # [1, T, 64]

    # [FIX] Dynamically get dtype from the new Conv1d Projector
    # This fixes the AttributeError: 'AudioInputProjector' object has no attribute 'pre'
    proj_dtype = next(model.input_proj.parameters()).dtype
    latent = latent.to(proj_dtype)

    with torch.no_grad():
        audio_embeds = model.input_proj(latent)

    if ablate_audio:
        audio_embeds = torch.zeros_like(audio_embeds)

    prompt = "Transcribe the following audio:"
    user_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(user_text, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        prompt_embeds = model.get_input_embeddings()(prompt_ids)

    inputs_embeds = torch.cat([audio_embeds, prompt_embeds], dim=1)
    B, T, _ = inputs_embeds.shape
    attention_mask = torch.ones((B, T), dtype=torch.long, device=device)

    im_end_id = getattr(tokenizer, "im_end_id", None) or _get_single_token_id(tokenizer, "<|im_end|>")
    eod_id = getattr(tokenizer, "eod_id", None)
    eos_id = tokenizer.eos_token_id
    eos_token_ids = [x for x in [im_end_id, eod_id, eos_id] if x is not None]

    # Dummy IDs for generation compatibility
    used_dummy = True
    dummy_ids = torch.full((B, T), tokenizer.pad_token_id, dtype=torch.long, device=device)

    with torch.no_grad():
        try:
            output_ids = model.llm.generate(
                input_ids=dummy_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
                use_cache=True,
            )
        except Exception:
            used_dummy = False
            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
                use_cache=True,
            )

    # Slice output
    prefix_len = T if used_dummy else prompt_len
    seq = output_ids[0]
    if seq.numel() > prefix_len:
        seq = seq[prefix_len:]
    else:
        seq = seq[-max_new_tokens:] if seq.numel() > max_new_tokens else seq

    pred_text = tokenizer.decode(seq, skip_special_tokens=True).strip()
    return pred_text

# =============================================================================
# 4. Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--qwen_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True, help="jsonl: {'text':..., 'latent_path':...}")

    parser.add_argument("--task", type=str, default="asr", choices=["asr", "tts", "both"])
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of samples to evaluate. -1 for all.")
    parser.add_argument("--max_new_tokens_asr", type=int, default=256)
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--ablate_audio", action="store_true", help="Zero out audio for diagnosis")

    parser.add_argument("--num_mixtures", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--latent_downsample", type=int, default=4)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eod_id", None) or tokenizer.eos_token_id
    if not hasattr(tokenizer, "im_end_id") or tokenizer.im_end_id is None:
        tokenizer.im_end_id = _get_single_token_id(tokenizer, "<|im_end|>")

    # Load Model
    model = load_calm_model(args, device)

    # [NEW] Load Hugging Face Metric
    print("Loading Hugging Face WER metric...")
    wer_metric = evaluate.load("wer")

    # Load Data
    data = []
    with open(args.test_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                pass

    if args.max_samples > 0 and len(data) > args.max_samples:
        print(f"Sampling {args.max_samples} from {len(data)} total samples.")
        import random
        random.shuffle(data)
        data = data[:args.max_samples]
    else:
        print(f"Evaluating on FULL dataset: {len(data)} samples.")

    print(f"Starting Evaluation: task={args.task}, samples={len(data)}")

    # Lists to store normalized strings for global calculation
    all_predictions = []
    all_references = []

    for i, item in enumerate(tqdm(data)):
        text_gt = item.get("text", "") or ""
        latent_path = item.get("latent_path", "") or ""

        if args.task in ("asr", "both"):
            if not latent_path or not os.path.exists(latent_path):
                # print(f"[Sample {i}] ASR skipped (no latent)")
                continue

            payload = torch.load(latent_path, map_location="cpu")
            if isinstance(payload, dict):
                latent = payload["latent"]
            else:
                latent = payload

            pred_text = transcribe_audio(
                model, tokenizer, latent, device=device,
                max_new_tokens=args.max_new_tokens_asr,
                ablate_audio=args.ablate_audio
            )

            # [CRITICAL] Normalize using Whisper's normalizer
            # This ensures fair comparison with Open ASR Leaderboard
            norm_gt = normalizer(text_gt)
            norm_pred = normalizer(pred_text)

            if len(norm_gt) > 0:
                all_references.append(norm_gt)
                all_predictions.append(norm_pred)

            if i % 10 == 0:
                print(f"[{i}] GT (Norm): {norm_gt[:60]}...")
                print(f"[{i}] PR (Norm): {norm_pred[:60]}...")

    # Calculate Global WER
    if len(all_references) > 0:
        final_wer = wer_metric.compute(predictions=all_predictions, references=all_references)
        
        print("\n" + "="*40)
        print(f"Final Global WER: {final_wer:.2%}")
        print(f"Total evaluated samples: {len(all_references)}")
        print("="*40)
        
        # Save results
        with open(os.path.join(args.output_dir, "wer_results.txt"), "w") as f:
            f.write(f"Global WER: {final_wer}\n")
            f.write(f"Samples: {len(all_references)}\n")
    else:
        print("No valid samples found for evaluation.")

if __name__ == "__main__":
    main()