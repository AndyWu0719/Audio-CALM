import os, argparse, torch, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm

from models.modeling_calm import QwenCALM, QwenCALMConfig, monotonic_alignment_search
from models.modeling_vae import AcousticVAE
from train.train_calm import CalmDataset, CalmCollator
from eval.eval_calm import Vocoder

@torch.no_grad()
def vae_upper_bound(dataset, model, vocoder, save_dir, num_samples=8):
    """VAE Upper Bound: 直接用 GT latent 解码，检查 VAE 质量"""
    os.makedirs(save_dir, exist_ok=True)
    
    if num_samples == 0:
        print("[VAE] save_wavs=0, skipping VAE upper bound")
        return
    
    saved = 0
    for i in range(min(len(dataset), num_samples * 10)):  # 多取一些以防无效样本
        try:
            item = dataset[i]
            if not item.get("_valid", False):
                continue
            
            audio_feat = item["audio_features"]  # [T, D] 或 [D, T]
            
            # 确保形状正确
            if audio_feat.dim() == 2:
                if audio_feat.shape[0] in (64, 80, 128, 192):
                    audio_feat = audio_feat.transpose(0, 1)  # -> [T, D]
            
            # [T, D] -> [1, D, T] for VAE
            latent = audio_feat.unsqueeze(0).transpose(1, 2).to(model.vae.device).float()
            
            # Decode
            mel = model.vae.decode(latent)  # [1, 80, T]
            print(f"[VAE] raw mel mean={mel.mean().item():.3f}, std={mel.std().item():.3f}")
            
            # Vocoder
            wav = vocoder.decode(mel).cpu().squeeze()
            
            # Save
            path = os.path.join(save_dir, f"vae_upper_{saved}.wav")
            import torchaudio
            torchaudio.save(path, wav.unsqueeze(0), 16000)
            print(f"[VAE] Saved {path}")
            
            saved += 1
            if saved >= num_samples:
                break
                
        except Exception as e:
            print(f"[VAE] Error processing sample {i}: {e}")
            continue
    
    print(f"[VAE] dumped {saved} wavs to {save_dir}")

@torch.no_grad()
def flow_baseline(batch, model):
    """
    Flow Baseline: 比较训练后的 loss 和 pred_v=0 的 loss
    如果 current ≈ baseline，说明 Flow 没学到东西
    """
    # 只处理 TTS 样本
    idx = [i for i, m in enumerate(batch["task_modes"]) if m == "tts"]
    if len(idx) == 0:
        return None
    
    idx_tensor = torch.tensor(idx, device=batch["text_input_ids"].device)
    
    # 准备数据 - 创建新的 batch 字典避免修改原始数据
    new_batch = {}
    for k in ["text_input_ids", "attention_mask", "labels", "audio_features", "audio_lens"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            new_batch[k] = batch[k][idx_tensor].to(model.llm.device)
    new_batch["task_modes"] = ["tts"] * len(idx)
    
    # 前向传播获取当前 loss
    try:
        out = model(
            text_input_ids=new_batch["text_input_ids"],
            audio_features=new_batch["audio_features"],
            attention_mask=new_batch["attention_mask"],
            labels=new_batch["labels"],
            task_modes=new_batch["task_modes"],
            audio_lens=new_batch["audio_lens"]
        )
        current_loss = out["loss_tts"].item()
    except Exception as e:
        print(f"[Flow] Error in forward: {e}")
        return None
    
    baseline = 2.0
    
    return (current_loss, baseline)

@torch.no_grad()
def len_dur_errors(loader, model, max_batches):
    len_err = []
    dur_err = []
    for bi, batch in enumerate(tqdm(loader, desc="len/dur")):
        if bi >= max_batches: break
        idx = [i for i,m in enumerate(batch["task_modes"]) if m=="tts"]
        if not idx: continue
        text_ids = batch["text_input_ids"][idx].to(model.llm.device)
        attn = batch["attention_mask"][idx].to(model.llm.device)
        audio = batch["audio_features"][idx].to(model.llm.device)
        lens = batch["audio_lens"][idx].to(model.llm.device)

        # prepare latent & normalize
        gt_lat = audio.transpose(1,2).to(model.llm.dtype)
        mean = torch.as_tensor(model.config.latent_mean, device=gt_lat.device, dtype=gt_lat.dtype)
        std  = torch.as_tensor(model.config.latent_std,  device=gt_lat.device, dtype=gt_lat.dtype)
        if mean.ndim==1: mean = mean.view(1,1,-1)
        if std.ndim==1:  std  = std.view(1,1,-1)
        gt_lat = (gt_lat - mean) / std
        audio_mask = (torch.arange(gt_lat.size(1), device=gt_lat.device)[None,:] < lens[:,None]).long()

        # [FIX] 使用 input_proj 投影 audio，与训练时一致
        aud_proj = model.input_proj(gt_lat)
        
        soa = model.soa_embed.expand(text_ids.size(0), -1, -1)
        full_mask = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
        pos_ids = full_mask.long().cumsum(-1) - 1
        pos_ids.masked_fill_(full_mask==0, 1)
        text_embeds = model.get_input_embeddings()(text_ids)
        out = model.llm(
            inputs_embeds=torch.cat([text_embeds, soa], dim=1),
            attention_mask=full_mask,
            position_ids=pos_ids,
            output_hidden_states=True,
        )
        cond_vec = out.hidden_states[-1][:, -1:, :]
        text_ctx = out.hidden_states[-1][:, :-1, :]
        text_mask = (full_mask[:, :-1]==0)
        valid_mask = ~text_mask

        # len error
        vlen = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(text_ctx.dtype)
        tmean = (text_ctx * valid_mask.unsqueeze(-1).to(text_ctx.dtype)).sum(dim=1) / vlen
        pred_dtype = model.tts_len_predictor[0].weight.dtype
        tmean = tmean.to(pred_dtype)
        len_pred = model.tts_len_predictor(tmean).squeeze(-1).clamp(min=1.0, max=float(model.config.max_audio_len))
        len_gt = audio_mask.sum(dim=1).float()
        len_err.extend((torch.abs(len_pred - len_gt) / len_gt.clamp(min=1)).cpu().tolist())

        # [FIX] MAS → dur gt，使用与训练时相同的方式
        B, T_txt, D = text_ctx.shape
        T_aud = gt_lat.size(1)
        
        # 使用 input_proj 的输出计算相似度
        txt_norm = F.normalize(text_ctx, p=2, dim=-1)
        aud_norm = F.normalize(aud_proj, p=2, dim=-1)
        
        sim = torch.bmm(txt_norm, aud_norm.transpose(1,2))
        sim = sim.masked_fill(text_mask.unsqueeze(-1), -1e9)
        sim = sim.masked_fill(~audio_mask[:,None,:].bool(), -1e9)
        
        log_p = F.log_softmax(sim, dim=1)  # [FIX] 对 text 维度 softmax，与训练一致
        align = monotonic_alignment_search(log_p)
        dur_gt = align.sum(dim=2)  # [B, T_txt]

        # dur pred
        dur_raw = model.tts_dur_predictor(text_ctx.to(pred_dtype)).squeeze(-1)
        dur_pred = F.softplus(dur_raw) + 1e-4
        dur_pred = dur_pred.masked_fill(text_mask, 0)
        
        # 归一化到目标长度
        dur_sum = dur_pred.sum(dim=1, keepdim=True).clamp(min=1e-4)
        dur_pred_scaled = dur_pred * (T_aud / dur_sum)
        
        mask = valid_mask.float()
        eps = 1e-4
        dur_err_batch = (torch.abs(torch.log(dur_pred_scaled+eps) - torch.log(dur_gt+eps)) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        dur_err.extend(dur_err_batch.cpu().tolist())
    return len_err, dur_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_batches", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--save_wavs", type=int, default=0)  # 默认为 0
    args = ap.parse_args()

    OmegaConf.register_new_resolver("hydra", lambda key: os.getcwd() if key == "runtime.cwd" else "")
    cfg = OmegaConf.load(args.cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # 加载模型
    from eval.eval_calm import load_model, Vocoder
    
    # 创建临时 config 用于 load_model
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.evaluation = OmegaConf.create({
            "checkpoint_path": args.ckpt,
            "datasets": cfg.data.datasets,
        })
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_model(cfg, cfg.device)
    model.eval()
    
    # dtype 对齐
    proj_dtype = model.llm.dtype
    model.input_proj.to(proj_dtype)
    model.tts_flow_head.to(proj_dtype)
    model.asr_flow_head.to(proj_dtype)
    if hasattr(model, "tts_len_predictor"): 
        model.tts_len_predictor.to(proj_dtype)
    if hasattr(model, "tts_dur_predictor"): 
        model.tts_dur_predictor.to(proj_dtype)
    if hasattr(model, "asr_query_embed"): 
        model.asr_query_embed.to(proj_dtype)
    model.soa_embed.data = model.soa_embed.data.to(proj_dtype)

    # 数据集
    from train.train_calm import CalmDataset, CalmCollator
    
    tts_cfg = cfg.data.datasets.tts
    dataset = CalmDataset(
        tts_latent_dir=tts_cfg.eval_latent_dir,
        tts_subsets=cfg.data.eval_subsets,
        tokenizer=tokenizer,
        max_text_len=cfg.data.max_text_len,
        max_audio_len=cfg.data.max_audio_len,
        task_mode="tts",
    )
    
    collator = CalmCollator(tokenizer.pad_token_id, training=False)
    loader = DataLoader(dataset, batch_size=args.bs, collate_fn=collator, shuffle=False)

    # 1) VAE upper bound
    if args.save_wavs > 0:
        vocoder = Vocoder(cfg.device)
        vae_upper_bound(dataset, model, vocoder, "outputs/sanity/vae_upper", num_samples=args.save_wavs)
    else:
        print("[VAE] dumped 0 wavs to outputs/sanity/vae_upper")

    # 2) Flow baseline
    fb = None
    for batch in loader:
        fb = flow_baseline(batch, model)
        if fb is not None: 
            break
    
    if fb:
        cur, base = fb
        print(f"[Flow] loss_tts (current) = {cur:.4f}, baseline (pred_v=0) = {base:.4f}")
        if cur < base * 0.5:
            print("[Flow] ✅ Flow is learning! Loss is significantly below baseline.")
        elif cur < base * 0.9:
            print("[Flow] ⚠️ Flow is learning slowly. Consider more training.")
        else:
            print("[Flow] ❌ Flow is NOT learning. Check alignment/conditioning.")
    
    # 3) Len/Dur errors
    len_err, dur_err = len_dur_errors(loader, model, args.max_batches)
    if len_err:
        import numpy as np
        print(f"[Len] rel err mean={np.mean(len_err):.3f}, p50={np.median(len_err):.3f}, p90={np.percentile(len_err, 90):.3f}")
    if dur_err:
        import numpy as np
        print(f"[Dur log] mean L1 (log space)={np.mean(dur_err):.3f}, p50={np.median(dur_err):.3f}, p90={np.percentile(dur_err, 90):.3f}")
    
if __name__ == "__main__":
    main()