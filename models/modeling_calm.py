"""
Flow-based Audio-CALM: Qwen-based Multimodal Model.
Features: PyTorch SDPA, Flow Matching Head, ASR-Optimized Linear Projector, and CTC Auxiliary Loss.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from models.modeling_vae import AcousticVAE

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EPS = 1e-8

# ---------------------------------------------------------------------
# Audio Input Projector (ASR Optimized)
# ---------------------------------------------------------------------
class AudioInputProjector(nn.Module):
    """
    Projector combining Conv1d (local context), Positional Embeddings (alignment),
    and Deep MLP (feature mixing).
    """
    def __init__(self, latent_dim, llm_dim, max_audio_len=1024):
        super().__init__()
        
        # 1. Local Feature Extraction
        self.conv_block = nn.Sequential(
            nn.Conv1d(latent_dim, llm_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(llm_dim, llm_dim, kernel_size=3, padding=1),
        )
        
        # 2. Positional Information
        self.pos_emb = nn.Embedding(max_audio_len, llm_dim)
        self.max_audio_len = max_audio_len

        # 3. Deep Context Mixing
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(llm_dim, eps=1e-6),
                nn.Linear(llm_dim, llm_dim * 2),
                nn.GELU(),
                nn.Linear(llm_dim * 2, llm_dim),
            ) for _ in range(2)
        ])
        self.post_norm = nn.LayerNorm(llm_dim, eps=1e-6)

    def forward(self, x):
        # x: [Batch, Time, Dim]
        B, T, _ = x.shape
        device = x.device
        
        # Step A: Conv1d Feature Extraction
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = x.transpose(1, 2)
        
        # Step B: Inject Positional Embeddings
        T_clamped = min(T, self.max_audio_len)
        pos_ids = torch.arange(T_clamped, device=device).unsqueeze(0).expand(B, -1)
        pos_emb_val = self.pos_emb(pos_ids)

        if T > self.max_audio_len:
            pos_emb_full = torch.zeros(B, T, x.size(-1), device=device, dtype=x.dtype)
            pos_emb_full[:, :T_clamped, :] = pos_emb_val
            x = x + pos_emb_full
        else:
            x = x + pos_emb_val
            
        # Step C: Deep Mixing
        for block in self.blocks:
            x = x + block(x)
        
        return self.post_norm(x)

# ---------------------------------------------------------------------
# Flow Matching Head
# ---------------------------------------------------------------------
class FlowMatchingHead(nn.Module):
    class SinusoidalPosEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            device = x.device
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb).to(dtype=x.dtype)
            emb = x[:, None] * emb[None, :]
            return torch.cat((emb.sin(), emb.cos()), dim=-1)

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 2048, num_layers: int = 4):
        super().__init__()
        self.time_dim = 256
        self.time_mlp = nn.Sequential(
            self.SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim), nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.in_proj = nn.Linear(input_dim + output_dim + self.time_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
        # Zero initialization for stability
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, condition, noisy_x, t, condition_mask=None):
        if t.dim() == 1: 
            t = t.unsqueeze(1).expand(-1, condition.size(1))
        
        # CFG Masking
        if condition_mask is not None:
            mask_expanded = condition_mask.view(-1, 1, 1) if condition_mask.dim() == 1 else condition_mask.unsqueeze(-1)
            condition = condition * mask_expanded.to(dtype=condition.dtype)

        t_emb = self.time_mlp(t.reshape(-1)).view(condition.shape[0], condition.shape[1], -1)
        x = torch.cat([condition, noisy_x, t_emb], dim=-1)
        x = self.in_proj(x)
        
        for layer in self.layers:
            x = x + layer(x)
            
        return self.out_proj(x)

# ---------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------
def compute_flow_loss(model_head, condition, target_latent, mask, cfg_dropout_prob=0.1):
    B, T, D = target_latent.shape
    device = target_latent.device
    dtype = target_latent.dtype
    mask = mask.bool()
    
    # Conditional Dropout for Classifier-Free Guidance
    if model_head.training and cfg_dropout_prob > 0:
        keep_prob = 1.0 - cfg_dropout_prob
        cfg_mask = torch.bernoulli(torch.full((B,), keep_prob, device=device)).to(dtype)
    else:
        cfg_mask = None 

    # Flow Matching Noise Generation
    t = torch.rand(B, device=device, dtype=dtype).unsqueeze(1).expand(-1, T)
    x0 = torch.randn_like(target_latent)
    x1 = target_latent
    xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
    target_v = x1 - x0
    
    pred_v = model_head(condition, xt, t, condition_mask=cfg_mask)
    loss = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=-1)
    
    return (loss * mask).sum() / mask.sum().clamp(min=1)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
class QwenCALMConfig(PretrainedConfig):
    model_type = "qwen_calm"
    def __init__(self, qwen_path=None, vae_path=None, use_precomputed_latents=False, 
                 latent_dim=64, audio_loss_weight=1.0, ctc_loss_weight=0.1,
                 downsample_rate=4, max_audio_len=1024, flow_hidden_dim=2048, 
                 flow_num_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.qwen_path = qwen_path
        self.vae_path = vae_path
        self.use_precomputed_latents = use_precomputed_latents
        self.latent_dim = latent_dim
        self.audio_loss_weight = audio_loss_weight
        self.ctc_loss_weight = ctc_loss_weight
        self.downsample_rate = downsample_rate
        self.max_audio_len = max_audio_len
        self.flow_hidden_dim = flow_hidden_dim
        self.flow_num_layers = flow_num_layers

# ---------------------------------------------------------------------
# Main Model: QwenCALM
# ---------------------------------------------------------------------
class QwenCALM(PreTrainedModel):
    config_class = QwenCALMConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config: QwenCALMConfig):
        super().__init__(config)
        self.config = config
        
        # 1. Load LLM Backbone
        print(f"Loading Qwen from {config.qwen_path}...")
        try:
            attn_impl = "flash_attention_2"
            import flash_attn
            print(f"✅ Flash Attention 2 found, using it for attention implementation.")
        except ImportError:
            print("⚠️ Flash Attention 2 not found, falling back to SDPA.")
            attn_impl = "sdpa"
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, trust_remote_code=True, 
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, 
            attn_implementation=attn_impl
        )

        # 2. VAE Setup
        self.use_precomputed_latents = config.use_precomputed_latents
        vae_latent_dim = config.latent_dim
        if not self.use_precomputed_latents:
            print(f"Loading VAE from {config.vae_path}...")
            self.vae = AcousticVAE.from_pretrained(config.vae_path)
            self.vae.requires_grad_(False)
            self.vae.eval()
            vae_latent_dim = self.vae.config.latent_channels
        
        # 3. Audio Projector
        qwen_dim = self.llm.config.hidden_size
        self.input_proj = AudioInputProjector(vae_latent_dim, qwen_dim, max_audio_len=config.max_audio_len)
        
        # 4. CTC Head (for ASR)
        if config.ctc_loss_weight > 0:
            vocab_size = self.llm.config.vocab_size
            print(f"[QwenCALM] Initializing CTC Head (Vocab={vocab_size})...")
            self.ctc_head = self.llm.lm_head 
            self.ctc_loss_fct = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # 5. Flow Head (for TTS)
        print(f"[QwenCALM] Flow Head (dim={config.flow_hidden_dim}, L={config.flow_num_layers})")
        self.output_head = FlowMatchingHead(
            qwen_dim, vae_latent_dim, config.flow_hidden_dim, config.flow_num_layers
        )

    def get_input_embeddings(self): 
        return self.llm.get_input_embeddings()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None: 
            gradient_checkpointing_kwargs = {}
        gradient_checkpointing_kwargs["use_reentrant"] = False 
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.input_proj.requires_grad_(True)
        
    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(self, text_input_ids, audio_features, attention_mask=None, labels=None, task_modes=None, audio_lens=None):
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device
        if task_modes is None: 
            task_modes = ["tts"] * batch_size

        # --- 1. Latent Extraction ---
        with torch.no_grad():
            if self.use_precomputed_latents:
                gt_latents = audio_features.transpose(1, 2)
            else:
                mu, _ = self.vae.encode(audio_features)
                gt_latents = mu.transpose(1, 2)
        
        # --- 2. Embedding & Mask Preparation ---
        B_aud, T_aud, _ = gt_latents.shape
        if audio_lens is not None:
            if self.use_precomputed_latents:
                latent_lens = audio_lens
            else:
                ds_rate = getattr(self.config, 'downsample_rate', 4) 
                latent_lens = torch.div(audio_lens + ds_rate - 1, ds_rate, rounding_mode='floor')
            latent_lens = latent_lens.clamp(max=T_aud)
            audio_mask = (torch.arange(T_aud, device=device)[None, :] < latent_lens[:, None]).long()
        else:
            audio_mask = torch.ones((B_aud, T_aud), device=device, dtype=torch.long)

        gt_latents = gt_latents.to(dtype=self.llm.dtype)
        audio_embeds = self.input_proj(gt_latents) 
        
        if self.training and self.llm.is_gradient_checkpointing:
            audio_embeds.requires_grad_(True)
            
        text_embeds = self.get_input_embeddings()(text_input_ids)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, text_input_ids.shape[1]), device=device, dtype=torch.long)

        # Loss Accumulators
        total_loss = torch.tensor(0.0, device=device)
        accum_tts_loss = torch.tensor(0.0, device=device)
        accum_asr_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        tts_count = 0
        asr_count = 0

        # --- 3. TTS Branch (Flow Matching) ---
        tts_indices = [i for i, m in enumerate(task_modes) if m == "tts"]
        if len(tts_indices) > 0:
            if hasattr(self.llm, "set_adapter"): 
                self.llm.set_adapter("tts")
            
            idx = torch.tensor(tts_indices, device=device)
            # Concatenate Text + Audio (Latent Conditioning)
            inp = torch.cat([text_embeds[idx], audio_embeds[idx][:, :-1, :]], dim=1)
            full_mask = torch.cat([attention_mask[idx], audio_mask[idx][:, :-1]], dim=1)
            pos_ids = full_mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(full_mask == 0, 1)

            out = self.llm(inputs_embeds=inp, attention_mask=full_mask, position_ids=pos_ids, output_hidden_states=True)
            
            # Extract Audio Output
            audio_hidden = out.hidden_states[-1][:, text_embeds[idx].shape[1]-1 :, :]
            tts_mask = audio_mask[idx][:, : gt_latents.size(1)].bool()
            
            # Compute Flow Loss
            tts_loss = compute_flow_loss(self.output_head, audio_hidden, gt_latents[idx], tts_mask, cfg_dropout_prob=0.1)

            total_loss += tts_loss * self.config.audio_loss_weight
            accum_tts_loss += tts_loss
            tts_count += 1
            valid_samples += 1

        # --- 4. ASR Branch (LLM + CTC) ---
        asr_indices = [i for i, m in enumerate(task_modes) if m == "asr"]
        if len(asr_indices) > 0:
            if hasattr(self.llm, "set_adapter"): 
                self.llm.set_adapter("asr")

            idx = torch.tensor(asr_indices, device=device)
            sub_audio = audio_embeds[idx]
            sub_text = text_embeds[idx]
            
            # Input: Audio -> Text
            inp = torch.cat([sub_audio, sub_text], dim=1)
            full_mask = torch.cat([audio_mask[idx], attention_mask[idx]], dim=1)
            
            # Label Construction (Mask Audio Part)
            B_sub = len(asr_indices)
            prefix_labels = torch.full((B_sub, sub_audio.shape[1]), -100, dtype=torch.long, device=device)
            full_labels = torch.cat([prefix_labels, labels[idx]], dim=1)
            
            pos_ids = full_mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(full_mask == 0, 1)

            # Main LLM Loss
            out = self.llm(inputs_embeds=inp, attention_mask=full_mask, position_ids=pos_ids, labels=full_labels, use_cache=False)
            main_loss = out.loss
            
            # CTC Auxiliary Loss
            ctc_loss = torch.tensor(0.0, device=device)
            if getattr(self, "ctc_head", None) is not None and self.config.ctc_loss_weight > 0:
                ctc_input = sub_audio # [Time, Batch, Vocab]
                ctc_logits = self.ctc_head(ctc_input).transpose(0, 1) # [T, B, Vocab]
                ctc_log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
                
                # Prepare CTC Targets (Filter -100 padding)
                target_list = []
                target_lengths = []
                raw_labels = labels[idx]
                
                for k in range(B_sub):
                    valid_tokens = raw_labels[k][raw_labels[k] != -100]
                    target_list.append(valid_tokens)
                    target_lengths.append(len(valid_tokens))
                
                if len(target_list) > 0:
                    ctc_targets = torch.cat(target_list)
                    target_lengths = torch.tensor(target_lengths, device=device, dtype=torch.long)
                    input_lengths = audio_mask[idx].sum(dim=1)
                    ctc_loss = self.ctc_loss_fct(ctc_log_probs, ctc_targets, input_lengths, target_lengths)
            
            total_asr_loss = main_loss + self.config.ctc_loss_weight * ctc_loss
            total_loss += total_asr_loss
            accum_asr_loss += main_loss
            asr_count += 1
            valid_samples += 1

        # --- 5. Aggregation ---
        if valid_samples > 0: 
            total_loss = total_loss / valid_samples
        
        avg_tts = accum_tts_loss / max(tts_count, 1)
        avg_asr = accum_asr_loss / max(asr_count, 1)

        return {"loss": total_loss, "loss_tts": avg_tts, "loss_asr": avg_asr}

    def save_pretrained(self, save_directory: str, **kwargs):
        self.config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory)
        state_dict = kwargs.get("state_dict", None)
        
        def save_part(prefix, filename):
            if state_dict is None: 
                if hasattr(self, prefix): 
                    torch.save(getattr(self, prefix).state_dict(), os.path.join(save_directory, filename))
            else:
                sd = {k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
                torch.save(sd, os.path.join(save_directory, filename))

        save_part("input_proj", "input_proj.bin")
        save_part("output_head", "output_head.bin")