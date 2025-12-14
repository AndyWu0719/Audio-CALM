import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from models.modeling_vae import AcousticVAE
import os
import math

# =============================================================================
# 1. GMM Head & Projector
# =============================================================================

class AudioInputProjector(nn.Module):
    def __init__(self, latent_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, llm_dim),
            nn.LayerNorm(llm_dim, eps=1e-6),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class GMMHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures

        self.proj = nn.Linear(input_dim, num_mixtures * (2 * output_dim + 1))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, S, _ = x.shape
        params = self.proj(x)

        pi_logits, split_rem = params.split([self.num_mixtures, 2 * self.num_mixtures * self.output_dim], dim=-1)
        
        # Unbind Mu and Sigma
        # View: [B, S, K, 2, D] -> Unbind dim 3
        mu, log_sigma = split_rem.view(B, S, self.num_mixtures, 2, self.output_dim).unbind(dim=3)

        log_sigma = torch.clamp(log_sigma, min=-5.0, max=3.0)

        return pi_logits, mu, log_sigma

def masked_gmm_loss(target, pi_logits, mu, log_sigma, mask):
    """
    计算 Negative Log Likelihood (NLL) Loss
    """
    target = target.float()          # [B,S,D]
    pi_logits = pi_logits.float()    # [B,S,K]
    mu = mu.float()                  # [B,S,K,D]
    log_sigma = log_sigma.float()    # [B,S,K,D]
    mask = mask.bool()               # [B,S]

    # 1. Log Probability
    # formula: -0.5 * (log(2pi) + 2*log_sigma + (x-mu)^2/var)
    var = torch.exp(2 * log_sigma)
    # dist_sq: [B, S, K, D]
    dist_sq = (target.unsqueeze(2) - mu) ** 2 
    
    log_prob_per_dim = -0.5 * (math.log(2 * math.pi) + 2 * log_sigma + dist_sq / (var + 1e-8))
    
    # Sum over dimensions D -> Log Prob per Component k
    log_prob_comp = log_prob_per_dim.sum(dim=-1) # [B, S, K]

    # 2. Pi
    # LogSumExp(Log_Pi + Log_Prob_Component)
    log_pi = F.log_softmax(pi_logits, dim=-1) # 确保 sum(pi)=1
    log_likelihood = torch.logsumexp(log_pi + log_prob_comp, dim=-1) # [B, S]

    # 3. Apply Mask
    log_likelihood = log_likelihood * mask
    
    # 4. Average Loss (Negative NLL)
    denom = mask.sum().clamp(min=1)
    return -(log_likelihood.sum() / denom)

# =============================================================================
# 2. QwenCALM Model
# =============================================================================

class QwenCALMConfig(PretrainedConfig):
    model_type = "qwen_calm"
    def __init__(self, 
                 qwen_path=None, 
                 vae_path=None, 
                 num_mixtures=8, 
                 use_precomputed_latents=False, 
                 latent_dim=64, 
                 audio_loss_weight=1.0,
                 downsample_rate=16, 
                 **kwargs):
        super().__init__(**kwargs)
        self.qwen_path = qwen_path
        self.vae_path = vae_path
        self.num_mixtures = num_mixtures
        self.use_precomputed_latents = use_precomputed_latents
        self.latent_dim = latent_dim
        self.audio_loss_weight = audio_loss_weight
        self.downsample_rate = downsample_rate

class QwenCALM(PreTrainedModel):
    config_class = QwenCALMConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        print(f"Loading Qwen from {config.qwen_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        self.use_precomputed_latents = config.use_precomputed_latents
        vae_latent_dim = 64

        if not self.use_precomputed_latents:
            print(f"Loading VAE from {config.vae_path}...")
            self.vae = AcousticVAE.from_pretrained(config.vae_path)
            self.vae.requires_grad_(False)
            self.vae.eval()
            vae_latent_dim = self.vae.config.latent_channels
        else:
            print("Skipping VAE loading (Using precomputed latents).")
            vae_latent_dim = config.latent_dim

        qwen_dim = self.llm.config.hidden_size
        
        self.input_proj = AudioInputProjector(vae_latent_dim, qwen_dim)
        self.output_head = GMMHead(qwen_dim, vae_latent_dim, num_mixtures=config.num_mixtures)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(self, 
                text_input_ids,          # [B, T_text]
                audio_features,          # [B, D, T_audio] or [B, T_audio, D]
                attention_mask=None,     # [B, T_text]
                labels=None,             # [B, T_text]
                task_modes=None,         # List[str] "tts" or "asr"
                audio_lens=None):        # [B]
        
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device
        if task_modes is None: task_modes = ["tts"] * batch_size

        # 1. Latents
        with torch.no_grad():
            if self.use_precomputed_latents:
                # B, D, T] -> [B, T, D]
                gt_latents = audio_features.transpose(1, 2)
            else:
                # 输入 Mel [B, 80, T] -> VAE Encode -> [B, 64, T] -> [B, T, 64]
                mu, _ = self.vae.encode(audio_features) 
                gt_latents = mu.transpose(1, 2)

        # 2. Audio Mask
        B_aud, T_aud, _ = gt_latents.shape
        if audio_lens is not None:
            if self.use_precomputed_latents:
                latent_lens = audio_lens
            else:
                downsample_rate = self.config.downsample_rate 
                latent_lens = torch.div(audio_lens + downsample_rate - 1, downsample_rate, rounding_mode='floor')
            
            # Mask: [B, T_aud] (1=Valid, 0=Pad)
            audio_mask = torch.arange(T_aud, device=device)[None, :] < latent_lens[:, None]
            audio_mask = audio_mask.to(dtype=torch.long)
        else:
            audio_mask = torch.ones((B_aud, T_aud), device=device, dtype=torch.long)
            
        # 3. Embedding
        # Audio Latents -> LLM Space
        gt_latents = gt_latents.to(dtype=self.input_proj.net[0].weight.dtype)
        audio_embeds = self.input_proj(gt_latents)  # [B, T_aud, 4096]
        # Text IDs -> LLM Space
        text_embeds = self.get_input_embeddings()(text_input_ids) # [B, T_text, 4096]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, text_input_ids.shape[1]), device=device, dtype=torch.long)

        total_loss = torch.tensor(0.0, device=device)
        accum_tts_loss = torch.tensor(0.0, device=device)
        accum_asr_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        tts_count = 0
        asr_count = 0

        # =====================================================================
        # Task 1: TTS (Text -> Audio)
        # Text + Audio(0...T-1), Audio(1...T)
        # =====================================================================
        tts_indices = [i for i, m in enumerate(task_modes) if m == "tts"]
        if len(tts_indices) > 0:
            idx = torch.tensor(tts_indices, device=device)

            sub_text = text_embeds[idx]       
            sub_audio = audio_embeds[idx]     
            sub_gt = gt_latents[idx]
            sub_text_mask = attention_mask[idx]
            sub_audio_mask = audio_mask[idx]

            # [Text, Audio[:-1]]
            # Teacher Forcing
            inp = torch.cat([sub_text, sub_audio[:, :-1, :]], dim=1)

            # Mask
            full_mask = torch.cat([sub_text_mask, sub_audio_mask[:, :-1]], dim=1)

            # Position IDs
            position_ids = full_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(full_mask == 0, 1)

            # LLM Forward
            out = self.llm(
                inputs_embeds=inp, 
                attention_mask=full_mask, 
                position_ids=position_ids, 
                output_hidden_states=True
            )
            last_hidden = out.hidden_states[-1]
            txt_len = sub_text.shape[1]
            audio_hidden = last_hidden[:, txt_len-1:, :] 

            # GMM Head
            pi, mu, log_sigma = self.output_head(audio_hidden)

            tts_mask = sub_audio_mask[:, :sub_gt.size(1)].bool() # [B, T_aud]
            
            tts_step_loss = masked_gmm_loss(sub_gt, pi, mu, log_sigma, tts_mask)
            
            total_loss += tts_step_loss * self.config.audio_loss_weight
            
            valid_samples += 1
            accum_tts_loss += tts_step_loss
            tts_count += 1

        # =====================================================================
        # Task 2: ASR (Audio -> Text)
        # Audio + Text, Text (Next Token)
        # =====================================================================
        asr_indices = [i for i, m in enumerate(task_modes) if m == "asr"]
        if len(asr_indices) > 0:
            idx = torch.tensor(asr_indices, device=device)

            sub_audio = audio_embeds[idx]
            sub_text = text_embeds[idx]
            sub_labels = labels[idx]
            sub_audio_mask = audio_mask[idx]
            sub_text_mask = attention_mask[idx]

            inp = torch.cat([sub_audio, sub_text], dim=1)
            full_mask = torch.cat([sub_audio_mask, sub_text_mask], dim=1)

            B_sub, T_aud, _ = sub_audio.shape
            prefix_labels = torch.full((B_sub, T_aud), -100, dtype=torch.long, device=device)
            full_labels = torch.cat([prefix_labels, sub_labels], dim=1)

            # Position IDs
            position_ids = full_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(full_mask == 0, 1)

            out = self.llm(
                inputs_embeds=inp,
                attention_mask=full_mask,
                position_ids=position_ids,
                labels=full_labels,
                use_cache=False,
            )
            loss_asr = out.loss

            total_loss += loss_asr
            valid_samples += 1
            accum_asr_loss += loss_asr
            asr_count += 1

        if len(tts_indices) == 0:
            dummy_in = audio_embeds[:, :1, :] * 0 
            pi, mu, log_sigma = self.output_head(dummy_in)
            total_loss += (pi.sum() + mu.sum() + log_sigma.sum()) * 0.0

        if len(asr_indices) == 0 and self.llm.lm_head.weight.requires_grad:
             dummy_in = text_embeds[:, :1, :] * 0
             dummy_logits = self.llm.lm_head(dummy_in)
             total_loss += dummy_logits.sum() * 0.0

        if valid_samples > 0:
            total_loss = total_loss / valid_samples

        avg_tts = accum_tts_loss / tts_count if tts_count > 0 else torch.tensor(0.0, device=device)
        avg_asr = accum_asr_loss / asr_count if asr_count > 0 else torch.tensor(0.0, device=device)

        return {
            "loss": total_loss,
            "loss_tts": avg_tts,
            "loss_asr": avg_asr
        }

    def save_pretrained(self, save_directory, **kwargs):
        self.config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory) 

        state_dict = kwargs.get("state_dict", None)

        def save_part(prefix, filename):
            if state_dict is None:
                module = getattr(self, prefix)
                sd = module.state_dict()
            else:
                sd = {k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
            torch.save(sd, os.path.join(save_directory, filename))

        save_part("input_proj", "input_proj.bin")
        save_part("output_head", "output_head.bin")