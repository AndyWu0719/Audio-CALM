"""
GMM-based audio head and Qwen-based multimodal model for Audio-CALM.
Cleaned, split into sections, and annotated with concise English comments.
"""

import os
import math
from typing import Dict, Any

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
# Audio Input Projector
# ---------------------------------------------------------------------
class AudioInputProjector(nn.Module):
    """
    Project latent audio frames (T, D) to LLM embedding space (T, H).
    Accepts input as [B, T, D] and returns [B, T, H].
    """

    def __init__(self, latent_dim: int, llm_dim: int, max_audio_len: int = 1024):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(latent_dim, llm_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(llm_dim, llm_dim, kernel_size=3, padding=1),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(llm_dim, eps=1e-6),
                    nn.Linear(llm_dim, llm_dim * 2),
                    nn.GELU(),
                    nn.Linear(llm_dim * 2, llm_dim),
                )
                for _ in range(2)
            ]
        )
        self.post_norm = nn.LayerNorm(llm_dim, eps=1e-6)
        self.pos_emb = nn.Embedding(max_audio_len, llm_dim)
        self.max_audio_len = max_audio_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, _ = x.shape
        device = x.device

        # conv expects [B, D, T]
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = x + block(x)
        x = self.post_norm(x)

        # Positional embeddings: support T > max_audio_len by zero-padding beyond max.
        T_clamped = min(T, self.max_audio_len)
        pos_ids = torch.arange(T_clamped, device=device).unsqueeze(0).expand(B, -1)

        if T > self.max_audio_len:
            pos_emb_full = torch.zeros(B, T, x.size(-1), device=device, dtype=x.dtype)
            pos_emb_full[:, :T_clamped, :] = self.pos_emb(pos_ids)
            return x + pos_emb_full
        else:
            return x + self.pos_emb(pos_ids)

# ---------------------------------------------------------------------
# GMM Head
# ---------------------------------------------------------------------
class GMMHead(nn.Module):
    """
    Predict GMM parameters (pi logits, mu, log_sigma) from LLM hidden states.
    Input: [B, S, H] -> outputs:
        pi_logits: [B, S, K]
        mu: [B, S, K, D]
        log_sigma: [B, S, K, D]
    """

    def __init__(self, input_dim: int, output_dim: int, num_mixtures: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures
        self.proj = nn.Linear(input_dim, num_mixtures * (2 * output_dim + 1))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        B, S, _ = x.shape
        params = self.proj(x)
        pi_logits, split_rem = params.split([self.num_mixtures, 2 * self.num_mixtures * self.output_dim], dim=-1)
        mu, log_sigma = split_rem.view(B, S, self.num_mixtures, 2, self.output_dim).unbind(dim=3)
        # Keep log_sigma within a reasonable range for stability
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=3.0)
        return pi_logits, mu, log_sigma

# ---------------------------------------------------------------------
# Loss: masked GMM negative log-likelihood
# ---------------------------------------------------------------------
def masked_gmm_loss(
    target: torch.Tensor,
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    mask: torch.Tensor,
    avg_over_dim: bool = True,
) -> torch.Tensor:
    """
    Compute masked negative log-likelihood under a diagonal GMM.
    Returns scalar loss averaged over valid (masked) time steps and optionally normalized by target dim.
    """
    target = target.float()
    pi_logits = pi_logits.float()
    mu = mu.float()
    log_sigma = log_sigma.float()
    mask = mask.bool()

    var = torch.exp(2 * log_sigma)
    dist_sq = (target.unsqueeze(2) - mu) ** 2  # [B, T, K, D]

    # log prob per dim: [B, T, K, D]
    log_prob_per_dim = -0.5 * (math.log(2 * math.pi) + 2 * log_sigma + dist_sq / (var + EPS))
    # sum over D -> [B, T, K]
    log_prob_comp = log_prob_per_dim.sum(dim=-1)

    log_pi = F.log_softmax(pi_logits, dim=-1)
    # log-sum-exp over mixtures -> [B, T]
    log_likelihood = torch.logsumexp(log_pi + log_prob_comp, dim=-1)

    log_likelihood = log_likelihood * mask
    denom = mask.sum().clamp(min=1)
    nll = -(log_likelihood.sum() / denom)

    if avg_over_dim:
        nll = nll / target.shape[-1]

    return nll

# ---------------------------------------------------------------------
# Model Config and QwenCALM Model
# ---------------------------------------------------------------------
class QwenCALMConfig(PretrainedConfig):
    model_type = "qwen_calm"

    def __init__(
        self,
        qwen_path: str = None,
        vae_path: str = None,
        num_mixtures: int = 8,
        use_precomputed_latents: bool = False,
        latent_dim: int = 64,
        audio_loss_weight: float = 1.0,
        downsample_rate: int = 16,
        max_audio_len: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.qwen_path = qwen_path
        self.vae_path = vae_path
        self.num_mixtures = num_mixtures
        self.use_precomputed_latents = use_precomputed_latents
        self.latent_dim = latent_dim
        self.audio_loss_weight = audio_loss_weight
        self.downsample_rate = downsample_rate
        self.max_audio_len = max_audio_len


class QwenCALM(PreTrainedModel):
    """
    Wrapper combining a Qwen causal LM and a GMM head to perform TTS/ASR tasks.
    """

    config_class = QwenCALMConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config: QwenCALMConfig):
        super().__init__(config)
        self.config = config

        print(f"Loading Qwen from {config.qwen_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
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
        max_audio_len = getattr(config, "max_audio_len", 1024)

        self.input_proj = AudioInputProjector(vae_latent_dim, qwen_dim, max_audio_len=max_audio_len)
        self.output_head = GMMHead(qwen_dim, vae_latent_dim, num_mixtures=config.num_mixtures)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(
        self,
        text_input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        task_modes: list = None,
        audio_lens: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """
        Supports a mixed batch of samples labeled as 'tts' or 'asr' in task_modes.
        Returns dict with 'loss', 'loss_tts', 'loss_asr'.
        """
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device
        if task_modes is None:
            task_modes = ["tts"] * batch_size

        with torch.no_grad():
            if self.use_precomputed_latents:
                # collator provides [B, D, T]; convert to [B, T, D]
                gt_latents = audio_features.transpose(1, 2)
            else:
                mu, _ = self.vae.encode(audio_features)
                gt_latents = mu.transpose(1, 2)

        B_aud, T_aud, _ = gt_latents.shape
        if audio_lens is not None:
            if self.use_precomputed_latents:
                latent_lens = audio_lens
            else:
                downsample_rate = self.config.downsample_rate
                latent_lens = torch.div(audio_lens + downsample_rate - 1, downsample_rate, rounding_mode="floor")

            audio_mask = (torch.arange(T_aud, device=device)[None, :] < latent_lens[:, None]).long()
        else:
            audio_mask = torch.ones((B_aud, T_aud), device=device, dtype=torch.long)

        gt_latents = gt_latents.to(dtype=self.llm.dtype)
        audio_embeds = self.input_proj(gt_latents)  # [B, T, H]
        text_embeds = self.get_input_embeddings()(text_input_ids)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, text_input_ids.shape[1]), device=device, dtype=torch.long)

        total_loss = torch.tensor(0.0, device=device)
        accum_tts_loss = torch.tensor(0.0, device=device)
        accum_asr_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        tts_count = 0
        asr_count = 0

        # ----------------------------
        # TTS path
        # ----------------------------
        tts_indices = [i for i, m in enumerate(task_modes) if m == "tts"]
        if len(tts_indices) > 0:
            idx = torch.tensor(tts_indices, device=device)
            sub_text = text_embeds[idx]
            sub_audio = audio_embeds[idx]
            sub_gt = gt_latents[idx]
            sub_text_mask = attention_mask[idx]
            sub_audio_mask = audio_mask[idx]

            # Input to LLM: text + audio[:-1] (autoregressive)
            inp = torch.cat([sub_text, sub_audio[:, :-1, :]], dim=1)
            full_mask = torch.cat([sub_text_mask, sub_audio_mask[:, :-1]], dim=1)

            position_ids = full_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(full_mask == 0, 1)

            out = self.llm(inputs_embeds=inp, attention_mask=full_mask, position_ids=position_ids, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]

            txt_len = sub_text.shape[1]
            # Include last text hidden + audio hidden states to align with GT positions
            audio_hidden = last_hidden[:, txt_len - 1 :, :]

            pi, mu, log_sigma = self.output_head(audio_hidden)
            tts_mask = sub_audio_mask[:, : sub_gt.size(1)].bool()

            tts_step_loss = masked_gmm_loss(sub_gt, pi, mu, log_sigma, tts_mask, avg_over_dim=True)

            total_loss += tts_step_loss * self.config.audio_loss_weight
            valid_samples += 1
            accum_tts_loss += tts_step_loss
            tts_count += 1

        # ----------------------------
        # ASR path
        # ----------------------------
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

        # Keep graph intact when one mode is missing
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

        return {"loss": total_loss, "loss_tts": avg_tts, "loss_asr": avg_asr}

    def save_pretrained(self, save_directory: str, **kwargs):
        self.config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory)

        state_dict = kwargs.get("state_dict", None)

        def save_part(prefix: str, filename: str):
            if state_dict is None:
                module = getattr(self, prefix)
                sd = module.state_dict()
            else:
                sd = {k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
            torch.save(sd, os.path.join(save_directory, filename))

        save_part("input_proj", "input_proj.bin")
        save_part("output_head", "output_head.bin")