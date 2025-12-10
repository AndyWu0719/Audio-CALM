import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from models.modeling_vae import AcousticVAE
import os
import math

# =============================================================================
# Energy-Based Components (Ported from CALM)
# =============================================================================

class MLPBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(2 * channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True)
        )
        self.gate_act = nn.SiLU()
        self.down_proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x, y):
        # x: noise embedding, y: condition (LLM hidden state)
        # Concatenate along the last dimension
        h = self.linears(torch.cat((self.in_ln(x), y), dim=-1))
        gate_proj, up_proj = torch.chunk(h, 2, dim=-1)
        gate_proj = self.gate_act(gate_proj)
        step = self.down_proj(gate_proj * up_proj)
        return x + step

class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(model_channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(model_channels, model_channels, bias=True),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels, bias=True)
        )

    def forward(self, x):
        return self.linears(self.in_ln(x))

class MLPGenerator(nn.Module):
    """
    Energy-based Implicit Generator.
    """
    def __init__(self, input_dim, output_dim, noise_size=64, num_mlp_layers=2):
        super().__init__()
        self.noise_size = noise_size
        self.input_dim = input_dim
        
        self.noise_embd = nn.Linear(noise_size, input_dim)
        self.hidden_embd = nn.Linear(input_dim, input_dim)
        self.norm_hidden = nn.LayerNorm(input_dim, eps=1e-6)
        self.norm_noise = nn.LayerNorm(input_dim, eps=1e-6)

        mlp_blocks = []
        for i in range(num_mlp_layers):
            mlp_blocks.append(MLPBlock(input_dim))
        self.mlp_blocks = nn.ModuleList(mlp_blocks)
        self.final_layer = FinalLayer(input_dim, output_dim) # output_dim 是 VAE latent dim

    def initialize_weights(self):
        nn.init.constant_(self.final_layer.linears[-1].weight, 0)
        nn.init.constant_(self.final_layer.linears[-1].bias, 0)

    def sample(self, hidden_states, temperature=1.0):
        # hidden_states shape: [Batch, ..., HiddenDim]
        noise = (torch.rand((*hidden_states.shape[:-1], self.noise_size),
                           dtype=hidden_states.dtype, device=hidden_states.device) - 0.5) * temperature
        
        noise_embds = self.norm_noise(self.noise_embd(noise))
        hidden_states_norm = self.norm_hidden(self.hidden_embd(hidden_states))

        for block in self.mlp_blocks:
            noise_embds = block(noise_embds, hidden_states_norm)

        latent_prediction = self.final_layer(noise_embds)
        return latent_prediction

# =============================================================================
# Qwen-CALM Model Definition
# =============================================================================

class QwenCALMConfig(PretrainedConfig):
    model_type = "qwen_calm"
    def __init__(self, 
                 qwen_path=None, 
                 vae_path=None, 
                 use_precomputed_latents=False, 
                 latent_dim=64,
                 # Energy-specific configs
                 noise_size=64,
                 num_mlp_layers=2,
                 num_samples=8,      # Number of samples to draw during training
                 beta=0.25,
                 temperature=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.qwen_path = qwen_path
        self.vae_path = vae_path
        self.use_precomputed_latents = use_precomputed_latents
        self.latent_dim = latent_dim
        self.noise_size = noise_size
        self.num_mlp_layers = num_mlp_layers
        self.num_samples = num_samples
        self.beta = beta
        self.temperature = temperature

class QwenCALM(PreTrainedModel):
    config_class = QwenCALMConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        
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
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
            vae_latent_dim = self.vae.config.latent_channels
        else:
            print("Skipping VAE loading (Using precomputed latents).")
            vae_latent_dim = config.latent_dim

        qwen_dim = self.llm.config.hidden_size
        self.input_proj = nn.Linear(vae_latent_dim, qwen_dim)
        
        self.output_head = MLPGenerator(
            input_dim=qwen_dim, 
            output_dim=vae_latent_dim,
            noise_size=config.noise_size,
            num_mlp_layers=config.num_mlp_layers
        )
        self.output_head.initialize_weights()

        # Energy Loss parameters
        self.num_samples = config.num_samples
        self.beta = config.beta
        self.temperature = config.temperature
        
        nn.init.normal_(self.input_proj.weight, std=0.02)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        try:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    # --- Energy Score Calculation ---
    def distance(self, x_1, x_2):
        return torch.pow(torch.linalg.norm(x_1 - x_2, ord=2, dim=-1) + 1e-8, self.beta)
    
    def energy_score(self, x, mean, log_std):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        mean = mean.to(torch.float32)
        log_std = log_std.to(torch.float32)

        n_x = x.shape[0]
        x_i = x.unsqueeze(1)  
        x_j = x.unsqueeze(0)  
        distance_matrix = self.distance(x_i, x_j)
        
        distance_x = distance_matrix.sum(dim=(0,1)) / (n_x * (n_x - 1))

        std = torch.exp(log_std)
        n_y = 100 
        eps = torch.randn((n_y, *mean.shape), device=mean.device)
        y = mean.unsqueeze(0) + eps * std.unsqueeze(0) 

        x_ = x.unsqueeze(1)       
        y_ = y.unsqueeze(0)       
        distance_y = self.distance(x_, y_).mean(dim=(0, 1))
        
        score = distance_x - distance_y * 2
        
        return score, distance_x, distance_y

    def forward(self, text_input_ids, audio_features, attention_mask=None, labels=None, task_modes=None, audio_lens=None):
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device
        
        if task_modes is None:
            task_modes = ["tts"] * batch_size

        # 1. Latents Preparation
        with torch.no_grad():
            if self.use_precomputed_latents:
                gt_latents = audio_features.transpose(1, 2)
                # Fallback if precomputed latents don't have variance info
                gt_mean = gt_latents
                gt_log_std = torch.zeros_like(gt_mean) - 5.0 
            else:
                # [Important] Ensure VAE returns (mu, log_var)
                # If your VAE only returns one value, check modeling_vae.py
                vae_out = self.vae.encode(audio_features)
                if isinstance(vae_out, tuple):
                    mu, log_var = vae_out
                else:
                    mu = vae_out
                    log_var = torch.zeros_like(mu) - 5.0 # Fallback
                
                gt_mean = mu.transpose(1, 2)
                raw_log_std = log_var.transpose(1, 2) * 0.5
                gt_log_std = torch.clamp(raw_log_std, min=-5.0, max=3.0).contiguous()

        # 2. Audio Mask
        B_aud, T_aud, _ = gt_mean.shape
        if audio_lens is not None:
            if self.use_precomputed_latents:
                latent_lens = audio_lens
            else:
                downsample_rate = getattr(self.vae, "total_stride", 16)
                latent_lens = torch.div(
                    audio_lens + downsample_rate - 1,
                    downsample_rate,
                    rounding_mode='floor'
                )
            
            audio_mask = torch.arange(T_aud, device=device)[None, :] < latent_lens[:, None]
            audio_mask = audio_mask.to(dtype=torch.long)
        else:
            audio_mask = torch.ones((B_aud, T_aud), device=device, dtype=torch.long)

        # 3. Project to LLM dimension
        audio_embeds = self.input_proj(gt_mean) 
        text_embeds = self.llm.transformer.wte(text_input_ids) 
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, text_input_ids.shape[1]), device=device, dtype=torch.long)

        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        accum_tts_loss = torch.tensor(0.0, device=device)
        accum_asr_loss = torch.tensor(0.0, device=device)
        accum_div_loss = torch.tensor(0.0, device=device)
        accum_fid_loss = torch.tensor(0.0, device=device)
        tts_count = 0
        asr_count = 0
        
        # === TTS (Text -> Audio) ===
        tts_indices = [i for i, m in enumerate(task_modes) if m == "tts"]
        if len(tts_indices) > 0:
            idx = torch.tensor(tts_indices, device=device)
            
            sub_text = text_embeds[idx]       
            sub_audio = audio_embeds[idx]     
            sub_text_mask = attention_mask[idx]
            sub_audio_mask = audio_mask[idx]
            
            # Input construction: [Pad, Text, Audio_Prefix]
            inp = torch.cat([sub_text, sub_audio[:, :-1, :]], dim=1)
            full_mask = torch.cat([sub_text_mask, sub_audio_mask[:, :-1]], dim=1)
            
            position_ids = full_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(full_mask == 0, 1)
            
            out = self.llm(
                inputs_embeds=inp, 
                attention_mask=full_mask, 
                position_ids=position_ids, 
                output_hidden_states=True
            )
            last_hidden = out.hidden_states[-1]
            
            # Extract Audio Part Hidden States
            txt_len = sub_text.shape[1]
            audio_hidden = last_hidden[:, txt_len-1:, :] 
            
            # Prepare Ground Truth
            sub_gt_mean = gt_mean[idx]
            sub_gt_log_std = gt_log_std[idx]
            
            # --- Energy Loss Calculation (Optimized with Masking) ---
            # Instead of masking the loss, we select valid tokens first
            tts_mask = sub_audio_mask[:, :sub_gt_mean.size(1)].bool()
            
            # Flatten: Select only valid frames across the batch
            valid_hidden = audio_hidden[tts_mask]       # [N_total_frames, 4096]
            valid_gt_mean = sub_gt_mean[tts_mask]       # [N_total_frames, 64]
            valid_gt_log_std = sub_gt_log_std[tts_mask] # [N_total_frames, 64]
            
            if valid_hidden.shape[0] > 0:
                # Repeat hidden states for multiple samples [num_samples, N_frames, dim]
                hidden_repeated = valid_hidden.unsqueeze(0).repeat(self.num_samples, 1, 1)
                
                # Generate predictions
                latent_pred = self.output_head.sample(hidden_repeated, temperature=self.temperature)
                
                # Calculate Energy Score (we minimize NEGATIVE score)
                raw_score, raw_div, raw_fid = self.energy_score(latent_pred, valid_gt_mean, valid_gt_log_std)
                
                tts_step_loss = -raw_score.mean()
                div_val = raw_div.mean()
                fid_val = raw_fid.mean()
                
                total_loss += tts_step_loss
                accum_tts_loss += tts_step_loss
                accum_div_loss += div_val
                accum_fid_loss += fid_val
                
                tts_count += 1
                valid_samples += 1

        # === ASR (Audio -> Text) ===
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
                output_hidden_states=True
            )
            
            logits = self.llm.lm_head(out.hidden_states[-1])
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss_asr = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            total_loss += loss_asr
            valid_samples += 1
            accum_asr_loss += loss_asr
            asr_count += 1

        # Handling dummy loss for DDP to avoid unused parameter errors
        if len(tts_indices) == 0:
            dummy_hidden = audio_embeds[:, :1, :].mean() 
            dummy_in = torch.zeros((1, 1, 4096), device=device, dtype=audio_embeds.dtype)
            dummy_pred = self.output_head.sample(dummy_in)
            total_loss += (dummy_pred.sum() * 0.0) + (dummy_hidden * 0.0)

        if len(asr_indices) == 0 and self.llm.lm_head.weight.requires_grad:
             dummy_in = text_embeds[:, :1, :] * 0
             dummy_logits = self.llm.lm_head(dummy_in)
             total_loss += dummy_logits.sum() * 0.0

        if valid_samples > 1: # Already summed, just normalization if needed? 
            # In mixed training, usually we average over tasks or just sum. 
            # Here keeping simple sum / count logic.
            # If tts_count=1 and asr_count=1, total_loss has sum. 
            # Let's just divide by valid_samples to denote "loss per task batch"
            total_loss = total_loss / valid_samples
            
        avg_tts = accum_tts_loss / tts_count if tts_count > 0 else torch.tensor(0.0, device=device)
        avg_asr = accum_asr_loss / asr_count if asr_count > 0 else torch.tensor(0.0, device=device)
        avg_div = accum_div_loss / tts_count if tts_count > 0 else torch.tensor(0.0, device=device)
        avg_fid = accum_fid_loss / tts_count if tts_count > 0 else torch.tensor(0.0, device=device)
        
        return {
            "loss": total_loss,
            "loss_tts": avg_tts,
            "loss_asr": avg_asr,
            "loss_diversity": avg_div,
            "loss_fidelity": avg_fid,
            "logits": None 
        }
    
    def save_pretrained(self, save_directory, **kwargs):
        self.config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory) 
        
        state_dict = kwargs.get("state_dict", None)
        
        # Helper to save specific modules
        def save_module(module, name, state_dict=None):
            if state_dict is None:
                sd = module.state_dict()
            else:
                sd = {k.replace(f"{name}.", ""): v for k, v in state_dict.items() if k.startswith(f"{name}.")}
            torch.save(sd, os.path.join(save_directory, f"{name}.bin"))

        save_module(self.input_proj, "input_proj", state_dict)
        save_module(self.output_head, "output_head", state_dict)