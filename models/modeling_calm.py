import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from models.modeling_vae import AcousticVAE
import os
import math

class GMMHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures
        
        self.proj = nn.Linear(input_dim, num_mixtures * (2 * output_dim + 1))

    def forward(self, x):
        B, S, _ = x.shape
        params = self.proj(x)
        
        pi_logits, split_rem = params.split([self.num_mixtures, 2 * self.num_mixtures * self.output_dim], dim=-1)
        mu, log_sigma = split_rem.view(B, S, self.num_mixtures, 2, self.output_dim).unbind(dim=3)
        
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=2.0)
        
        return pi_logits, mu, log_sigma

def gmm_loss(target, pi_logits, mu, log_sigma):
    B, S, D = target.shape
    target = target.unsqueeze(2) 
    
    var = torch.exp(2 * log_sigma)
    log_prob_comp = -0.5 * ((target - mu)**2 / var) - log_sigma - 0.5 * math.log(2 * math.pi)
    log_prob_comp = log_prob_comp.sum(dim=-1)
    
    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_likelihood = torch.logsumexp(log_pi + log_prob_comp, dim=-1)
    
    return -torch.mean(log_likelihood)

class QwenCALMConfig(PretrainedConfig):
    model_type = "qwen_calm"
    def __init__(self, qwen_path=None, vae_path=None, num_mixtures=8, use_precomputed_latents=False, **kwargs):
        super().__init__(**kwargs)
        self.qwen_path = qwen_path
        self.vae_path = vae_path
        self.num_mixtures = num_mixtures
        self.use_precomputed_latents = use_precomputed_latents

class QwenCALM(PreTrainedModel):
    config_class = QwenCALMConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        
        print(f"Loading Qwen from {config.qwen_path}...")
        # [优化] 使用 low_cpu_mem_usage 加速加载并降低内存峰值
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True 
        )
        
        self.use_precomputed_latents = config.use_precomputed_latents
        
        vae_latent_dim = 64 # 默认值
        
        if not self.use_precomputed_latents:
            print(f"Loading VAE from {config.vae_path}...")
            self.vae = AcousticVAE.from_pretrained(config.vae_path)
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
            vae_latent_dim = self.vae.config.latent_channels
        else:
            print("Skipping VAE loading (Using precomputed latents).")
        
        qwen_dim = self.llm.config.hidden_size
        
        self.input_proj = nn.Linear(vae_latent_dim, qwen_dim)
        self.output_head = GMMHead(qwen_dim, vae_latent_dim, num_mixtures=config.num_mixtures)
        
        nn.init.normal_(self.input_proj.weight, std=0.02)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        try:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(self, text_input_ids, audio_features, attention_mask=None, labels=None, task_modes=None, audio_lens=None):
        """
        text_input_ids: [B, T_text] (Left padded)
        audio_features: [B, D, T] (可以是 Mel 也可以是 Latent，取决于 config)
        attention_mask: [B, T_text] (Text Mask)
        audio_lens: [B] (真实长度)
        """
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device
        
        if task_modes is None:
            task_modes = ["tts"] * batch_size

        # 1. 获取 Latents
        with torch.no_grad():
            if self.use_precomputed_latents:
                # audio_features 是 latents [B, 64, T] -> 转置为 [B, T, 64]
                # 这里假设 dataset 返回的是 [64, T]，并在 collator 堆叠成 [B, 64, T]
                gt_latents = audio_features.transpose(1, 2)
            else:
                # 实时计算 VAE: audio_features 是 Mel [B, 80, T]
                mu, _ = self.vae.encode(audio_features) 
                gt_latents = mu.transpose(1, 2)

        # 2. 构造 Audio Mask
        B_aud, T_aud, _ = gt_latents.shape
        if audio_lens is not None:
            if self.use_precomputed_latents:
                # audio_lens 已经是 latent 长度
                latent_lens = audio_lens
            else:
                # Mel 长度 -> Latent 长度 (除以16)
                downsample_rate = 16
                latent_lens = torch.div(audio_lens + downsample_rate - 1, downsample_rate, rounding_mode='floor')
            
            # 创建 Mask: [B, T_aud] (1 for valid, 0 for pad)
            audio_mask = torch.arange(T_aud, device=device)[None, :] < latent_lens[:, None]
            audio_mask = audio_mask.to(dtype=torch.long)
        else:
            audio_mask = torch.ones((B_aud, T_aud), device=device, dtype=torch.long)

        # 3. Project to LLM dimension
        audio_embeds = self.input_proj(gt_latents) # [B, T_aud, 4096]
        text_embeds = self.llm.transformer.wte(text_input_ids) # [B, T_text, 4096]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, text_input_ids.shape[1]), device=device, dtype=torch.long)

        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        # === TTS (Text -> Audio) ===
        tts_indices = [i for i, m in enumerate(task_modes) if m == "tts"]
        if len(tts_indices) > 0:
            idx = torch.tensor(tts_indices, device=device)
            
            sub_text = text_embeds[idx]       
            sub_audio = audio_embeds[idx]     
            sub_gt = gt_latents[idx]
            
            sub_text_mask = attention_mask[idx]
            sub_audio_mask = audio_mask[idx]
            
            # Input: Text + Audio_Prefix
            # Text 是 Left Padded: [Pad, Text]
            # Audio 是 Right Padded: [Audio, Pad]
            # 拼接时：[Pad, Text, Audio, Pad] (最后一个Pad不参与预测)
            
            # 取 Audio 的前 T-1 帧作为 Input
            inp = torch.cat([sub_text, sub_audio[:, :-1, :]], dim=1)
            
            # 拼接 Mask
            full_mask = torch.cat([sub_text_mask, sub_audio_mask[:, :-1]], dim=1)
            
            out = self.llm(inputs_embeds=inp, attention_mask=full_mask, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]
            
            txt_len = sub_text.shape[1]
            audio_hidden = last_hidden[:, txt_len-1:, :] 
            
            pi, mu, log_sigma = self.output_head(audio_hidden)
            
            # 计算 Loss (简单平均，忽略 Mask 为 0 的部分)
            # 更好的做法是用 Mask 过滤
            valid_len_mask = sub_audio_mask.bool()
            
            # 这里简化处理：循环计算 Valid Loss
            loss_list = []
            for b_i in range(len(idx)):
                # 获取该样本的有效 Latent 长度
                curr_len = sub_audio_mask[b_i].sum().item()
                if curr_len > 0:
                    l = gmm_loss(
                        sub_gt[b_i:b_i+1, :curr_len, :],
                        pi[b_i:b_i+1, :curr_len, :],
                        mu[b_i:b_i+1, :curr_len, :, :],
                        log_sigma[b_i:b_i+1, :curr_len, :, :]
                    )
                    loss_list.append(l)
            
            if len(loss_list) > 0:
                total_loss += torch.stack(loss_list).mean()
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
            
            # Input: [Audio, Text]
            inp = torch.cat([sub_audio, sub_text], dim=1)
            
            # Full Mask: [Audio_Mask, Text_Mask]
            # Audio Mask 会屏蔽掉 Audio 的 Padding
            full_mask = torch.cat([sub_audio_mask, sub_text_mask], dim=1)
            
            # Labels
            B_sub, T_aud, _ = sub_audio.shape
            prefix_labels = torch.full((B_sub, T_aud), -100, dtype=torch.long, device=device)
            full_labels = torch.cat([prefix_labels, sub_labels], dim=1)
            
            out = self.llm(inputs_embeds=inp, attention_mask=full_mask, output_hidden_states=True)
            
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

        # =================================================================
        # [关键修复] 确保所有参数参与计算，以便关闭 ddp_find_unused_parameters
        # =================================================================
        
        # 1. 检查是否用到了 Output Head (TTS Head)
        # 如果当前 Batch 没有 TTS 任务，或者虽然有但被 Mask 掉了，我们需要伪造一个计算
        if len(tts_indices) == 0:
            # 取一个 dummy 输入过一遍 output_head
            # 只要输入还在计算图中（比如 audio_embeds），梯度就能传导，虽然是 0
            dummy_hidden = audio_embeds[:, :1, :].mean() # Scalar
            # 我们需要用到 output_head 的参数
            # 简单办法：直接调用 output_head，然后 * 0
            # 注意：输入维度要匹配
            dummy_in = torch.zeros((1, 1, 4096), device=device, dtype=audio_embeds.dtype)
            # 为了让 dummy_in 连上计算图，我们可以用 audio_embeds * 0 + dummy_in
            dummy_in = audio_embeds[:, :1, :] * 0 
            
            pi, mu, log_sigma = self.output_head(dummy_in)
            # 将输出求和 * 0 加到 loss
            total_loss += (pi.sum() + mu.sum() + log_sigma.sum()) * 0.0

        # 2. 检查是否用到了 LM Head (ASR Head)
        # Qwen 的 lm_head 通常绑定在 embed_tokens 或者独立，如果是 LoRA，lm_head 可能不参与训练
        # 但如果 lm_head 参与训练（比如全量微调或特定设置），也需要处理。
        # 通常 LoRA 模式下 lm_head 是冻结的，不需要处理。
        # 但为了保险起见，如果 lm_head 需要梯度：
        if len(asr_indices) == 0 and self.llm.lm_head.weight.requires_grad:
             # 同样伪造计算
             dummy_in = text_embeds[:, :1, :] * 0
             dummy_logits = self.llm.lm_head(dummy_in)
             total_loss += dummy_logits.sum() * 0.0

        # =================================================================

        if valid_samples > 0:
            total_loss = total_loss / valid_samples
            
        return {
            "loss": total_loss,
            "logits": None 
        }
    
    def save_pretrained(self, save_directory, **kwargs):
        self.config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory) 
        
        state_dict = kwargs.get("state_dict", None)
        
        if state_dict is None:
            input_proj_state = self.input_proj.state_dict()
            output_head_state = self.output_head.state_dict()
        else:
            input_proj_state = {k.replace("input_proj.", ""): v for k, v in state_dict.items() if k.startswith("input_proj.")}
            output_head_state = {k.replace("output_head.", ""): v for k, v in state_dict.items() if k.startswith("output_head.")}
            
        torch.save(input_proj_state, os.path.join(save_directory, "input_proj.bin"))
        torch.save(output_head_state, os.path.join(save_directory, "output_head.bin"))