"""
Flow-based Audio-CALM: Qwen-based Multimodal Model.
Features: PyTorch SDPA, Flow Matching Head, ASR-Optimized Linear Projector.
FIXED: Right Padding Logic, SOA Token, and Correct Slicing.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
# 【对应关系】：导入 VAE 模型定义，用于在 CALM 中加载 VAE 权重进行编解码
from models.modeling_vae import AcousticVAE

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EPS = 1e-8

# ---------------------------------------------------------------------
# Audio Input Projector (Fixed with Offset Support)
# ---------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        # 左侧 Padding 大小 = kernel_size - 1
        self.pad_len = kernel_size - 1

    def forward(self, x):
        # x: [B, C, T]
        # 在时间轴 (最后一维) 的左侧填充
        x = F.pad(x, (self.pad_len, 0)) 
        return self.conv(x)
    
class AudioInputProjector(nn.Module):
    """
    功能：音频特征投影器。
    作用：将 VAE 的 Latent (64维) 映射到 Qwen LLM 的 Hidden Size (例如 3584维)，并添加位置编码。
    
    【对应关系】：
    - 输入：来自 VAE Encoder 的 latent vectors。
    - 输出：作为 `inputs_embeds` 的一部分输入给 Qwen LLM。
    """
    def __init__(self, latent_dim, llm_dim, max_audio_len=1024):
        super().__init__()
        
        # 1. 局部特征提取 (Local Feature Extraction)
        # 1. kernel_size=3, stride=1
        # 2. padding=0 (我们手动 Pad)
        # 3. 这样第 t 帧的输出只依赖于 [t-2, t-1, t] (取决于具体实现，或者 [t-1, t, t+1] if right pad)
        # 我们需要实现左侧 Padding，使得输出 t 只看 [t-2, t-1, t]
        
        self.conv_block = nn.Sequential(
            # 第一层
            CausalConv1d(latent_dim, llm_dim, kernel_size=3),
            nn.GELU(),
            # 第二层
            CausalConv1d(llm_dim, llm_dim, kernel_size=3),
        )
        
        # 2. 位置编码 (Positional Information)
        # 学习音频帧的绝对位置 Embedding。
        self.pos_emb = nn.Embedding(max_audio_len, llm_dim)
        self.max_audio_len = max_audio_len

        # 3. 深度混合 (Deep Context Mixing)
        # 使用 MLP 层进一步融合特征。
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(llm_dim, eps=1e-6),
                nn.Linear(llm_dim, llm_dim * 2),
                nn.GELU(),
                nn.Linear(llm_dim * 2, llm_dim),
            ) for _ in range(2)
        ])
        self.post_norm = nn.LayerNorm(llm_dim, eps=1e-6)
        
        # 初始化
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x, offset=0):
        """
        功能：前向投影计算。
        
        参数：
        - x: [Batch, Time, Dim] 输入音频特征
        - offset: 用于流式推理或自回归生成的起始位置索引。
        """
        # x: [Batch, Time, Dim]
        B, T, _ = x.shape
        device = x.device
        
        # 1. 卷积处理
        # 调整维度适配 Conv1d: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = x.transpose(1, 2)
        
        # 2. 位置编码 (Positional Embeddings)
        # 核心逻辑：根据 offset 生成正确的绝对位置 ID
        start_pos = offset
        end_pos = offset + T
        
        # 生成 ID: [offset, offset+1, ..., offset+T-1]
        pos_ids = torch.arange(start_pos, end_pos, device=device).unsqueeze(0).expand(B, -1)
        
        # 越界截断 (Clamping) 防止索引错误
        pos_ids_clamped = pos_ids.clamp(max=self.max_audio_len - 1)
        pos_emb_val = self.pos_emb(pos_ids_clamped)

        # 叠加位置编码
        x = x + pos_emb_val
            
        # 3. MLP 混合
        for block in self.blocks:
            x = x + block(x)
        
        return self.post_norm(x)

# ---------------------------------------------------------------------
# Flow Matching Head
# ---------------------------------------------------------------------
class FlowMatchingHead(nn.Module):
    """
    功能：流匹配生成头 (Generative Head)。
    作用：预测条件流匹配（Conditional Flow Matching）中的速度场（Velocity Field）。
    
    【对应关系】：
    - 输入：LLM 的输出隐状态 (Condition) + 当前噪声 Latent (xt) + 时间步 (t)。
    - 输出：预测的速度向量 v (用于 ODE 积分更新 xt)。
    """
    class SinusoidalPosEmb(nn.Module):
        """内部类：正弦位置编码，用于将标量时间步 t 映射为向量。"""
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
        # 时间步 MLP
        self.time_mlp = nn.Sequential(
            self.SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim), nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        # 输入投影：Condition + Noisy Latent + Time Emb
        self.in_proj = nn.Linear(input_dim + output_dim + self.time_dim, hidden_dim)
        
        # 残差网络主体
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
        # [重要] 零初始化：使初始预测接近 0，增加训练稳定性
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, condition, noisy_x, t, condition_mask=None):
        if t.dim() == 1: 
            t = t.unsqueeze(1).expand(-1, condition.size(1))
        
        # 1. CFG (Classifier-Free Guidance) 掩码
        # 如果训练时传入 mask，将对应样本的 condition 置零，训练无条件生成能力。
        if condition_mask is not None:
            mask_expanded = condition_mask.view(-1, 1, 1) if condition_mask.dim() == 1 else condition_mask.unsqueeze(-1)
            condition = condition * mask_expanded.to(dtype=condition.dtype)

        # 2. 获取时间嵌入
        t_emb = self.time_mlp(t.reshape(-1)).view(condition.shape[0], condition.shape[1], -1)
        
        # 3. 拼接所有输入特征
        x = torch.cat([condition, noisy_x, t_emb], dim=-1)
        x = self.in_proj(x)
        
        # 4. 通过残差层
        for layer in self.layers:
            x = x + layer(x)
            
        return self.out_proj(x)

# ---------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------
def compute_flow_loss(model_head, condition, target_latent, mask, cfg_dropout_prob=0.1):
    """
    功能：计算流匹配损失。
    
    参数：
    - target_latent: 真实的音频 Latent (x1)。
    - condition: LLM 输出的上下文特征。
    """
    B, T, D = target_latent.shape
    device = target_latent.device
    dtype = target_latent.dtype
    mask = mask.bool()
    
    # 1. 随机生成 CFG Dropout Mask
    if model_head.training and cfg_dropout_prob > 0:
        keep_prob = 1.0 - cfg_dropout_prob
        cfg_mask = torch.bernoulli(torch.full((B,), keep_prob, device=device)).to(dtype)
    else:
        cfg_mask = None 

    # 2. 构造流匹配训练样本
    # 2.1 采样时间步 t ~ Uniform[0, 1]
    t = torch.rand(B, device=device, dtype=dtype).unsqueeze(1).expand(-1, T)
    # 2.2 采样源噪声 x0 ~ N(0, 1)
    x0 = torch.randn_like(target_latent)
    # 2.3 目标数据 x1
    x1 = target_latent
    # 2.4 插值得到 xt (当前时刻状态)
    xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
    # 2.5 计算目标速度场 target_v (即 x1 - x0)
    target_v = x1 - x0
    
    # 3. 模型预测速度场
    pred_v = model_head(condition, xt, t, condition_mask=cfg_mask)
    
    # 4. 计算 MSE Loss (预测速度 vs 真实速度)
    loss = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=-1)
    
    # 5. Mask 并求平均 (忽略 Padding 部分)
    return (loss * mask).sum() / mask.sum().clamp(min=1)

# ---------------------------------------------------------------------
# Configuration & Main Model
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

class QwenCALM(PreTrainedModel):
    """
    功能：Audio-CALM 核心模型类。
    作用：整合 LLM (Qwen2), VAE, Projector, Flow Head，根据不同任务计算 Loss。
    """
    config_class = QwenCALMConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config: QwenCALMConfig):
        super().__init__(config)
        self.config = config
        
        # 1. 加载 LLM 基座
        print(f"Loading Qwen from {config.qwen_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, trust_remote_code=True, 
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, 
            attn_implementation="sdpa"
        )

        # 2. VAE 设置 (用于在线提取 Latent 或推理时的解码)
        self.use_precomputed_latents = config.use_precomputed_latents
        vae_latent_dim = config.latent_dim
        if not self.use_precomputed_latents:
            print(f"Loading VAE from {config.vae_path}...")
            self.vae = AcousticVAE.from_pretrained(config.vae_path)
            self.vae.requires_grad_(False)
            self.vae.eval()
            vae_latent_dim = self.vae.config.latent_channels
        
        # 3. 初始化音频投影器 (ASR 核心组件)
        qwen_dim = self.llm.config.hidden_size
        self.input_proj = AudioInputProjector(vae_latent_dim, qwen_dim, max_audio_len=config.max_audio_len)
        
        # 4. [NEW] 音频起始 Token (SOA)
        # 这是一个可学习向量，作为 Text 和 Audio 的分隔符，触发音频生成。
        with torch.no_grad():
            dummy_ids = torch.arange(1000, 2000, device=self.llm.device)
            mean_embed = self.get_input_embeddings()(dummy_ids).mean(dim=0, keepdim=True).unsqueeze(0)
            self.soa_embed = nn.Parameter(mean_embed.clone()) # [1, 1, Dim]
            
        # 确保它可训练
        self.soa_embed.requires_grad_(True)
        
        # 5. 初始化流匹配头 (TTS 核心组件)
        print(f"[QwenCALM] Flow Head (dim={config.flow_hidden_dim}, L={config.flow_num_layers})")
        self.output_head = FlowMatchingHead(
            qwen_dim, vae_latent_dim, config.flow_hidden_dim, config.flow_num_layers
        )
        
        # 6. CTC 头 (可选，辅助 ASR 训练)
        if config.ctc_loss_weight > 0:
            vocab_size = self.llm.config.vocab_size
            self.ctc_head = self.llm.lm_head 
            self.ctc_loss_fct = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def get_input_embeddings(self): 
        return self.llm.get_input_embeddings()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None: gradient_checkpointing_kwargs = {}
        gradient_checkpointing_kwargs["use_reentrant"] = False 
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.input_proj.requires_grad_(True)
        
    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(self, text_input_ids, audio_features, attention_mask=None, labels=None, task_modes=None, audio_lens=None):
        """
        功能：模型前向计算。支持 TTS 和 ASR 两种模式的混合 Batch。
        """
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device
        if task_modes is None: task_modes = ["tts"] * batch_size

        # --- 1. 获取 Latent ---
        with torch.no_grad():
            if self.use_precomputed_latents:
                # 预计算 Latent 通常是 [B, Dim, Time]，这里转置为 [B, Time, Dim]
                gt_latents = audio_features.transpose(1, 2)
            else:
                # 否则实时使用 VAE 编码
                mu, _ = self.vae.encode(audio_features)
                gt_latents = mu.transpose(1, 2)
        
        gt_latents = gt_latents.to(dtype=self.llm.dtype)
        
        # --- 2. 嵌入 (Embedding) ---
        # 2.1 音频特征通过 Projector 映射到 LLM 空间
        audio_embeds = self.input_proj(gt_latents) 
        if self.training and self.llm.is_gradient_checkpointing:
            audio_embeds.requires_grad_(True)
            
        # 2.2 文本转 Embedding
        text_embeds = self.get_input_embeddings()(text_input_ids)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, text_input_ids.shape[1]), device=device, dtype=torch.long)

        # 2.3 构建音频 Mask
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

        # 2.4 扩展 SOA Embedding
        soa_tokens = self.soa_embed.expand(batch_size, -1, -1)
        soa_mask = torch.ones((batch_size, 1), device=device, dtype=torch.long)

        # 损失初始化
        total_loss = torch.tensor(0.0, device=device)
        accum_tts_loss = torch.tensor(0.0, device=device)
        accum_asr_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        tts_count = 0
        asr_count = 0

        # --- 3. TTS 分支 (Text-to-Speech) ---
        tts_indices = [i for i, m in enumerate(task_modes) if m == "tts"]
        if len(tts_indices) > 0:
            # 激活 TTS Adapter
            if hasattr(self.llm, "set_adapter"): self.llm.set_adapter("tts")
            
            idx = torch.tensor(tts_indices, device=device)
            
            # 3.1 构建输入: [文本, SOA, 音频(错位)]
            # 这里的 Audio 是作为 History Condition 输入的 (Teacher Forcing)
            inp = torch.cat([
                text_embeds[idx], 
                soa_tokens[idx], 
                audio_embeds[idx][:, :-1, :] # 移除最后一个，因为要预测未来
            ], dim=1)
            
            # 3.2 构建 Mask
            full_mask = torch.cat([
                attention_mask[idx], 
                soa_mask[idx], 
                audio_mask[idx][:, :-1]
            ], dim=1)
            
            # 3.3 [关键] Position ID 计算
            # 使用 cumsum 确保 SOA 的位置 ID 紧接在有效的 Text 之后，忽略 Text 的 Padding。
            pos_ids = full_mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(full_mask == 0, 1)

            # 3.4 LLM 前向传播
            out = self.llm(inputs_embeds=inp, attention_mask=full_mask, position_ids=pos_ids, output_hidden_states=True)
            
            # 3.5 提取音频部分的隐状态
            # 切片起点 = Text长度 (SOA 的位置)
            slice_start_idx = text_embeds[idx].shape[1]
            audio_hidden = out.hidden_states[-1][:, slice_start_idx:, :]
            
            # 3.6 对齐长度 (处理潜在的 Shape 差异)
            target_len = gt_latents.size(1)
            if audio_hidden.shape[1] > target_len:
                audio_hidden = audio_hidden[:, :target_len, :]
            elif audio_hidden.shape[1] < target_len:
                pad_len = target_len - audio_hidden.shape[1]
                audio_hidden = F.pad(audio_hidden, (0, 0, 0, pad_len))

            # 3.7 计算流匹配损失
            cur_target = gt_latents[idx]
            cur_target_mask = audio_mask[idx][:, :cur_target.size(1)].bool()
            
            tts_loss = compute_flow_loss(self.output_head, audio_hidden, cur_target, cur_target_mask, cfg_dropout_prob=0.1)

            total_loss += tts_loss * self.config.audio_loss_weight
            accum_tts_loss += tts_loss
            tts_count += 1
            valid_samples += 1

        # --- 4. ASR 分支 (Speech-to-Text) ---
        asr_indices = [i for i, m in enumerate(task_modes) if m == "asr"]
        if len(asr_indices) > 0:
            # 激活 ASR Adapter
            if hasattr(self.llm, "set_adapter"): self.llm.set_adapter("asr")

            idx = torch.tensor(asr_indices, device=device)
            sub_audio = audio_embeds[idx]
            sub_text = text_embeds[idx]
            
            # 4.1 构建输入: [音频, 文本]
            inp = torch.cat([sub_audio, sub_text], dim=1)
            full_mask = torch.cat([audio_mask[idx], attention_mask[idx]], dim=1)
            
            # 4.2 构建 Labels
            # 音频部分设为 -100 (不计算 Loss)，只计算文本的 Next Token Prediction
            B_sub = len(asr_indices)
            prefix_labels = torch.full((B_sub, sub_audio.shape[1]), -100, dtype=torch.long, device=device)
            full_labels = torch.cat([prefix_labels, labels[idx]], dim=1)
            
            # 4.3 Position ID
            pos_ids = full_mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(full_mask == 0, 1)

            # 4.4 LLM 前向 (Cross Entropy)
            out = self.llm(inputs_embeds=inp, attention_mask=full_mask, position_ids=pos_ids, labels=full_labels, use_cache=False)
            main_loss = out.loss
            
            # 4.5 CTC 辅助损失
            ctc_loss = torch.tensor(0.0, device=device)
            if getattr(self, "ctc_head", None) is not None and self.config.ctc_loss_weight > 0:
                ctc_input = sub_audio 
                ctc_logits = self.ctc_head(ctc_input).transpose(0, 1)
                ctc_log_probs = F.log_softmax(ctc_logits.float(), dim=-1)
                
                # 准备 CTC Targets (过滤 Padding)
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

        # 平均损失
        if valid_samples > 0: 
            total_loss = total_loss / valid_samples
        
        avg_tts = accum_tts_loss / max(tts_count, 1)
        avg_asr = accum_asr_loss / max(asr_count, 1)

        return {"loss": total_loss, "loss_tts": avg_tts, "loss_asr": avg_asr}

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        功能：保存模型，额外处理 Projector 和 Head 的保存。
        """
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