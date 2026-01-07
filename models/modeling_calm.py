"""
Omni-Flow Audio-CALM: Unified Bidirectional Flow Matching Model.
Final Architecture (NeurIPS Ready):
- TTS: NAR Flow (DiT Transformer) + Global Conditioning via SOA (Concatenation)
- ASR: NAR Flow (DiT Transformer) + Positional Query Cross-Attention
- Projector: Causal Conv (Stride=1) + No Abs Pos Emb (Rely on RoPE)
- Backbone: Qwen2-1.5B (Frozen + LoRA)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cuda
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from models.modeling_vae import AcousticVAE

# ---------------------------------------------------------------------
# Audio Input Projector
# ---------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.pad_len = kernel_size - 1

    def forward(self, x):
        x = F.pad(x, (self.pad_len, 0)) 
        return self.conv(x)
    
class AudioInputProjector(nn.Module):
    def __init__(self, latent_dim, llm_dim, max_audio_len=1024, rope_base=10000, use_rope=True):
        super().__init__()
        self.use_rope = use_rope
        # [DECISION] Keep Stride=1 to preserve VAE latent details
        self.conv_block = nn.Sequential(
            CausalConv1d(latent_dim, llm_dim, kernel_size=3),
            nn.GELU(),
            CausalConv1d(llm_dim, llm_dim, kernel_size=3),
        )
        # [DECISION] REMOVED absolute positional embedding. Rely on Qwen's RoPE.
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(llm_dim, eps=1e-6),
                nn.Linear(llm_dim, llm_dim * 2),
                nn.GELU(),
                nn.Linear(llm_dim * 2, llm_dim),
            ) for _ in range(2)
        ])
        self.post_norm = nn.LayerNorm(llm_dim, eps=1e-6)

        # [FIX] RoPE 可选
        if use_rope:
            self.rope_base = rope_base
            dim = llm_dim
            inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("inv_freq", None, persistent=False)

    def _apply_rope(self, x):
        """
        x: [B, T, D] (D must be even). Applies rotary embedding on last dim.
        """
        if self.inv_freq is None:
            return x
        B, T, D = x.shape
        assert D % 2 == 0, "RoPE requires even hidden size"
        # positions [T]
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq.to(x.dtype))  # [T, D/2]
        sin, cos = freqs.sin(), freqs.cos()
        sin = sin.unsqueeze(0).unsqueeze(-1)  # [1, T, D/2, 1]
        cos = cos.unsqueeze(0).unsqueeze(-1)  # [1, T, D/2, 1]

        x = x.view(B, T, D // 2, 2)
        x1, x2 = x.unbind(-1)
        x_rope = torch.stack([x1 * cos.squeeze(-1) - x2 * sin.squeeze(-1),
                              x1 * sin.squeeze(-1) + x2 * cos.squeeze(-1)], dim=-1)
        return x_rope.view(B, T, D)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = x.transpose(1, 2)
        
        for block in self.blocks:
            x = x + block(x)
        x = self.post_norm(x)
        # [FIX] 只在 use_rope=True 时应用
        if self.use_rope:
            x = self._apply_rope(x)
        return x
    
# ---------------------------------------------------------------------
# Flow Matching Head (Legacy: ResNet1d - Kept for backward compatibility)
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

    class ResBlock1d(nn.Module):
        def __init__(self, dim, dilation, kernel_size=3):
            super().__init__()
            self.conv = nn.Sequential(
                nn.SiLU(),
                nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=dilation, dilation=dilation),
                nn.SiLU(),
                nn.Conv1d(dim, dim, kernel_size=1)
            )
        def forward(self, x):
            return x + self.conv(x)

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.time_dim = 256
        
        self.time_mlp = nn.Sequential(
            self.SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim), nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        
        self.in_proj = nn.Conv1d(input_dim + output_dim + self.time_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.layers = nn.ModuleList([
            self.ResBlock1d(hidden_dim, dilation=2**i) 
            for i in range(num_layers)
        ])
        
        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim), 
            nn.SiLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)
        )
        
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, condition, noisy_x, t, condition_mask=None):
        if t.dim() == 1: 
            t = t.unsqueeze(1).expand(-1, condition.size(1))
        t_emb = self.time_mlp(t.reshape(-1)).view(condition.shape[0], condition.shape[1], -1)
        
        x = torch.cat([condition, noisy_x, t_emb], dim=-1)
        x = x.transpose(1, 2)
        
        if condition_mask is not None:
            mask_expanded = condition_mask.view(-1, 1, 1).to(dtype=x.dtype)
            x = x * mask_expanded

        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out_proj(x)
        x = x.transpose(1, 2)
        return x
    
# =====================================================================
# [CORE ARCHITECTURE] Transformer-based Flow Head (DiT)
# Used for BOTH ASR and TTS to ensure unified high-capacity modeling.
# =====================================================================
class TransformerFlowHead(nn.Module):
    """
    DiT-style Transformer for Flow Matching.
    Uses Adaptive Layer Norm (AdaLN) for Time Conditioning.
    Structure: Input Concat -> Linear -> DiT Blocks (Self-Attn only) -> Output
    """
    class AdaLN(nn.Module):
        def __init__(self, dim, time_dim):
            super().__init__()
            self.emb = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, dim * 2)
            )
            self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            
        def forward(self, x, t_emb):
            scale, shift = self.emb(t_emb).chunk(2, dim=1)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            return self.norm(x) * (1 + scale) + shift

    class DiTBlock(nn.Module):
        def __init__(self, dim, num_heads, time_dim, mlp_ratio=4.0):
            super().__init__()
            self.adaLN1 = TransformerFlowHead.AdaLN(dim, time_dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
            
            self.adaLN_ctx = TransformerFlowHead.AdaLN(dim, time_dim)
            self.ctx_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
            self.ctx_gate = nn.Parameter(torch.zeros(1))

            self.adaLN2 = TransformerFlowHead.AdaLN(dim, time_dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim)
            )
            
        def forward(self, x, t_emb, context=None, context_mask=None, x_mask=None):
            x = x.contiguous()
            
            x_norm = self.adaLN1(x, t_emb)
            x_norm = x_norm.contiguous()
            
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=x_mask)

            x = x + attn_out

            if context is not None:
                context = context.contiguous()
                x_ctx_norm = self.adaLN_ctx(x, t_emb)
                x_ctx_norm = x_ctx_norm.contiguous()
                key_padding_mask = context_mask if context_mask is not None else None
                ctx_out, _ = self.ctx_attn(
                    query=x_ctx_norm,
                    key=context,
                    value=context,
                    key_padding_mask=key_padding_mask
                )
                x = x + torch.sigmoid(self.ctx_gate) * ctx_out

            x_norm = self.adaLN2(x, t_emb)
            x = x + self.mlp(x_norm)
            return x

    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, num_heads=16, context_dim=None):
        super().__init__()
        self.time_dim = 256
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            FlowMatchingHead.SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim), nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        
        self.in_proj = nn.Linear(input_dim + output_dim, hidden_dim)
        
        # [NEW] 序列位置编码
        self.max_seq_len = 2048
        self.register_buffer(
            "pos_emb",
            self._build_sinusoidal_pos_emb(self.max_seq_len, hidden_dim),
            persistent=False
        )
        
        self.blocks = nn.ModuleList([
            self.DiTBlock(hidden_dim, num_heads, self.time_dim)
            for _ in range(num_layers)
        ])
        
        self.final_adaLN = self.AdaLN(hidden_dim, self.time_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # [NEW] 构建正弦位置编码
    @staticmethod
    def _build_sinusoidal_pos_emb(max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, dim]

    def forward(self, condition, noisy_x, t, condition_mask=None, context=None, context_mask=None, x_mask=None):
        B, T, _ = noisy_x.shape
        if t.dim() > 1: 
            t = t[:, 0]
        t_emb = self.time_mlp(t)

        condition = condition.contiguous()
        noisy_x = noisy_x.contiguous()
        
        x = torch.cat([condition, noisy_x], dim=-1)
        x = self.in_proj(x)
        
        # [NEW] 添加位置编码
        x = x + self.pos_emb[:, :T, :].to(x.dtype)
        
        x = x.contiguous()

        proj_context = None
        if context is not None and self.context_proj is not None:
            proj_context = self.context_proj(context.contiguous())
            proj_context = proj_context.contiguous()
            
        for block in self.blocks:
            x = block(x, t_emb, context=proj_context, context_mask=context_mask, x_mask=x_mask)
            
        x = self.final_adaLN(x, t_emb)
        x = self.out_proj(x)
        return x

def build_alignment_from_durations(dur_int, valid_mask, T_aud, device, dtype):
    """
    dur_int: [B, T_txt], valid_mask: [B, T_txt], 返回 [B, T_txt, T_aud]
    """
    B, T_txt = dur_int.shape

    dur_int_masked = dur_int * valid_mask.long()
    total_dur = dur_int_masked.sum(dim=1, keepdim=True)

    # 超预算按比例缩放
    scale = torch.where(
        total_dur > T_aud,
        T_aud / total_dur.float().clamp(min=1),
        torch.ones_like(total_dur, dtype=torch.float)
    )
    dur_int_scaled = (dur_int_masked.float() * scale).long()

    # 缩放后 valid 位置至少 1
    dur_int_scaled = torch.where(
        valid_mask & (dur_int_scaled < 1),
        torch.ones_like(dur_int_scaled),
        dur_int_scaled
    )

    # 计算余数（可能为负）
    remain = T_aud - dur_int_scaled.sum(dim=1)

    # 如余数为负，尝试从最长的 valid token 递减到不低于 1
    if (remain < 0).any():
        for b in range(B):
            if remain[b] < 0:
                deficit = -remain[b].item()
                while deficit > 0:
                    valid_durs = dur_int_scaled[b] * valid_mask[b].long()
                    max_idx = valid_durs.argmax()
                    if dur_int_scaled[b, max_idx] > 1:
                        dec = min(deficit, dur_int_scaled[b, max_idx].item() - 1)
                        dur_int_scaled[b, max_idx] -= dec
                        deficit -= dec
                    else:
                        break
                remain[b] = 0

    remain = remain.clamp(min=0)
    dur_int_scaled = distribute_remainder_vectorized(dur_int_scaled, remain, valid_mask)

    cum_dur = torch.cumsum(dur_int_scaled, dim=1)
    start_pos = cum_dur - dur_int_scaled
    end_pos = torch.clamp(cum_dur, max=T_aud)

    frame_idx = torch.arange(T_aud, device=device).view(1, 1, T_aud)
    start_exp = start_pos.unsqueeze(-1)
    end_exp = end_pos.unsqueeze(-1)
    valid_exp = valid_mask.unsqueeze(-1)

    align = ((frame_idx >= start_exp) & (frame_idx < end_exp) & valid_exp).to(dtype)
    return align

def distribute_remainder_vectorized(dur_int, remain, valid_mask):
    """
    向量化分配余数到 valid tokens
    dur_int: [B, T_txt]
    remain: [B] - 每个样本需要分配的余数
    valid_mask: [B, T_txt]
    返回: 更新后的 dur_int
    """
    B, T_txt = dur_int.shape
    device = dur_int.device
    
    # 为每个 batch 创建 valid token 的累积索引
    valid_cumsum = torch.cumsum(valid_mask.long(), dim=1)  # [B, T_txt]
    
    # 对于每个 token，判断是否需要 +1
    # 条件：是 valid token 且 累积索引 <= remain
    should_add = valid_mask & (valid_cumsum <= remain.unsqueeze(1))
    
    dur_int = dur_int + should_add.long()
    return dur_int

# ---------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------
def compute_flow_loss(model_head, condition, target_latent, mask, cfg_dropout_prob=0.0, **kwargs):
    """
    kwargs 可传递 context / context_mask / x_mask 供带 cross-attn 的头使用
    """
    B, T, D = target_latent.shape
    device = target_latent.device
    dtype = target_latent.dtype
    mask = mask.bool()

    context = kwargs.get('context', None)
    context_mask = kwargs.get('context_mask', None)
    
    # [FIX] 构建 x_mask 用于 self-attention（True=PAD）
    x_mask = ~mask  # mask: True=valid, x_mask: True=PAD
    
    if model_head.training and cfg_dropout_prob > 0:
        drop_mask = torch.rand(B, device=device) < cfg_dropout_prob
        condition = condition.clone()
        condition[drop_mask] = 0
        if context is not None:
            context = context.clone()
            context[drop_mask] = 0

    t = torch.rand(B, device=device, dtype=dtype).unsqueeze(1).expand(-1, T)
    x0 = torch.randn_like(target_latent)
    x1 = target_latent
    xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
    target_v = x1 - x0
    
    # [FIX] 传入 x_mask
    pred_v = model_head(condition, xt, t, context=context, context_mask=context_mask, x_mask=x_mask)
    loss = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1)

# ------------------ MAS: Monotonic Alignment Search ------------------
def monotonic_alignment_search(log_p):
    """
    MAS: 动态规划找最优单调对齐路径
    
    log_p: [B, T_text, T_audio] - 对数概率/相似度
    返回: [B, T_text, T_audio] - 二值对齐矩阵
    
    约束:
    1. 单调性: 如果帧 t 对应 token n，则帧 t+1 只能对应 token n 或 n+1
    2. 覆盖性: 每个 audio 帧恰好对应一个 text token
    3. 完整性: 第一帧对应第一个 token，最后一帧对应最后一个 token
    """
    B, N, T = log_p.shape
    device = log_p.device
    dtype = log_p.dtype
    
    # DP 表: dp[n, t] = 从 (0,0) 到 (n, t) 的最大 log-prob
    # 转移: dp[n, t] = log_p[n, t] + max(dp[n, t-1], dp[n-1, t-1])
    
    # 使用 CPU 进行 DP（避免 CUDA 同步开销）
    log_p_cpu = log_p.detach().cpu().numpy()
    
    import numpy as np
    
    alignments = []
    for b in range(B):
        lp = log_p_cpu[b]  # [N, T]
        
        # 初始化 DP
        dp = np.full((N, T), -np.inf, dtype=np.float32)
        
        # 第一个 token 可以从任意起始帧开始（但通常从 0 开始）
        dp[0, 0] = lp[0, 0]
        for t in range(1, T):
            dp[0, t] = dp[0, t-1] + lp[0, t]
        
        # 填充 DP 表
        for n in range(1, N):
            for t in range(n, T):  # 至少要有 n 帧才能到达 token n
                # 两种转移：停留在当前 token，或从上一个 token 转移
                stay = dp[n, t-1] if t > 0 else -np.inf
                move = dp[n-1, t-1] if t > 0 else -np.inf
                dp[n, t] = max(stay, move) + lp[n, t]
        
        # 回溯找最优路径
        align = np.zeros((N, T), dtype=np.float32)
        n = N - 1
        t = T - 1
        
        while n >= 0 and t >= 0:
            align[n, t] = 1.0
            if n == 0:
                # 必须停留在第一个 token
                t -= 1
            elif t == 0:
                # 无法回溯，结束
                break
            else:
                # 选择来自哪个方向
                stay = dp[n, t-1] if t > 0 else -np.inf
                move = dp[n-1, t-1] if t > 0 else -np.inf
                if move > stay:
                    n -= 1
                t -= 1
        
        alignments.append(align)
    
    return torch.from_numpy(np.stack(alignments)).to(device=device, dtype=dtype)

# ---------------------------------------------------------------------
# Configuration & Main Model
# ---------------------------------------------------------------------
class QwenCALMConfig(PretrainedConfig):
    model_type = "qwen_calm"
    def __init__(self, qwen_path=None, vae_path=None, use_precomputed_latents=True, 
                 latent_dim=128, tts_loss_weight=1.0, asr_loss_weight=1.0, 
                 downsample_rate=1, max_audio_len=1024, 
                 tts_flow_hidden_dim=1024, tts_flow_num_layers=6,
                 asr_flow_hidden_dim=1024, asr_flow_num_layers=6, 
                 max_text_len=256,                 # [NEW]
                 len_pred_loss_weight=0.1,         # [NEW]
                 dur_pred_loss_weight=0.1,
                 mel_mean=0.0, mel_std=1.0, 
                 latent_mean=0.0, latent_std=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.qwen_path = qwen_path
        self.vae_path = vae_path
        self.use_precomputed_latents = use_precomputed_latents
        self.latent_dim = latent_dim
        self.tts_loss_weight = tts_loss_weight
        self.asr_loss_weight = asr_loss_weight
        self.downsample_rate = downsample_rate
        self.max_audio_len = max_audio_len
        self.tts_flow_hidden_dim = tts_flow_hidden_dim
        self.tts_flow_num_layers = tts_flow_num_layers
        self.asr_flow_hidden_dim = asr_flow_hidden_dim
        self.asr_flow_num_layers = asr_flow_num_layers
        self.max_text_len = max_text_len          # [NEW]
        self.len_pred_loss_weight = len_pred_loss_weight  # [NEW]
        self.dur_pred_loss_weight = dur_pred_loss_weight
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.latent_mean = latent_mean
        self.latent_std = latent_std

class QwenCALM(PreTrainedModel):
    config_class = QwenCALMConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config: QwenCALMConfig):
        super().__init__(config)
        self.config = config

        print(f"Loading Qwen from {config.qwen_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, trust_remote_code=True, 
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, 
            attn_implementation="sdpa"
        )

        self.use_precomputed_latents = config.use_precomputed_latents
        vae_latent_dim = config.latent_dim
        if not self.use_precomputed_latents:
            print(f"Loading VAE from {config.vae_path}...")
            self.vae = AcousticVAE.from_pretrained(config.vae_path)
            self.vae.requires_grad_(False)
            self.vae.eval()
            vae_latent_dim = self.vae.config.latent_channels
        
        # [FIX] 从 LLM 获取 hidden_size，而不是 config.llm_dim
        qwen_dim = self.llm.config.hidden_size
        
        # [FIX] 关闭 projector 的 RoPE，让 LLM 统一处理位置编码
        self.input_proj = AudioInputProjector(
            vae_latent_dim, qwen_dim, 
            max_audio_len=config.max_audio_len, 
            use_rope=False  # ASR 分支会进入 LLM，避免双重 RoPE
        )

        with torch.no_grad():
            dummy_ids = torch.arange(1000, 2000, device=self.llm.device)
            mean_embed = self.get_input_embeddings()(dummy_ids).mean(dim=0, keepdim=True).unsqueeze(0)
            self.soa_embed = nn.Parameter(mean_embed.clone()) 
        self.soa_embed.requires_grad_(True)
        
        print(f"[Omni-Flow] Initializing ASR Cross-Attention")
        self.asr_cross_attn = nn.MultiheadAttention(
            embed_dim=qwen_dim, 
            num_heads=16, 
            batch_first=True, 
            dropout=0.1
        )
        self.asr_query_embed = nn.Embedding(config.max_text_len, qwen_dim)

        print(f"[Omni-Flow] TTS Flow Head (DiT Transformer)")
        self.tts_flow_head = TransformerFlowHead(
            input_dim=qwen_dim,        
            output_dim=vae_latent_dim, 
            hidden_dim=config.tts_flow_hidden_dim, 
            num_layers=config.tts_flow_num_layers,
            num_heads=16,
            context_dim=qwen_dim
        )
        
        # [NEW] TTS 长度预测器
        self.tts_len_predictor = nn.Sequential(
            nn.Linear(qwen_dim, qwen_dim // 2),
            nn.GELU(),
            nn.Linear(qwen_dim // 2, 1)
        )

        print(f"[Omni-Flow] ASR Flow Head (DiT Transformer)")
        self.asr_flow_head = TransformerFlowHead(
            input_dim=qwen_dim, 
            output_dim=qwen_dim, 
            hidden_dim=config.asr_flow_hidden_dim,
            num_layers=config.asr_flow_num_layers,
            num_heads=16,
            context_dim=None
        )

        # [NEW] Duration Predictor
        self.tts_dur_predictor = nn.Sequential(
            nn.Linear(qwen_dim, qwen_dim // 2),
            nn.GELU(),
            nn.Linear(qwen_dim // 2, 1)
        )

        target_dtype = self.llm.dtype
        self.input_proj.to(target_dtype)
        self.asr_cross_attn.to(target_dtype)
        self.asr_query_embed.to(target_dtype)
        self.tts_flow_head.to(target_dtype)
        self.asr_flow_head.to(target_dtype)
        self.tts_len_predictor.to(target_dtype)
        self.tts_dur_predictor.to(target_dtype)
        self.soa_embed.data = self.soa_embed.data.to(target_dtype)

    def get_input_embeddings(self): 
        return self.llm.get_input_embeddings()
    
    def search_nearest_tokens(self, continuous_embeddings):
        norm_pred = F.normalize(continuous_embeddings, p=2, dim=-1)
        all_embeddings = self.get_input_embeddings().weight 
        norm_vocab = F.normalize(all_embeddings, p=2, dim=-1)
        dists = torch.cdist(norm_pred.float(), norm_vocab.float())
        token_ids = torch.argmin(dists, dim=-1)
        return token_ids

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None: gradient_checkpointing_kwargs = {}
        gradient_checkpointing_kwargs["use_reentrant"] = False 
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.input_proj.requires_grad_(True)
        
    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(self, text_input_ids, audio_features, attention_mask=None, labels=None, task_modes=None, audio_lens=None):
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device
        if task_modes is None: task_modes = ["tts"] * batch_size

        with torch.no_grad():
            if self.use_precomputed_latents:
                gt_latents = audio_features.transpose(1, 2)
            else:
                mu, _ = self.vae.encode(audio_features)
                gt_latents = mu.transpose(1, 2)
        gt_latents = gt_latents.to(dtype=self.llm.dtype)

        # === 对 VAE latent 做标准化，供 Flow 使用 ===
        # 支持标量或向量形状的 mean/std（如 [latent_dim]）
        latent_mean = getattr(self.config, "latent_mean", 0.0)
        latent_std = getattr(self.config, "latent_std", 1.0)
        latent_mean = torch.as_tensor(latent_mean, device=gt_latents.device, dtype=gt_latents.dtype)
        latent_std = torch.as_tensor(latent_std, device=gt_latents.device, dtype=gt_latents.dtype)
        if latent_mean.ndim == 1:    # [D] -> [1,1,D]
            latent_mean = latent_mean.view(1, 1, -1)
        if latent_std.ndim == 1:
            latent_std = latent_std.view(1, 1, -1)
        gt_latents = (gt_latents - latent_mean) / latent_std
        
        audio_embeds = self.input_proj(gt_latents) 
        if self.training and self.llm.is_gradient_checkpointing:
            audio_embeds.requires_grad_(True)
        text_embeds = self.get_input_embeddings()(text_input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, text_input_ids.shape[1]), device=device, dtype=torch.long)

        B_aud, T_aud, _ = gt_latents.shape
        if audio_lens is not None:
            ds_rate = getattr(self.config, 'downsample_rate', 1) 
            latent_lens = torch.div(audio_lens + ds_rate - 1, ds_rate, rounding_mode='floor')
            latent_lens = latent_lens.clamp(max=T_aud)
            audio_mask = (torch.arange(T_aud, device=device)[None, :] < latent_lens[:, None]).long()
        else:
            audio_mask = torch.ones((B_aud, T_aud), device=device, dtype=torch.long)

        soa_tokens = self.soa_embed.expand(batch_size, -1, -1)
        soa_mask = torch.ones((batch_size, 1), device=device, dtype=torch.long)

        total_loss = torch.tensor(0.0, device=device)
        accum_tts_loss = torch.tensor(0.0, device=device)
        accum_asr_loss = torch.tensor(0.0, device=device)
        accum_len_loss = torch.tensor(0.0, device=device)
        accum_dur_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        tts_count = 0
        asr_count = 0
        len_count = 0
        dur_count = 0

        # --- TTS Branch (NAR Mode) ---
        tts_indices = [i for i, m in enumerate(task_modes) if m == "tts"]
        if len(tts_indices) > 0:
            idx = torch.tensor(tts_indices, device=device)

            # LLM 编码文本 + SOA
            inp = torch.cat([text_embeds[idx], soa_tokens[idx]], dim=1)
            full_mask = torch.cat([attention_mask[idx], soa_mask[idx]], dim=1)
            pos_ids = full_mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(full_mask == 0, 1)

            out = self.llm(
                inputs_embeds=inp,
                attention_mask=full_mask,
                position_ids=pos_ids,
                output_hidden_states=True
            )
            hidden = out.hidden_states[-1]
            
            condition_vec = hidden[:, -1:, :]  # [B_tts, 1, D] - Global SOA
            text_context = hidden[:, :-1, :]   # [B_tts, T_txt, D] - Local text
            text_ctx_mask = (full_mask[:, :-1] == 0)  # True=PAD

            cur_target = gt_latents[idx]
            cur_audio_mask = audio_mask[idx]
            cur_target_mask = cur_audio_mask.bool()

            # === Length Prediction ===
            valid_mask = ~text_ctx_mask
            valid_len = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(text_context.dtype)
            masked_text = text_context * valid_mask.unsqueeze(-1).to(text_context.dtype)
            text_mean = masked_text.sum(dim=1) / valid_len
            
            pred_dtype = self.tts_len_predictor[0].weight.dtype
            text_mean = text_mean.to(pred_dtype)
            len_pred = self.tts_len_predictor(text_mean).squeeze(-1)

            gt_len = cur_audio_mask.sum(dim=1).float()
            
            # [FIX] 训练时也对 len_pred 施加与推理一致的范围约束
            text_len = valid_mask.sum(dim=1).float()
            min_frames = torch.clamp(text_len * 2, min=10)
            max_frames = torch.clamp(text_len * 12, max=float(self.config.max_audio_len))
            len_pred_clamped = torch.clamp(len_pred, min=min_frames, max=max_frames)
            
            len_loss = F.smooth_l1_loss(
                torch.log1p(len_pred_clamped), 
                torch.log1p(gt_len)
            )

            # === Duration Prediction with MAS-based GT ===
            B_tts, T_txt, D = text_context.shape
            T_aud = cur_target.shape[1]
            
            with torch.no_grad():
                audio_for_align = self.input_proj(cur_target)
                
                text_norm = F.normalize(text_context, p=2, dim=-1)
                audio_norm = F.normalize(audio_for_align, p=2, dim=-1)
                
                sim = torch.bmm(text_norm, audio_norm.transpose(1, 2))
                
                sim = sim.masked_fill(text_ctx_mask.unsqueeze(-1), -1e9)
                sim = sim.masked_fill(~cur_target_mask.unsqueeze(1), -1e9)
                
                log_p = F.log_softmax(sim, dim=1)
                
                align_gt = monotonic_alignment_search(log_p)
                
                gt_dur = align_gt.sum(dim=-1)
            
            # Duration Predictor
            dur_raw = self.tts_dur_predictor(text_context.to(pred_dtype)).squeeze(-1)
            dur_pred = F.softplus(dur_raw) + 1e-4
            dur_pred = dur_pred.masked_fill(text_ctx_mask, 0)
            
            # 归一化到目标长度
            dur_sum = dur_pred.sum(dim=1, keepdim=True).clamp(min=1e-4)
            dur_pred_scaled = dur_pred * (T_aud / dur_sum)
            
            # Duration Loss（对数域，避免大值主导）
            dur_loss = F.l1_loss(
                torch.log1p(dur_pred_scaled * valid_mask.float()),
                torch.log1p(gt_dur * valid_mask.float())
            )

            # === 训练时用 GT Alignment，推理时用预测的 Duration ===
            if self.training:
                align = align_gt.to(text_context.dtype)
            else:
                dur_int = torch.floor(dur_pred_scaled).long()
                remain = (T_aud - dur_int.sum(dim=1)).clamp(min=0)
                dur_int = distribute_remainder_vectorized(dur_int, remain, valid_mask)
                align = build_alignment_from_durations(dur_int, valid_mask, T_aud, device, text_context.dtype)
            
            aligned_text = torch.bmm(align.transpose(1, 2), text_context)
            condition = aligned_text + condition_vec.expand(-1, T_aud, -1)

            # [FIX] 将 PAD 位置的 condition 和 target 置零，避免 self-attn 看到无效信息
            condition = condition * cur_target_mask.unsqueeze(-1).to(condition.dtype)
            cur_target_masked = cur_target * cur_target_mask.unsqueeze(-1).to(cur_target.dtype)

            # Flow Loss
            tts_loss = compute_flow_loss(
                self.tts_flow_head,
                condition=condition,
                target_latent=cur_target_masked,
                mask=cur_target_mask,
                context=text_context,
                context_mask=text_ctx_mask,
                cfg_dropout_prob=0.1,
            )
            
            # Accumulate losses
            total_loss += tts_loss * self.config.tts_loss_weight
            total_loss += len_loss * self.config.len_pred_loss_weight
            total_loss += dur_loss * self.config.dur_pred_loss_weight
            
            accum_tts_loss += tts_loss
            accum_len_loss += len_loss
            accum_dur_loss += dur_loss
            tts_count += 1
            len_count += 1
            dur_count += 1
            valid_samples += 1

        # --- ASR Branch (Cross-Attention Mode + Positional Query Fix) ---
        asr_indices = [i for i, m in enumerate(task_modes) if m == "asr"]
        if len(asr_indices) > 0:
            idx = torch.tensor(asr_indices, device=device)
            
            sub_audio = audio_embeds[idx] 
            sub_prompt = text_embeds[idx] 
            
            inp = torch.cat([sub_audio, soa_tokens[idx], sub_prompt], dim=1)
            full_mask = torch.cat([audio_mask[idx], soa_mask[idx], attention_mask[idx]], dim=1)
            
            pos_ids = full_mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(full_mask == 0, 1)

            out = self.llm(inputs_embeds=inp, attention_mask=full_mask, position_ids=pos_ids, output_hidden_states=True)
            
            # 1. Get Audio Context
            T_aud_curr = sub_audio.shape[1]
            audio_context = out.hidden_states[-1][:, :T_aud_curr, :] 
            
            # 2. Get Target Text (for Loss)
            sub_labels = labels[idx]
            valid_target_mask = (sub_labels != -100).long()
            safe_label_ids = sub_labels.clone()
            safe_label_ids[safe_label_ids == -100] = 0
            
            raw_target_embeds = self.get_input_embeddings()(safe_label_ids)
            # [FIX] 不要 normalize target embedding
            target_text_embs = raw_target_embeds
            
            # 3. Construct Positional Query
            B_sub, T_text, _ = target_text_embs.shape
            pos_query_ids = torch.arange(T_text, device=device).unsqueeze(0).expand(B_sub, -1)
            pos_query_ids = pos_query_ids.clamp(max=self.asr_query_embed.num_embeddings - 1)
            query_embeds = self.asr_query_embed(pos_query_ids)

            # 4. Cross Attention
            key_padding_mask = (audio_mask[idx][:, :T_aud_curr] == 0)
            
            attn_output, _ = self.asr_cross_attn(
                query=query_embeds,
                key=audio_context,
                value=audio_context,
                key_padding_mask=key_padding_mask
            )
            condition = attn_output 
            
            # [FIX] 将 PAD 位置的 condition 和 target 置零，与 TTS 分支保持一致
            valid_target_mask_bool = valid_target_mask.bool()
            condition = condition * valid_target_mask_bool.unsqueeze(-1).to(condition.dtype)
            target_text_embs_masked = target_text_embs * valid_target_mask_bool.unsqueeze(-1).to(target_text_embs.dtype)
            
            # 5. Flow Match
            asr_flow_loss = compute_flow_loss(
                self.asr_flow_head, 
                condition=condition, 
                target_latent=target_text_embs_masked,
                mask=valid_target_mask,
                cfg_dropout_prob=0.1,
                x_mask=~valid_target_mask_bool
            )

            total_loss += asr_flow_loss * self.config.asr_loss_weight
            accum_asr_loss += asr_flow_loss
            asr_count += 1
            valid_samples += 1

        if valid_samples > 0: 
            total_loss = total_loss / valid_samples
        
        avg_tts = accum_tts_loss / max(tts_count, 1)
        avg_asr = accum_asr_loss / max(asr_count, 1)
        avg_len = accum_len_loss / max(len_count, 1)
        avg_dur = accum_dur_loss / max(dur_count, 1)
        return {
            "loss": total_loss,
            "loss_tts": avg_tts,
            "loss_asr": avg_asr,
            "loss_len": avg_len,
            "loss_dur": avg_dur,
        }
    
    def save_pretrained(self, save_directory: str, **kwargs):
        self.config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory)
        state_dict = kwargs.get("state_dict", None)
        
        def save_part(prefix, filename):
            if state_dict is None: 
                if hasattr(self, prefix): 
                    obj = getattr(self, prefix)
                    if isinstance(obj, nn.Module):
                        torch.save(obj.state_dict(), os.path.join(save_directory, filename))
                    elif isinstance(obj, nn.Parameter):
                        torch.save({"weight": obj.data}, os.path.join(save_directory, filename))
            else:
                sd = {k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
                if sd:
                    torch.save(sd, os.path.join(save_directory, filename))

        save_part("input_proj", "input_proj.bin")
        save_part("tts_flow_head", "tts_flow_head.bin")
        save_part("asr_flow_head", "asr_flow_head.bin")
        save_part("soa_embed", "soa_embed.bin")
        save_part("tts_len_predictor", "tts_len_predictor.bin")
        save_part("tts_dur_predictor", "tts_dur_predictor.bin")
        save_part("asr_query_embed", "asr_query_embed.bin")
        save_part("asr_cross_attn", "asr_cross_attn.bin")