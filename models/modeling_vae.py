# models/modeling_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import List
import math

# =============================================================================
# 1. SSIM Loss (结构相似性损失)
# =============================================================================
class SSIMLoss(nn.Module):
    """
    功能：计算两张频谱图之间的结构相似性损失 (Structural Similarity Loss)。
    
    【原理】：
    普通的 MSE (均方误差) 只关注像素点数值的差异，容易导致生成的频谱“模糊”。
    SSIM 关注局部的亮度、对比度和结构信息，能让生成的 Mel 频谱纹理更清晰，
    这对于后续 Vocoder 还原出高质量语音至关重要。
    
    【文件间关系】：
    - 被 `AcousticVAE.forward` 调用，作为重建损失的一部分。
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1 # Mel 频谱被视为单通道图像
        self.window = self.create_window(window_size, self.channel)

    # 1. 生成高斯核，用于计算局部均值和方差
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    # 2. 创建 2D 卷积窗口
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        """
        输入: [Batch, 80, Time] 或 [Batch, 1, 80, Time]
        输出: Scalar Loss (1 - SSIM)
        """
        # 维度调整：确保输入是 [B, C, H, W] 格式 (image-like)
        if img1.dim() == 3:
            img1 = img1.unsqueeze(1)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(1)

        (_, channel, _, _) = img1.size()

        # 动态创建窗口以匹配设备和数据类型
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        # 3. 计算局部统计量 (均值 Mu, 方差 Sigma, 协方差 Sigma12)
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        # 4. SSIM 公式
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        # 5. 返回 Loss (1 - SSIM)
        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)

class AudioVAEConfig(PretrainedConfig):
    """
    功能：VAE 的配置类，继承自 HF PretrainedConfig，方便保存和加载。
    
    【文件间关系】：
    - 对应 `vae_config.yaml` 中的 `model` 部分参数。
    - 被 `train_vae.py` 实例化并传入 `AcousticVAE`。
    """
    model_type = "audio_vae"
    def __init__(
        self, 
        in_channels=80,       # 输入 Mel 维度
        hidden_channels=512,  # 隐藏层通道数
        latent_channels=128,   # 压缩后的 Latent 维度 (CALM 模型的输入维度)
        strides: List[int] = [2, 2], # 下采样倍率，[2,2] 意味着时间轴压缩 4 倍
        kl_weight=0.00005,     # KL 散度权重的上限
        kl_clamp=2.0,         # KL Loss 的截断阈值，防止梯度爆炸
        latent_dropout=0.05,  # Latent 层的 Dropout
        norm_num_groups=32,   # GroupNorm 的组数
        use_l1_loss=True,     # 是否使用 L1 Loss (比 MSE 更鲁棒)
        ssim_weight=1.0,      # SSIM Loss 的权重
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.strides = strides
        self.kl_weight = kl_weight
        self.kl_clamp = kl_clamp
        self.latent_dropout = latent_dropout
        self.norm_num_groups = norm_num_groups
        self.use_l1_loss = use_l1_loss
        self.ssim_weight = ssim_weight

class ResBlock(nn.Module):
    """
    功能：标准的残差卷积块 (Residual Block)。
    结构: Input -> [Norm -> Act -> Conv] x 2 -> Add Input
    """
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels, eps=1e-6),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, 1, 1),
            
            nn.GroupNorm(num_groups, channels, eps=1e-6),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, 1, 1),
        )
    def forward(self, x):
        return x + self.conv(x)

class AcousticVAE(PreTrainedModel):
    """
    功能：声学 VAE 主模型。
    包含 Encoder (压缩), Reparameterization (采样), Decoder (还原)。
    
    【文件间关系】：
    - 被 `train_vae.py` 训练。
    - 被 `preprocess/core.py` 加载，用于离线提取 Latent。
    - 被 `modeling_calm.py` 中的 `QwenCALM` 引用 (在 Stage 2 训练/推理时，如果开启了 VAE 模式)。
    """
    config_class = AudioVAEConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.strides = config.strides
        self.total_stride = 1
        for s in self.strides: self.total_stride *= s
        print(f"[AcousticVAE] Total Downsampling Stride: {self.total_stride}")
        
        # === 1. 构建 Encoder (下采样) ===
        encoder_layers = []
        current_channels = config.in_channels
        
        # 初始卷积：提升通道数
        encoder_layers.append(
            nn.Conv1d(config.in_channels, config.hidden_channels, 3, 1, 1)
        )
        current_channels = config.hidden_channels

        # 下采样层循环
        for stride in config.strides:
            kernel_size = stride * 2
            padding = stride // 2
            encoder_layers.append(
                nn.Sequential(
                    # 步长卷积进行下采样
                    nn.Conv1d(
                        current_channels, 
                        config.hidden_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding
                    ),
                    # 残差块提取特征
                    ResBlock(config.hidden_channels, config.norm_num_groups)
                )
            )
            
        # 投影到 Latent 空间 (输出 2 * latent_channels，因为要预测均值和方差)
        self.encoder = nn.Sequential(
            *encoder_layers,
            nn.GroupNorm(config.norm_num_groups, config.hidden_channels, eps=1e-6), 
            nn.GELU(),
            nn.Conv1d(config.hidden_channels, config.latent_channels * 2, 3, 1, 1)
        )
        
        # === 2. 构建 Decoder (上采样) ===
        decoder_layers = []
        # 初始投影：从 Latent 恢复通道数
        decoder_layers.append(
            nn.Sequential(
                nn.Conv1d(config.latent_channels, config.hidden_channels, 3, 1, 1),
                ResBlock(config.hidden_channels, config.norm_num_groups)
            )
        )
        
        # 上采样层循环 (倒序)
        for stride in reversed(config.strides):
            kernel_size = stride * 2
            padding = stride // 2
            decoder_layers.append(
                nn.Sequential(
                    # 转置卷积进行上采样
                    nn.ConvTranspose1d(
                        config.hidden_channels, 
                        config.hidden_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding
                    ),
                    ResBlock(config.hidden_channels, config.norm_num_groups)
                )
            )
            
        # 输出层：还原回 Mel 维度 (in_channels)
        self.decoder_net = nn.Sequential(*decoder_layers)
        self.final_proj = nn.Conv1d(config.hidden_channels, config.in_channels, 3, 1, 1)
        
        # 损失函数组件
        self.use_l1_loss = config.use_l1_loss
        self.ssim_loss = SSIMLoss()
        self.ssim_weight = getattr(config, 'ssim_weight', 1.0)

    def encode(self, x):
        """
        功能：将 Mel 频谱编码为高斯分布参数。
        输入: [B, 80, T]
        输出: mu [B, 64, T'], logvar [B, 64, T']
        """
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1) # 切分通道
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        功能：重参数化技巧 (Reparameterization Trick)。
        使得采样过程 z ~ N(mu, std) 变得可微，从而能进行反向传播。
        z = mu + epsilon * std
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # Latent 上的 Dropout (增强鲁棒性)
            if self.config.latent_dropout > 0:
                z = F.dropout(z, p=self.config.latent_dropout, training=True)
            return z
        return mu # 推理时直接使用均值

    def decode(self, z):
        """
        功能：从 Latent 还原 Mel 频谱。
        """
        h = self.decoder_net(z)
        return self.final_proj(h)
    
    @staticmethod
    def _stft_mag(x, n_fft=1024, hop_length=256, win_length=None):
        B, C, T = x.shape
        win_length = win_length or n_fft
        x_2d = x.reshape(-1, T).float()  # 用 float32 防止数值过小
        window = torch.hann_window(win_length, device=x.device, dtype=x_2d.dtype)
        X = torch.stft(
            x_2d,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            normalized=False,
            center=False,
            pad_mode="reflect"
        )
        mag = torch.abs(X)
        return mag.view(B, C, *mag.shape[-2:])

    def stft_loss(self, x, y):
        # x, y: [B, C=80, T]
        T = x.size(-1)
        specs = [(256, 64), (128, 32), (64, 16)]
        specs = [(n_fft, hop) for n_fft, hop in specs if n_fft <= T]
        if len(specs) == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        loss = 0.0
        for n_fft, hop in specs:
            loss = loss + F.l1_loss(
                self._stft_mag(x, n_fft=n_fft, hop_length=hop),
                self._stft_mag(y, n_fft=n_fft, hop_length=hop)
            )
        return loss / len(specs)

    def forward(self, mel, labels=None, return_dict=True, **kwargs):
        """
        功能：训练时的前向传播，包含 Loss 计算。
        
        【文件间关系】：
        - 被 `train_vae.py` 中的 `VAETrainer.compute_loss` 调用。
        """
        batch, channels, seq_len = mel.shape

        # [FIX] Global normalization using config stats (not per-sample)
        mel_mean = getattr(self.config, 'mel_mean', -6.589515)
        mel_std = getattr(self.config, 'mel_std', 3.860679)
        mel_normalized = (mel - mel_mean) / mel_std
        
        # Padding
        remainder = seq_len % self.total_stride
        if remainder != 0:
            pad_len = self.total_stride - remainder
            mel_padded = F.pad(mel_normalized, (0, pad_len), mode='reflect')
        else:
            mel_padded = mel_normalized

        # VAE core
        mu, logvar = self.encode(mel_padded)
        z = self.reparameterize(mu, logvar)
        recon_mel = self.decode(z)
        
        # Crop
        if recon_mel.shape[2] != seq_len:
            recon_mel = recon_mel[:, :, :seq_len]
            
        # Loss computed on normalized mel
        if self.use_l1_loss:
            rec_loss = F.l1_loss(recon_mel, mel_normalized)
        else:
            rec_loss = F.mse_loss(recon_mel, mel_normalized)
        
        ssim_loss = self.ssim_loss(recon_mel, mel_normalized)
        stft_l = self.stft_loss(recon_mel, mel_normalized)

        # KL loss
        mu_f = mu.float()
        logvar_f = logvar.float()
        kl_element = 0.5 * (mu_f.pow(2) + logvar_f.exp() - 1 - logvar_f)
        kl_loss = kl_element.mean()

        total_loss = rec_loss + self.ssim_weight * ssim_loss + 0.25 * stft_l + self.config.kl_weight * kl_loss

        if return_dict:
            return {
                "loss": total_loss,
                "rec_loss": rec_loss,
                "ssim_loss": ssim_loss,
                "stft_loss": stft_l,
                "kl_loss": kl_loss,
                "recon_mel": recon_mel * mel_std + mel_mean,  # [FIX] Return denormalized
                "z": z
            }
        return total_loss, recon_mel, z