import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import List
import math

# =============================================================================
# 1. SSIM Loss (Perceptual Loss for Mel Spectrogram)
# =============================================================================
class SSIMLoss(nn.Module):
    """
    结构相似性损失 (Structural Similarity Loss)。
    用于在不引入 GAN 的情况下提升 Mel 频谱的清晰度，减少模糊。
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        # img: [B, C, H, W] -> Mel通常是 [B, 80, T]
        # 我们将其视为单通道图像 [B, 1, 80, T] 进行计算
        if img1.dim() == 3:
            img1 = img1.unsqueeze(1)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(1)

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1.0 - ssim_map.mean() # 返回 1 - SSIM 作为 Loss
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)

class AudioVAEConfig(PretrainedConfig):
    model_type = "audio_vae"
    def __init__(
        self, 
        in_channels=80, 
        hidden_channels=512, 
        latent_channels=64, 
        strides: List[int] = [2, 2], 
        kl_weight=0.0001,
        kl_clamp=2.0,
        latent_dropout=0.05,
        norm_num_groups=32,
        use_l1_loss=True,
        ssim_weight=1.0,
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
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv = nn.Sequential(
            # Norm -> Act -> Conv
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
    config_class = AudioVAEConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.strides = config.strides
        self.total_stride = 1
        for s in self.strides: self.total_stride *= s
        print(f"[AcousticVAE] Total Downsampling Stride: {self.total_stride}")
        
        # === 1. Encoder ===
        encoder_layers = []
        current_channels = config.in_channels
        
        # First convolution layer to increase dimension
        encoder_layers.append(
            nn.Conv1d(config.in_channels, config.hidden_channels, 3, 1, 1)
        )
        current_channels = config.hidden_channels

        for stride in config.strides:
            kernel_size = stride * 2
            padding = stride // 2
            encoder_layers.append(
                nn.Sequential(
                    # Downsampling convolution
                    nn.Conv1d(
                        current_channels, 
                        config.hidden_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding
                    ),
                    # Residual block (with Norm)
                    ResBlock(config.hidden_channels, config.norm_num_groups)
                )
            )
            
        # Projection to Latent
        self.encoder = nn.Sequential(
            *encoder_layers,
            nn.GroupNorm(config.norm_num_groups, config.hidden_channels, eps=1e-6), 
            nn.GELU(),
            nn.Conv1d(config.hidden_channels, config.latent_channels * 2, 3, 1, 1)
        )
        
        # === 2. Decoder ===
        decoder_layers = []
        # Input Projection
        decoder_layers.append(
            nn.Sequential(
                nn.Conv1d(config.latent_channels, config.hidden_channels, 3, 1, 1),
                ResBlock(config.hidden_channels, config.norm_num_groups)
            )
        )
        
        # Upsampling
        for stride in reversed(config.strides):
            kernel_size = stride * 2
            padding = stride // 2
            decoder_layers.append(
                nn.Sequential(
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
            
        # Output Layer
        self.decoder_net = nn.Sequential(*decoder_layers)
        self.final_proj = nn.Conv1d(config.hidden_channels, config.in_channels, 3, 1, 1)
        self.use_l1_loss = config.use_l1_loss
        self.ssim_loss = SSIMLoss()
        self.ssim_weight = getattr(config, 'ssim_weight', 1.0)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            if self.config.latent_dropout > 0:
                z = F.dropout(z, p=self.config.latent_dropout, training=True)
            return z
        return mu 

    def decode(self, z):
        h = self.decoder_net(z)
        return self.final_proj(h)

    def forward(self, mel, labels=None, return_dict=True, **kwargs):
        batch, channels, seq_len = mel.shape
        
        remainder = seq_len % self.total_stride
        if remainder != 0:
            pad_len = self.total_stride - remainder
            mel_padded = F.pad(mel, (0, pad_len), mode='reflect')
        else:
            mel_padded = mel

        mu, logvar = self.encode(mel_padded)
        z = self.reparameterize(mu, logvar)
        recon_mel = self.decode(z)
        
        if recon_mel.shape[2] != seq_len:
            recon_mel = recon_mel[:, :, :seq_len]
            
        if self.use_l1_loss:
            rec_loss = F.l1_loss(recon_mel, mel)
        else:
            rec_loss = F.mse_loss(recon_mel, mel)
        ssim_loss = self.ssim_loss(recon_mel, mel)
        
        kl_element = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl_per_step = torch.sum(kl_element, dim=1)
        
        if self.config.kl_clamp > 0:
            kl_per_step = torch.clamp(kl_per_step, min=self.config.kl_clamp)
            
        kl_loss = kl_per_step.mean()
        
        total_loss = rec_loss + self.ssim_weight * ssim_loss + self.config.kl_weight * kl_loss

        if return_dict:
            return {
                "loss": total_loss,
                "rec_loss": rec_loss,
                "ssim_loss": ssim_loss,
                "kl_loss": kl_loss,
                "recon_mel": recon_mel,
                "z": z
            }
        return total_loss, recon_mel, z