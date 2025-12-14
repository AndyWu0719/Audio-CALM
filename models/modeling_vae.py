import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import List

class AudioVAEConfig(PretrainedConfig):
    model_type = "audio_vae"
    def __init__(
        self, 
        in_channels=80, 
        hidden_channels=512, 
        latent_channels=64, 
        strides: List[int] = [2, 2, 2, 2], 
        kl_weight=0.0001,
        kl_clamp=2.0,
        latent_dropout=0.05,
        norm_num_groups=32,
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
        
        # 1. Padding
        remainder = seq_len % self.total_stride
        if remainder != 0:
            pad_len = self.total_stride - remainder
            mel_padded = F.pad(mel, (0, pad_len), mode='reflect')
        else:
            mel_padded = mel

        # 2. VAE Flow
        mu, logvar = self.encode(mel_padded)
        z = self.reparameterize(mu, logvar)
        recon_mel = self.decode(z)
        
        # 3. Un-padding
        if recon_mel.shape[2] != seq_len:
            recon_mel = recon_mel[:, :, :seq_len]
            
        # 4. Loss with KL Clamping (Free Bits) [MODIFIED]
        rec_loss = F.mse_loss(recon_mel, mel)
        
        # Calculate KL per element: 0.5 * (mu^2 + var - 1 - logvar)
        # Shape: [Batch, LatentDim, Time]
        kl_element = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        
        # Sum over Latent Dimension (Dim 1) to get KL per timestep
        # Shape: [Batch, Time]
        kl_per_step = torch.sum(kl_element, dim=1)
        
        if self.config.kl_clamp > 0:
            kl_per_step = torch.clamp(kl_per_step, min=self.config.kl_clamp)
            
        # Average over Batch and Time
        kl_loss = kl_per_step.mean()
        
        total_loss = rec_loss + self.config.kl_weight * kl_loss
        
        if return_dict:
            return {
                "loss": total_loss,
                "rec_loss": rec_loss,
                "kl_loss": kl_loss,
                "recon_mel": recon_mel,
                "z": z
            }
        return total_loss, recon_mel, z