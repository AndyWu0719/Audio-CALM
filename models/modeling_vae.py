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
        norm_num_groups=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.strides = strides
        self.kl_weight = kl_weight
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
        
        # First convolution layer to increase dimension, no normalization to preserve original feature distribution
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
            nn.GroupNorm(config.norm_num_groups, config.hidden_channels, eps=1e-6), # 瓶颈处 Norm
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
        # [Key] The last layer does not add Norm or activation function, keeping linear output to fit any value of Log-Mel
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
            return mu + eps * std
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
            
        # 4. Loss
        rec_loss = F.mse_loss(recon_mel, mel)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (batch * seq_len) 
        
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