"""
Conditional UNet for ECG Diffusion Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (B,) timestep tensor
        
        Returns:
            embeddings: (B, dim) position embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with condition and timestep"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Main conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Condition projection (for cross-attention)
        if condition_dim > 0:
            self.condition_proj = nn.Conv2d(condition_dim, out_channels, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        condition: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) input features
            time_emb: (B, time_emb_dim) time embeddings
            condition: (B, condition_dim, H', W') condition features (optional)
        
        Returns:
            out: (B, out_channels, H, W) output features
        """
        residual = self.residual_conv(x)
        
        # First conv
        x = self.conv1(x)
        x = self.norm1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)  # (B, out_channels)
        time_emb = time_emb[:, :, None, None]  # (B, out_channels, 1, 1)
        x = x + time_emb
        
        # Add condition if provided
        if condition is not None:
            # Resize condition to match x if needed
            if condition.shape[2:] != x.shape[2:]:
                condition = F.interpolate(condition, size=x.shape[2:], mode='bilinear', align_corners=False)
            condition_proj = self.condition_proj(condition)
            x = x + condition_proj
        
        x = F.silu(x)
        x = self.dropout(x)
        
        # Second conv
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        
        # Residual connection
        out = x + residual
        
        return out


class ConditionalUNet(nn.Module):
    """
    Conditional UNet for diffusion model
    
    Takes noisy data and conditions, predicts noise
    """
    
    def __init__(
        self,
        in_channels: int = 2,  # freq_embeddings + images
        condition_dim: int = 256,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4, 8),
        time_emb_dim: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Input channels (2 for freq + images)
            condition_dim: Condition feature dimension
            base_channels: Base number of channels
            channel_multipliers: Channel multipliers for each resolution level
            time_emb_dim: Time embedding dimension
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions to apply attention
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        input_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            output_ch = base_channels * mult
            
            # Residual blocks at this resolution
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        input_ch, output_ch, condition_dim, time_emb_dim, dropout
                    )
                )
                input_ch = output_ch
            
            self.encoder_blocks.append(nn.ModuleList(blocks))
            
            # Downsample (except last level)
            if i < len(channel_multipliers) - 1:
                self.downsample_layers.append(
                    nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=2, padding=1)
                )
            else:
                self.downsample_layers.append(nn.Identity())
        
        # Middle block
        self.middle_block1 = ResidualBlock(
            input_ch, input_ch, condition_dim, time_emb_dim, dropout
        )
        self.middle_block2 = ResidualBlock(
            input_ch, input_ch, condition_dim, time_emb_dim, dropout
        )
        
        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        # Decoder channel tracking
        # After upsample and skip connection, we have: upsample_output_ch + skip_ch
        decoder_input_ch = input_ch  # Start with middle block output channels
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            skip_ch = base_channels * mult  # Skip connection channel size
            
            # Calculate input channels for this decoder level
            # This is: upsample_output_ch + skip_ch
            if i == 0:
                # First decoder level: middle output (no upsample) + skip connection
                decoder_level_input_ch = decoder_input_ch + skip_ch
            else:
                # Subsequent levels: upsample output + skip connection
                # Upsample takes previous level's output_ch and produces next_ch
                # The next_ch is the same as skip_ch for this level (upsample goes from prev to current)
                # Previous level output is skip_ch of previous level
                prev_mult = channel_multipliers[-(i-1)]  # Previous level's multiplier
                prev_output_ch = base_channels * prev_mult  # Previous level's output
                # Upsample: prev_output_ch -> skip_ch (current level)
                upsample_output_ch = skip_ch  # After upsample, we get skip_ch
                decoder_level_input_ch = upsample_output_ch + skip_ch
            
            # Residual blocks at this resolution
            blocks = []
            current_input_ch = decoder_level_input_ch
            for _ in range(num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        current_input_ch, skip_ch, condition_dim, time_emb_dim, dropout
                    )
                )
                current_input_ch = skip_ch
            
            self.decoder_blocks.append(nn.ModuleList(blocks))
            
            # Upsample (except last level)
            if i < len(channel_multipliers) - 1:
                # Next level's channel (going backwards)
                next_mult = channel_multipliers[-(i+2)]
                next_ch = base_channels * next_mult
                self.upsample_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(skip_ch, next_ch, kernel_size=4, stride=2, padding=1),
                        nn.GroupNorm(8, next_ch),
                        nn.SiLU()
                    )
                )
            else:
                self.upsample_layers.append(nn.Identity())
        
        # Output projection
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) noisy input data
            timestep: (B,) timestep tensor
            condition: (B, condition_dim, H', W') condition features
        
        Returns:
            noise_pred: (B, in_channels, H, W) predicted noise
        """
        # Time embedding
        time_emb = self.time_embed(timestep)  # (B, time_emb_dim)
        
        # Input projection
        x = self.input_conv(x)  # (B, base_channels, H, W)
        
        # Encoder
        encoder_features = []
        for blocks, downsample in zip(self.encoder_blocks, self.downsample_layers):
            for block in blocks:
                x = block(x, time_emb, condition)
            encoder_features.append(x)
            x = downsample(x)
        
        # Middle
        x = self.middle_block1(x, time_emb, condition)
        x = self.middle_block2(x, time_emb, condition)
        
        # Decoder
        for blocks, upsample, skip_features in zip(
            self.decoder_blocks, self.upsample_layers, reversed(encoder_features)
        ):
            x = upsample(x)
            # Resize skip connection to match x if needed
            if x.shape[2:] != skip_features.shape[2:]:
                skip_features = F.interpolate(
                    skip_features, size=x.shape[2:], mode='bilinear', align_corners=False
                )
            # Concatenate skip connection
            x = torch.cat([x, skip_features], dim=1)
            for block in blocks:
                x = block(x, time_emb, condition)
        
        # Output
        x = self.output_norm(x)
        x = F.silu(x)
        noise_pred = self.output_conv(x)
        
        return noise_pred

