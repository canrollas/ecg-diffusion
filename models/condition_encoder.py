"""
Condition Encoder for ECG Diffusion Model
Encodes amp_init and emb_init inputs into condition features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEncoder(nn.Module):
    """
    Encodes two input conditions (amp_init, emb_init) into unified condition features
    """
    
    def __init__(
        self,
        condition_dim: int = 256,
        base_channels: int = 64,
        use_cross_attention: bool = True
    ):
        """
        Args:
            condition_dim: Output condition feature dimension
            base_channels: Base number of channels for encoders
            use_cross_attention: Use cross-attention for fusion (else concatenation)
        """
        super().__init__()
        self.condition_dim = condition_dim
        self.base_channels = base_channels
        self.use_cross_attention = use_cross_attention
        
        # Encoder for amp_init
        self.amp_encoder = self._build_encoder(base_channels)
        
        # Encoder for emb_init
        self.emb_encoder = self._build_encoder(base_channels)
        
        # Fusion mechanism
        if use_cross_attention:
            self.fusion = CrossAttentionFusion(base_channels * 4, condition_dim)
        else:
            self.fusion = ConcatenationFusion(base_channels * 4, condition_dim)
    
    def _build_encoder(self, base_channels: int) -> nn.Module:
        """Build encoder network for one input"""
        return nn.Sequential(
            # First conv block
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            
            # Second conv block
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            
            # Third conv block
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            
            # Fourth conv block
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
    
    def forward(self, amp_init: torch.Tensor, emb_init: torch.Tensor) -> torch.Tensor:
        """
        Args:
            amp_init: (B, 1, H, W) amplitude initialization
            emb_init: (B, 1, H, W) embedding initialization
        
        Returns:
            condition: (B, condition_dim, H', W') condition features
        """
        # Encode each input
        amp_features = self.amp_encoder(amp_init)  # (B, base_channels*4, H', W')
        emb_features = self.emb_encoder(emb_init)  # (B, base_channels*4, H', W')
        
        # Fuse features
        condition = self.fusion(amp_features, emb_features)  # (B, condition_dim, H', W')
        
        return condition


class CrossAttentionFusion(nn.Module):
    """Cross-attention based fusion"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Query, Key, Value projections
        self.q_proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        
        # Layer norm
        self.norm = nn.GroupNorm(8, output_dim)
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features1: (B, C, H, W) first feature map
            features2: (B, C, H, W) second feature map
        
        Returns:
            fused: (B, output_dim, H, W) fused features
        """
        B, C, H, W = features1.shape
        
        # Reshape for attention: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        f1_flat = features1.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        f2_flat = features2.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Project to Q, K, V
        Q = self.q_proj(features1).view(B, self.output_dim, H * W).transpose(1, 2)  # (B, H*W, output_dim)
        K = self.k_proj(features2).view(B, self.output_dim, H * W).transpose(1, 2)  # (B, H*W, output_dim)
        V = self.v_proj(features2).view(B, self.output_dim, H * W).transpose(1, 2)  # (B, H*W, output_dim)
        
        # Scaled dot-product attention
        scale = (self.output_dim ** -0.5)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)  # (B, H*W, H*W)
        out = attn @ V  # (B, H*W, output_dim)
        
        # Reshape back: (B, H*W, output_dim) -> (B, output_dim, H*W) -> (B, output_dim, H, W)
        out = out.transpose(1, 2).view(B, self.output_dim, H, W)
        
        # Output projection and normalization
        out = self.out_proj(out)
        out = self.norm(out)
        
        return out


class ConcatenationFusion(nn.Module):
    """Simple concatenation based fusion"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_dim * 2, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, output_dim),
            nn.SiLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=1),
            nn.GroupNorm(8, output_dim),
            nn.SiLU(),
        )
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features1: (B, C, H, W) first feature map
            features2: (B, C, H, W) second feature map
        
        Returns:
            fused: (B, output_dim, H, W) fused features
        """
        # Concatenate along channel dimension
        fused = torch.cat([features1, features2], dim=1)  # (B, C*2, H, W)
        
        # Project to output dimension
        fused = self.fusion_conv(fused)  # (B, output_dim, H, W)
        
        return fused

