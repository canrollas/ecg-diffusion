"""
DDIM Diffusion Process for ECG Generation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class DDIMDiffusion:
    """
    DDIM (Denoising Diffusion Implicit Models) for conditional generation
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        eta: float = 0.0  # DDIM parameter (0.0 = deterministic, 1.0 = DDPM)
    ):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
            beta_schedule: "linear" or "cosine"
            eta: DDIM parameter (0.0 for deterministic sampling)
        """
        self.num_timesteps = num_timesteps
        self.eta = eta
        
        # Beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas = self.betas.float()
        
        # Alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # For DDIM
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
    
    def _cosine_beta_schedule(self, num_timesteps: int) -> torch.Tensor:
        """Cosine beta schedule"""
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: (B, C, H, W) clean data
            t: (B,) timestep tensor
            noise: (B, C, H, W) optional noise (if None, sample new)
        
        Returns:
            x_t: (B, C, H, W) noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Reverse diffusion process: p(x_{t-1} | x_t, condition)
        DDIM sampling
        
        Args:
            model: UNet model
            x_t: (B, C, H, W) noisy data at timestep t
            t: (B,) timestep tensor
            condition: (B, condition_dim, H', W') condition features
            clip_denoised: Clip denoised values to [-1, 1]
        
        Returns:
            x_{t-1}: (B, C, H, W) denoised data at timestep t-1
        """
        # Predict noise
        noise_pred = model(x_t, t, condition)
        
        # Predict x_0
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        pred_x_start = (
            sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred
        )
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        # Get previous timestep
        prev_t = t - 1
        prev_t = torch.clamp(prev_t, min=0)
        
        # DDIM sampling
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, x_t.shape)
        
        pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
        
        random_noise = self.eta * torch.randn_like(x_t)
        pred_dir = pred_dir + torch.sqrt(alpha_cumprod_t_prev) * random_noise
        
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x_start + pred_dir
        
        return x_prev
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        num_inference_steps: int = 50,
        clip_denoised: bool = True,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Full sampling loop
        
        Args:
            model: UNet model
            shape: (B, C, H, W) output shape
            condition: (B, condition_dim, H', W') condition features
            num_inference_steps: Number of sampling steps
            clip_denoised: Clip denoised values
            device: Device to run on
        
        Returns:
            x_0: (B, C, H, W) generated sample
        """
        B = shape[0]
        device = condition.device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Create timestep schedule
        timesteps = np.linspace(self.num_timesteps - 1, 0, num_inference_steps).astype(int)
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition, clip_denoised)
        
        return x
    
    def compute_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Compute training loss
        
        Args:
            model: UNet model
            x_start: (B, C, H, W) clean data
            condition: (B, condition_dim, H', W') condition features
            device: Device to run on
        
        Returns:
            loss: Scalar loss value
        """
        B = x_start.shape[0]
        
        # Sample random timestep
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict noise
        noise_pred = model(x_t, t, condition)
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss
    
    def _extract(
        self,
        arr: torch.Tensor,
        t: torch.Tensor,
        x_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Extract values from array at timestep t
        
        Args:
            arr: (num_timesteps,) array
            t: (B,) timestep tensor
            x_shape: Shape of x (for broadcasting)
        
        Returns:
            extracted: (B, 1, 1, 1) or similar shape for broadcasting
        """
        B = t.shape[0]
        out = arr.to(t.device).gather(0, t.long())
        return out.reshape(B, *((1,) * (len(x_shape) - 1)))

