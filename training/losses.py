"""
Loss functions for ECG Diffusion Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Standard diffusion loss (MSE between predicted and actual noise)"""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noise_pred: (B, C, H, W) predicted noise
            noise_target: (B, C, H, W) target noise
        
        Returns:
            loss: Scalar loss value
        """
        return F.mse_loss(noise_pred, noise_target, reduction=self.reduction)


class MultiOutputLoss(nn.Module):
    """
    Loss for multiple outputs (freq_embeddings and images)
    Computes separate losses and combines them
    """
    
    def __init__(
        self,
        freq_weight: float = 1.0,
        image_weight: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Args:
            freq_weight: Weight for frequency embeddings loss
            image_weight: Weight for images loss
            reduction: Loss reduction method
        """
        super().__init__()
        self.freq_weight = freq_weight
        self.image_weight = image_weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noise_pred: (B, 2, H, W) predicted noise [freq, image]
            noise_target: (B, 2, H, W) target noise [freq, image]
        
        Returns:
            loss: Weighted combination of losses
        """
        # Split into freq and image channels
        freq_pred = noise_pred[:, 0:1, :, :]  # (B, 1, H, W)
        image_pred = noise_pred[:, 1:2, :, :]  # (B, 1, H, W)
        
        freq_target = noise_target[:, 0:1, :, :]  # (B, 1, H, W)
        image_target = noise_target[:, 1:2, :, :]  # (B, 1, H, W)
        
        # Compute separate losses
        freq_loss = self.mse_loss(freq_pred, freq_target)
        image_loss = self.mse_loss(image_pred, image_target)
        
        # Weighted combination
        total_loss = self.freq_weight * freq_loss + self.image_weight * image_loss
        
        return total_loss, {
            "freq_loss": freq_loss.item() if isinstance(freq_loss, torch.Tensor) else freq_loss,
            "image_loss": image_loss.item() if isinstance(image_loss, torch.Tensor) else image_loss,
            "total_loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        }


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between freq_embeddings and images
    Encourages consistency in the generated outputs
    """
    
    def __init__(self, weight: float = 0.1, reduction: str = "mean"):
        """
        Args:
            weight: Weight for consistency loss
            reduction: Loss reduction method
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(
        self,
        freq_pred: torch.Tensor,
        image_pred: torch.Tensor,
        freq_target: torch.Tensor,
        image_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            freq_pred: (B, 1, H, W) predicted frequency embeddings
            image_pred: (B, 1, H, W) predicted images
            freq_target: (B, 1, H, W) target frequency embeddings
            image_target: (B, 1, H, W) target images
        
        Returns:
            loss: Consistency loss value
        """
        # Compute gradients or differences to measure consistency
        # Simple approach: encourage similar patterns
        
        # Resize to same spatial dimensions if needed
        if freq_pred.shape[2:] != image_pred.shape[2:]:
            freq_pred = F.interpolate(
                freq_pred, size=image_pred.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Compute correlation or similarity
        # Normalize
        freq_norm = (freq_pred - freq_pred.mean()) / (freq_pred.std() + 1e-8)
        image_norm = (image_pred - image_pred.mean()) / (image_pred.std() + 1e-8)
        
        # Correlation loss (encourage positive correlation)
        correlation = (freq_norm * image_norm).mean()
        consistency_loss = -correlation  # Negative because we want to maximize correlation
        
        return self.weight * consistency_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for training
    Combines diffusion loss, multi-output loss, and consistency loss
    """
    
    def __init__(
        self,
        freq_weight: float = 1.0,
        image_weight: float = 1.0,
        consistency_weight: float = 0.1,
        use_consistency: bool = True
    ):
        """
        Args:
            freq_weight: Weight for frequency embeddings loss
            image_weight: Weight for images loss
            consistency_weight: Weight for consistency loss
            use_consistency: Whether to use consistency loss
        """
        super().__init__()
        self.use_consistency = use_consistency
        self.multi_output_loss = MultiOutputLoss(freq_weight, image_weight)
        if use_consistency:
            self.consistency_loss = ConsistencyLoss(consistency_weight)
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        x_pred: torch.Tensor = None,
        x_target: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            noise_pred: (B, 2, H, W) predicted noise
            noise_target: (B, 2, H, W) target noise
            x_pred: (B, 2, H, W) predicted clean data (optional, for consistency)
            x_target: (B, 2, H, W) target clean data (optional, for consistency)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Multi-output loss
        multi_loss, loss_dict = self.multi_output_loss(noise_pred, noise_target)
        total_loss = multi_loss
        
        # Consistency loss (if enabled and data provided)
        if self.use_consistency and x_pred is not None and x_target is not None:
            freq_pred = x_pred[:, 0:1, :, :]
            image_pred = x_pred[:, 1:2, :, :]
            freq_target = x_target[:, 0:1, :, :]
            image_target = x_target[:, 1:2, :, :]
            
            consistency = self.consistency_loss(
                freq_pred, image_pred, freq_target, image_target
            )
            total_loss = total_loss + consistency
            loss_dict["consistency_loss"] = consistency.item() if isinstance(consistency, torch.Tensor) else consistency
        
        loss_dict["total_loss"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict

