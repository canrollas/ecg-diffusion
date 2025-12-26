"""
Trainer for ECG Diffusion Model
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.condition_encoder import ConditionEncoder
from models.unet import ConditionalUNet
from models.diffusion import DDIMDiffusion
from training.losses import CombinedLoss


class Trainer:
    """Trainer class for ECG Diffusion Model"""
    
    def __init__(
        self,
        condition_encoder: ConditionEncoder,
        unet: ConditionalUNet,
        diffusion: DDIMDiffusion,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 100,
        device: str = "cuda",
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        gradient_accumulation_steps: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        clip_grad_norm: float = 1.0,
        save_every: int = 10,
        val_every: int = 5,
        use_mixed_precision: bool = True
    ):
        """
        Args:
            condition_encoder: Condition encoder model
            unet: UNet model
            diffusion: Diffusion process
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            device: Device to train on
            save_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            gradient_accumulation_steps: Gradient accumulation steps
            use_ema: Use exponential moving average
            ema_decay: EMA decay rate
            clip_grad_norm: Gradient clipping norm
            save_every: Save checkpoint every N epochs
            val_every: Validate every N epochs
            use_mixed_precision: Use mixed precision training
        """
        self.condition_encoder = condition_encoder.to(device)
        self.unet = unet.to(device)
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.clip_grad_norm = clip_grad_norm
        self.save_every = save_every
        self.val_every = val_every
        self.use_mixed_precision = use_mixed_precision
        
        # Create directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Optimizers
        self.condition_encoder_optimizer = optim.AdamW(
            self.condition_encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.unet_optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate schedulers
        self.condition_encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.condition_encoder_optimizer,
            T_max=num_epochs
        )
        self.unet_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.unet_optimizer,
            T_max=num_epochs
        )
        
        # Loss function
        self.loss_fn = CombinedLoss(
            freq_weight=1.0,
            image_weight=1.0,
            consistency_weight=0.1,
            use_consistency=True
        )
        
        # EMA models
        if use_ema:
            self.condition_encoder_ema = self._create_ema_model(self.condition_encoder)
            self.unet_ema = self._create_ema_model(self.unet)
        else:
            self.condition_encoder_ema = None
            self.unet_ema = None
        
        # Mixed precision scaler (only for CUDA, MPS doesn't need it)
        if use_mixed_precision and device == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Determine device type for autocast
        if device == "cuda":
            self.autocast_device = "cuda"
        elif device == "mps":
            self.autocast_device = "cpu"  # MPS doesn't support autocast, use CPU or disable
        else:
            self.autocast_device = "cpu"
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _create_ema_model(self, model: nn.Module) -> Dict:
        """Create EMA state dict"""
        ema_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema_state[name] = param.data.clone()
        return ema_state
    
    def _update_ema(self, model: nn.Module, ema_state: Dict):
        """Update EMA model"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema_state:
                ema_state[name] = (
                    self.ema_decay * ema_state[name] + (1 - self.ema_decay) * param.data
                )
    
    def _load_ema_to_model(self, model: nn.Module, ema_state: Dict):
        """Load EMA weights to model"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema_state:
                param.data.copy_(ema_state[name])
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.condition_encoder.train()
        self.unet.train()
        
        total_loss = 0.0
        loss_components = {
            "total_loss": 0.0,
            "freq_loss": 0.0,
            "image_loss": 0.0,
            "consistency_loss": 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            amp_init = batch["amp_init"].to(self.device)
            emb_init = batch["emb_init"].to(self.device)
            freq_embeddings = batch["freq_embeddings"].to(self.device)
            images = batch["images"].to(self.device)
            
            # Concatenate outputs for diffusion
            x_start = torch.cat([freq_embeddings, images], dim=1)  # (B, 2, H, W)
            
            # Forward pass
            # Only use autocast for CUDA, MPS doesn't support it well
            autocast_enabled = self.use_mixed_precision and self.autocast_device == "cuda"
            with torch.amp.autocast(device_type=self.autocast_device, enabled=autocast_enabled):
                # Encode conditions
                condition = self.condition_encoder(amp_init, emb_init)
                
                # Compute diffusion loss
                loss = self.diffusion.compute_loss(
                    self.unet, x_start, condition, self.device
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.clip_grad_norm > 0:
                    if self.use_mixed_precision and self.scaler is not None:
                        self.scaler.unscale_(self.condition_encoder_optimizer)
                        self.scaler.unscale_(self.unet_optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.condition_encoder.parameters(), self.clip_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.unet.parameters(), self.clip_grad_norm
                    )
                
                # Optimizer step
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.step(self.condition_encoder_optimizer)
                    self.scaler.step(self.unet_optimizer)
                    self.scaler.update()
                else:
                    self.condition_encoder_optimizer.step()
                    self.unet_optimizer.step()
                
                # Zero gradients
                self.condition_encoder_optimizer.zero_grad()
                self.unet_optimizer.zero_grad()
                
                # Update EMA
                if self.use_ema:
                    self._update_ema(self.condition_encoder, self.condition_encoder_ema)
                    self._update_ema(self.unet, self.unet_ema)
                
                self.global_step += 1
            
            # Accumulate losses
            total_loss += loss.item() * self.gradient_accumulation_steps
            loss_components["total_loss"] += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                self.writer.add_scalar("train/loss", loss.item() * self.gradient_accumulation_steps, self.global_step)
                self.writer.add_scalar("train/lr", self.unet_optimizer.param_groups[0]['lr'], self.global_step)
        
        # Average losses
        num_batches = len(self.train_loader)
        total_loss /= num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {"total_loss": total_loss, **loss_components}
    
    def validate(self) -> Dict:
        """Validate model"""
        if self.val_loader is None:
            return {}
        
        self.condition_encoder.eval()
        self.unet.eval()
        
        total_loss = 0.0
        loss_components = {
            "total_loss": 0.0,
            "freq_loss": 0.0,
            "image_loss": 0.0,
            "consistency_loss": 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                amp_init = batch["amp_init"].to(self.device)
                emb_init = batch["emb_init"].to(self.device)
                freq_embeddings = batch["freq_embeddings"].to(self.device)
                images = batch["images"].to(self.device)
                
                x_start = torch.cat([freq_embeddings, images], dim=1)
                
                autocast_enabled = self.use_mixed_precision and self.autocast_device == "cuda"
                with torch.amp.autocast(device_type=self.autocast_device, enabled=autocast_enabled):
                    condition = self.condition_encoder(amp_init, emb_init)
                    loss = self.diffusion.compute_loss(
                        self.unet, x_start, condition, self.device
                    )
                
                total_loss += loss.item()
                loss_components["total_loss"] += loss.item()
        
        num_batches = len(self.val_loader)
        total_loss /= num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {"total_loss": total_loss, **loss_components}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "condition_encoder_state_dict": self.condition_encoder.state_dict(),
            "unet_state_dict": self.unet.state_dict(),
            "condition_encoder_optimizer_state_dict": self.condition_encoder_optimizer.state_dict(),
            "unet_optimizer_state_dict": self.unet_optimizer.state_dict(),
            "condition_encoder_scheduler_state_dict": self.condition_encoder_scheduler.state_dict(),
            "unet_scheduler_state_dict": self.unet_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.use_ema:
            checkpoint["condition_encoder_ema"] = self.condition_encoder_ema
            checkpoint["unet_ema"] = self.unet_ema
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.save_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_ema: bool = False):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.condition_encoder.load_state_dict(checkpoint["condition_encoder_state_dict"])
        self.unet.load_state_dict(checkpoint["unet_state_dict"])
        self.condition_encoder_optimizer.load_state_dict(checkpoint["condition_encoder_optimizer_state_dict"])
        self.unet_optimizer.load_state_dict(checkpoint["unet_optimizer_state_dict"])
        self.condition_encoder_scheduler.load_state_dict(checkpoint["condition_encoder_scheduler_state_dict"])
        self.unet_scheduler.load_state_dict(checkpoint["unet_scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        if self.use_ema and "condition_encoder_ema" in checkpoint:
            self.condition_encoder_ema = checkpoint["condition_encoder_ema"]
            self.unet_ema = checkpoint["unet_ema"]
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        if load_ema:
            self._load_ema_to_model(self.condition_encoder, self.condition_encoder_ema)
            self._load_ema_to_model(self.unet, self.unet_ema)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log training metrics
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"Train Loss: {train_metrics['total_loss']:.6f}")
            self.writer.add_scalar("epoch/train_loss", train_metrics['total_loss'], epoch)
            
            # Validate
            if self.val_loader and (epoch + 1) % self.val_every == 0:
                val_metrics = self.validate()
                print(f"Val Loss: {val_metrics['total_loss']:.6f}")
                self.writer.add_scalar("epoch/val_loss", val_metrics['total_loss'], epoch)
                
                # Save best model
                is_best = val_metrics['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
            else:
                is_best = False
            
            # Update schedulers
            self.condition_encoder_scheduler.step()
            self.unet_scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
                print(f"Checkpoint saved at epoch {epoch + 1}")
        
        print("Training completed!")
        self.writer.close()

