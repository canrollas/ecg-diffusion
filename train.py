"""
Training script for ECG Diffusion Model
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np

from data.dataset import ECGDataset
from models.condition_encoder import ConditionEncoder
from models.unet import ConditionalUNet
from models.diffusion import DDIMDiffusion
from training.trainer import Trainer
from utils.config import load_config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_models(config: dict, device: str):
    """Create model instances"""
    # Condition encoder
    condition_encoder_config = config["model"]["condition_encoder"]
    condition_encoder = ConditionEncoder(
        condition_dim=condition_encoder_config["condition_dim"],
        base_channels=condition_encoder_config["base_channels"],
        use_cross_attention=condition_encoder_config["use_cross_attention"]
    )
    
    # UNet
    unet_config = config["model"]["unet"]
    unet = ConditionalUNet(
        in_channels=unet_config["in_channels"],
        condition_dim=unet_config["condition_dim"],
        base_channels=unet_config["base_channels"],
        channel_multipliers=tuple(unet_config["channel_multipliers"]),
        time_emb_dim=unet_config["time_emb_dim"],
        num_res_blocks=unet_config["num_res_blocks"],
        attention_resolutions=tuple(unet_config["attention_resolutions"]),
        dropout=unet_config["dropout"]
    )
    
    # Diffusion
    diffusion_config = config["model"]["diffusion"]
    diffusion = DDIMDiffusion(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
        beta_schedule=diffusion_config["beta_schedule"],
        eta=diffusion_config["eta"]
    )
    
    return condition_encoder, unet, diffusion


def create_data_loaders(config: dict):
    """Create data loaders"""
    dataset_config = config["dataset"]
    
    # Training dataset
    try:
        train_dataset = ECGDataset(
            dataset_path=dataset_config["dataset_path"],
            split=dataset_config["train_split"],
            file_prefix=dataset_config.get("file_prefix"),
            normalize=dataset_config["normalize"],
            augment=dataset_config["augment"]
        )
    except ValueError as e:
        print(f"Error creating training dataset: {e}")
        raise
    
    # Pin memory only for CUDA, not for MPS
    pin_memory = dataset_config["pin_memory"] and config.get("device", "cpu") == "cuda"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True,
        num_workers=dataset_config["num_workers"],
        pin_memory=pin_memory
    )
    
    # Validation dataset
    val_dataset = None
    val_loader = None
    if dataset_config.get("val_split"):
        try:
            val_dataset = ECGDataset(
                dataset_path=dataset_config["dataset_path"],
                split=dataset_config["val_split"],
                file_prefix=dataset_config.get("file_prefix"),
                normalize=dataset_config["normalize"],
                augment=False  # No augmentation for validation
            )
        except ValueError as e:
            print(f"Warning: Error creating validation dataset: {e}")
            print("Continuing without validation dataset...")
            val_dataset = None
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=dataset_config["batch_size"],
            shuffle=False,
            num_workers=dataset_config["num_workers"],
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train ECG Diffusion Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config["device"] = args.device
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Device
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, trying MPS...")
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Create models
    print("Creating models...")
    condition_encoder, unet, diffusion = create_models(config, device)
    
    # Print model sizes
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Condition Encoder parameters: {count_parameters(condition_encoder):,}")
    print(f"UNet parameters: {count_parameters(unet):,}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    
    # Create trainer
    training_config = config["training"]
    trainer = Trainer(
        condition_encoder=condition_encoder,
        unet=unet,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        num_epochs=training_config["num_epochs"],
        device=device,
        save_dir=training_config["save_dir"],
        log_dir=training_config["log_dir"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        use_ema=training_config["use_ema"],
        ema_decay=training_config["ema_decay"],
        clip_grad_norm=training_config["clip_grad_norm"],
        save_every=training_config["save_every"],
        val_every=training_config["val_every"],
        use_mixed_precision=training_config["use_mixed_precision"]
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume, load_ema=False)
    
    # Train
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()

