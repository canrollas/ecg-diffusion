"""
DDIM Sampler for ECG Diffusion Model Inference
"""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.condition_encoder import ConditionEncoder
from models.unet import ConditionalUNet
from models.diffusion import DDIMDiffusion


class DDIMSampler:
    """
    DDIM Sampler for generating ECG data from conditions
    """
    
    def __init__(
        self,
        condition_encoder: ConditionEncoder,
        unet: ConditionalUNet,
        diffusion: DDIMDiffusion,
        device: str = "cuda",
        use_ema: bool = True
    ):
        """
        Args:
            condition_encoder: Condition encoder model
            unet: UNet model
            diffusion: Diffusion process
            device: Device to run on
            use_ema: Use EMA weights if available
        """
        self.condition_encoder = condition_encoder.to(device)
        self.unet = unet.to(device)
        self.diffusion = diffusion
        self.device = device
        self.use_ema = use_ema
        
        self.condition_encoder.eval()
        self.unet.eval()
    
    def sample(
        self,
        amp_init: torch.Tensor,
        emb_init: torch.Tensor,
        num_inference_steps: int = 50,
        output_shape: Optional[Tuple[int, ...]] = None,
        clip_denoised: bool = True,
        progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples from conditions
        
        Args:
            amp_init: (B, 1, H, W) amplitude initialization
            emb_init: (B, 1, H, W) embedding initialization
            num_inference_steps: Number of DDIM sampling steps
            output_shape: Output shape (B, 2, H, W). If None, inferred from inputs
            clip_denoised: Clip denoised values to [-1, 1]
            progress: Show progress bar
        
        Returns:
            Dictionary with 'freq_embeddings' and 'images'
        """
        B = amp_init.shape[0]
        
        # Determine output shape
        if output_shape is None:
            # Use the larger of the two input dimensions
            H = max(amp_init.shape[2], emb_init.shape[2])
            W = max(amp_init.shape[3], emb_init.shape[3])
            output_shape = (B, 2, H, W)
        else:
            H, W = output_shape[2], output_shape[3]
        
        # Encode conditions
        with torch.no_grad():
            condition = self.condition_encoder(amp_init, emb_init)
        
        # Generate samples
        with torch.no_grad():
            samples = self.diffusion.p_sample_loop(
                model=self.unet,
                shape=output_shape,
                condition=condition,
                num_inference_steps=num_inference_steps,
                clip_denoised=clip_denoised,
                device=self.device
            )
        
        # Split into freq_embeddings and images
        freq_embeddings = samples[:, 0:1, :, :]  # (B, 1, H, W)
        images = samples[:, 1:2, :, :]  # (B, 1, H, W)
        
        return {
            "freq_embeddings": freq_embeddings,
            "images": images,
            "condition": condition
        }
    
    def sample_batch(
        self,
        amp_init: torch.Tensor,
        emb_init: torch.Tensor,
        num_inference_steps: int = 50,
        batch_size: int = 4,
        clip_denoised: bool = True,
        progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples in batches
        
        Args:
            amp_init: (N, 1, H, W) amplitude initialization
            emb_init: (N, 1, H, W) embedding initialization
            num_inference_steps: Number of DDIM sampling steps
            batch_size: Batch size for generation
            clip_denoised: Clip denoised values
            progress: Show progress bar
        
        Returns:
            Dictionary with 'freq_embeddings' and 'images' for all samples
        """
        N = amp_init.shape[0]
        all_freq_embeddings = []
        all_images = []
        
        iterator = range(0, N, batch_size)
        if progress:
            iterator = tqdm(iterator, desc="Generating samples")
        
        for i in iterator:
            end_idx = min(i + batch_size, N)
            batch_amp = amp_init[i:end_idx]
            batch_emb = emb_init[i:end_idx]
            
            results = self.sample(
                batch_amp,
                batch_emb,
                num_inference_steps=num_inference_steps,
                clip_denoised=clip_denoised,
                progress=False
            )
            
            all_freq_embeddings.append(results["freq_embeddings"])
            all_images.append(results["images"])
        
        # Concatenate all results
        freq_embeddings = torch.cat(all_freq_embeddings, dim=0)
        images = torch.cat(all_images, dim=0)
        
        return {
            "freq_embeddings": freq_embeddings,
            "images": images
        }
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_ema: bool = True
    ):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_ema: Load EMA weights if available
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load condition encoder
        if "condition_encoder_state_dict" in checkpoint:
            self.condition_encoder.load_state_dict(checkpoint["condition_encoder_state_dict"])
        elif "condition_encoder" in checkpoint:
            self.condition_encoder.load_state_dict(checkpoint["condition_encoder"])
        
        # Load UNet
        if "unet_state_dict" in checkpoint:
            self.unet.load_state_dict(checkpoint["unet_state_dict"])
        elif "unet" in checkpoint:
            self.unet.load_state_dict(checkpoint["unet"])
        
        # Load EMA weights if requested
        if load_ema and self.use_ema:
            if "condition_encoder_ema" in checkpoint:
                ema_state = checkpoint["condition_encoder_ema"]
                for name, param in self.condition_encoder.named_parameters():
                    if name in ema_state:
                        param.data.copy_(ema_state[name])
            
            if "unet_ema" in checkpoint:
                ema_state = checkpoint["unet_ema"]
                for name, param in self.unet.named_parameters():
                    if name in ema_state:
                        param.data.copy_(ema_state[name])
        
        self.condition_encoder.eval()
        self.unet.eval()
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_samples(
        self,
        samples: Dict[str, torch.Tensor],
        output_dir: str,
        filenames: Optional[list] = None,
        denormalize_fn: Optional[callable] = None
    ):
        """
        Save generated samples to disk
        
        Args:
            samples: Dictionary with 'freq_embeddings' and 'images'
            output_dir: Directory to save samples
            filenames: List of filenames (optional)
            denormalize_fn: Function to denormalize data (optional)
        """
        import numpy as np
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        freq_embeddings = samples["freq_embeddings"]
        images = samples["images"]
        
        B = freq_embeddings.shape[0]
        
        for i in range(B):
            # Get filenames
            if filenames is not None and i < len(filenames):
                base_name = filenames[i]
            else:
                base_name = f"sample_{i:04d}"
            
            # Denormalize if function provided
            if denormalize_fn is not None:
                freq = denormalize_fn(freq_embeddings[i], "freq_embeddings")
                img = denormalize_fn(images[i], "images")
            else:
                freq = freq_embeddings[i].cpu().numpy()
                img = images[i].cpu().numpy()
            
            # Remove channel dimension if present
            if len(freq.shape) == 3 and freq.shape[0] == 1:
                freq = freq[0]
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img[0]
            
            # Save as numpy arrays
            freq_path = output_dir / f"{base_name}_freq_embeddings.npy"
            img_path = output_dir / f"{base_name}_images.npy"
            
            np.save(freq_path, freq)
            np.save(img_path, img)
        
        print(f"Saved {B} samples to {output_dir}")

