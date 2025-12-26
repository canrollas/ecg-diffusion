"""
Visualization utilities for ECG Diffusion Model
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


def visualize_samples(
    freq_embeddings: np.ndarray,
    images: np.ndarray,
    save_path: Optional[str] = None,
    titles: Optional[list] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize generated samples
    
    Args:
        freq_embeddings: (N, H, W) or (N, 1, H, W) frequency embeddings
        images: (N, H, W) or (N, 1, H, W) images
        save_path: Path to save figure (optional)
        titles: List of titles for each sample (optional)
        figsize: Figure size
    """
    # Remove channel dimension if present
    if len(freq_embeddings.shape) == 4:
        freq_embeddings = freq_embeddings[:, 0, :, :]
    if len(images.shape) == 4:
        images = images[:, 0, :, :]
    
    N = freq_embeddings.shape[0]
    
    fig, axes = plt.subplots(N, 2, figsize=figsize)
    if N == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(N):
        # Frequency embeddings
        ax = axes[i, 0]
        im = ax.imshow(freq_embeddings[i], cmap='seismic', aspect='auto')
        ax.set_title(f"Freq Embeddings {i+1}" if titles is None else titles[i])
        plt.colorbar(im, ax=ax)
        
        # Images
        ax = axes[i, 1]
        im = ax.imshow(images[i], cmap='hot', aspect='auto')
        ax.set_title(f"Images {i+1}")
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def compare_samples(
    generated_freq: np.ndarray,
    generated_images: np.ndarray,
    target_freq: Optional[np.ndarray] = None,
    target_images: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Compare generated samples with targets
    
    Args:
        generated_freq: Generated frequency embeddings
        generated_images: Generated images
        target_freq: Target frequency embeddings (optional)
        target_images: Target images (optional)
        save_path: Path to save figure (optional)
    """
    # Remove channel dimension if present
    if len(generated_freq.shape) == 4:
        generated_freq = generated_freq[:, 0, :, :]
    if len(generated_images.shape) == 4:
        generated_images = generated_images[:, 0, :, :]
    
    if target_freq is not None and len(target_freq.shape) == 4:
        target_freq = target_freq[:, 0, :, :]
    if target_images is not None and len(target_images.shape) == 4:
        target_images = target_images[:, 0, :, :]
    
    N = generated_freq.shape[0]
    num_cols = 4 if target_freq is not None else 2
    
    fig, axes = plt.subplots(N, num_cols, figsize=(5 * num_cols, 5 * N))
    if N == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(N):
        col = 0
        
        # Generated freq
        ax = axes[i, col]
        im = ax.imshow(generated_freq[i], cmap='seismic', aspect='auto')
        ax.set_title(f"Generated Freq {i+1}")
        plt.colorbar(im, ax=ax)
        col += 1
        
        # Generated images
        ax = axes[i, col]
        im = ax.imshow(generated_images[i], cmap='hot', aspect='auto')
        ax.set_title(f"Generated Images {i+1}")
        plt.colorbar(im, ax=ax)
        col += 1
        
        # Target freq (if available)
        if target_freq is not None:
            ax = axes[i, col]
            im = ax.imshow(target_freq[i], cmap='seismic', aspect='auto')
            ax.set_title(f"Target Freq {i+1}")
            plt.colorbar(im, ax=ax)
            col += 1
        
        # Target images (if available)
        if target_images is not None:
            ax = axes[i, col]
            im = ax.imshow(target_images[i], cmap='hot', aspect='auto')
            ax.set_title(f"Target Images {i+1}")
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    plt.show()

