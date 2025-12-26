"""
Inference script for ECG Diffusion Model
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import ECGDataset
from models.condition_encoder import ConditionEncoder
from models.unet import ConditionalUNet
from models.diffusion import DDIMDiffusion
from inference.sampler import DDIMSampler
from utils.config import load_config
from utils.visualization import visualize_samples, compare_samples


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


def main():
    parser = argparse.ArgumentParser(description="Inference with ECG Diffusion Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory with input files (amp_init, emb_init). If None, uses dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of inference steps (overrides config)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize generated samples"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with ground truth (requires dataset)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config["device"] = args.device
    
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
    
    # Create sampler
    inference_config = config["inference"]
    sampler = DDIMSampler(
        condition_encoder=condition_encoder,
        unet=unet,
        diffusion=diffusion,
        device=device,
        use_ema=inference_config.get("use_ema", True)
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    sampler.load_checkpoint(args.checkpoint, load_ema=inference_config.get("use_ema", True))
    
    # Prepare inputs
    if args.input_dir:
        # Load from directory
        input_dir = Path(args.input_dir)
        amp_init_files = sorted(list((input_dir / "amp_init").glob("*.npy")))
        emb_init_files = sorted(list((input_dir / "emb_init").glob("*.npy")))
        
        amp_init_list = []
        emb_init_list = []
        filenames = []
        
        for amp_file, emb_file in zip(amp_init_files[:args.num_samples], emb_init_files[:args.num_samples]):
            amp_init_list.append(np.load(amp_file))
            emb_init_list.append(np.load(emb_file))
            filenames.append(amp_file.stem)
        
        amp_init = torch.FloatTensor(np.array(amp_init_list)).unsqueeze(1).to(device)
        emb_init = torch.FloatTensor(np.array(emb_init_list)).unsqueeze(1).to(device)
    else:
        # Load from dataset
        dataset_config = config["dataset"]
        dataset = ECGDataset(
            dataset_path=dataset_config["dataset_path"],
            split=dataset_config.get("val_split", "val"),
            file_prefix=dataset_config.get("file_prefix"),
            normalize=dataset_config["normalize"],
            augment=False
        )
        
        dataloader = DataLoader(dataset, batch_size=args.num_samples, shuffle=False)
        batch = next(iter(dataloader))
        
        amp_init = batch["amp_init"].to(device)
        emb_init = batch["emb_init"].to(device)
        filenames = batch.get("filename", [f"sample_{i}" for i in range(args.num_samples)])
        
        # Store targets for comparison
        target_freq = batch["freq_embeddings"].cpu().numpy()
        target_images = batch["images"].cpu().numpy()
    
    # Generate samples
    num_steps = args.num_steps if args.num_steps else inference_config["num_inference_steps"]
    print(f"Generating {args.num_samples} samples with {num_steps} steps...")
    
    results = sampler.sample(
        amp_init=amp_init,
        emb_init=emb_init,
        num_inference_steps=num_steps,
        clip_denoised=inference_config["clip_denoised"],
        progress=True
    )
    
    freq_embeddings = results["freq_embeddings"].cpu().numpy()
    images = results["images"].cpu().numpy()
    
    # Save outputs
    output_dir = args.output_dir if args.output_dir else inference_config["output_dir"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize function (if dataset available)
    denormalize_fn = None
    if not args.input_dir:
        def denormalize(data, key):
            return dataset.get_denormalized(torch.FloatTensor(data), key)
        denormalize_fn = denormalize
    
    sampler.save_samples(
        results,
        output_dir=str(output_dir),
        filenames=filenames if isinstance(filenames, list) else None,
        denormalize_fn=denormalize_fn
    )
    
    # Visualize
    if args.visualize:
        print("Visualizing samples...")
        visualize_samples(
            freq_embeddings,
            images,
            save_path=str(output_dir / "generated_samples.png")
        )
    
    # Compare with ground truth
    if args.compare and not args.input_dir:
        print("Comparing with ground truth...")
        compare_samples(
            freq_embeddings,
            images,
            target_freq=target_freq,
            target_images=target_images,
            save_path=str(output_dir / "comparison.png")
        )
    
    print(f"Generated samples saved to {output_dir}")


if __name__ == "__main__":
    main()

