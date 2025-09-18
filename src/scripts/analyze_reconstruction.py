#!/usr/bin/env python3
"""
Analyze VAE reconstructions on TEMPO validation data.
"""

import argparse
import yaml
import sys
import json
import torch
import numpy as np
import netCDF4 as nc
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model import get_model
from src.utils import init_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to analysis config')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')
    parser.add_argument('--debug', action='store_true', help='Process only 1 file')
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")

    # Setup output using init_directory
    output_dir = init_directory(config['output_dir'], overwrite=args.overwrite)

    # Copy config to output directory (as per repo_usage.md)
    import shutil
    shutil.copy2(args.config_path, output_dir / 'config.yaml')

    # Load split info to get validation files
    tiles_path = Path(config['data']['tiles_path'])
    split_info_path = tiles_path / 'split_info.json'
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    # Get validation NC files
    val_nc_files = [Path(config['data']['nc_path']) / 'raw' / f for f in split_info['val_sources'].values()]

    if args.debug:
        val_nc_files = val_nc_files[:1]

    # Load normalization stats
    mean_spectrum = torch.load(tiles_path / 'mean_spectrum.pt')
    std_spectrum = torch.load(tiles_path / 'std_spectrum.pt')

    # Load model
    with open(config['model']['training_config_path'], 'r') as f:
        train_config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model params exactly like training script
    model_params = {
        "architecture_type": "vae",
        "architecture_params": {
            "enc_dec_params": train_config.get('model', {})
        },
        "optimizer_type": "AdamW",
        "optimizer_params": {
            "lr": train_config.get('optimizer', {}).get('lr', 0.0001),
            "weight_decay": train_config.get('optimizer', {}).get('weight_decay', 0.05),
            "betas": train_config.get('optimizer', {}).get('betas', [0.9, 0.95])
        }
    }

    model = get_model(model_params, device)

    # Load checkpoint
    checkpoint = torch.load(config['model']['checkpoint_path'], map_location=device)

    # Check if checkpoint has model_state_dict (from trainer) or is direct state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Set seed for VAE latent sampling
    torch.manual_seed(config['seed'])

    # Process each validation file
    for nc_file in tqdm(val_nc_files, desc="Processing"):
        # Read radiance data
        with nc.Dataset(nc_file) as f:
            rad = torch.tensor(np.array(f['band_290_490_nm']['radiance'][...]))

        # Apply EXACT same normalization as tiles
        # 1. Log transform with min_radiance=1.0
        log_rad = torch.log(torch.clamp(rad, 1.0, torch.inf))

        # 2. Normalize with global mean/std
        z_rad = (log_rad - mean_spectrum) / (std_spectrum + 1e-8)

        # 3. Clip to [-10, 10] (from prepare_tiles config)
        z_rad = torch.clamp(z_rad, -10, 10)

        # Get largest 64 multiple that fits (for encoder/decoder compatibility)
        # z_rad is [mirror, track, spectral] = [131, 2048, 1028]
        mirror_size, track_size, n_spectral = z_rad.shape
        mirror_crop = (mirror_size // 64) * 64
        track_crop = (track_size // 64) * 64
        z_rad_crop = z_rad[:mirror_crop, :track_crop, :]  # [mirror_crop, track_crop, 1028]

        # Convert to channel-first for model: [1, 1028, mirror_crop, track_crop]
        input_tensor = z_rad_crop.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

        # Reconstruct
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            recon = model(input_tensor)

        # Convert back: [1, 1028, mirror_crop, track_crop] -> [mirror_crop, track_crop, 1028]
        recon = recon.squeeze(0).permute(1, 2, 0).cpu()

        # Ground truth
        gt = z_rad_crop  # [mirror_crop, track_crop, 1028]

        # Visualization based on config
        viz_config = config.get('visualization', {})
        viz_mode = viz_config.get('mode', 'single_channel')

        if viz_mode == 'pca_rgb':
            # Load PCA components
            pca_path = Path(viz_config.get('pca_components_path'))
            if not pca_path.exists():
                raise ValueError(f"PCA components not found at {pca_path}")

            pca_data = torch.load(pca_path, weights_only=False)
            pca_components = pca_data['components']  # [3, 1028]
            pca_mean = pca_data['mean']  # [1028]

            # Project data onto PCA components
            # First center the data
            gt_centered = gt - pca_mean
            recon_centered = recon - pca_mean

            # Project: [H, W, 1028] @ [1028, 3] -> [H, W, 3]
            gt_rgb = torch.matmul(gt_centered.cpu(), pca_components.T)
            recon_rgb = torch.matmul(recon_centered, pca_components.T)

            # Normalize to [0, 1] for RGB display
            # Use percentiles from GT for robust and consistent scaling
            for i in range(3):
                channel_gt = gt_rgb[:, :, i]
                vmin = torch.quantile(channel_gt, 0.02)
                vmax = torch.quantile(channel_gt, 0.98)

                # Apply same scaling to both GT and reconstruction
                gt_rgb[:, :, i] = torch.clamp((channel_gt - vmin) / (vmax - vmin + 1e-8), 0, 1)
                recon_rgb[:, :, i] = torch.clamp((recon_rgb[:, :, i] - vmin) / (vmax - vmin + 1e-8), 0, 1)

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Ground truth RGB
            axes[0].imshow(gt_rgb.numpy(), aspect='auto')
            axes[0].set_title('Ground Truth (PCA RGB)')
            axes[0].axis('off')

            # Reconstruction RGB
            axes[1].imshow(recon_rgb.numpy(), aspect='auto')
            axes[1].set_title('Reconstruction (PCA RGB)')
            axes[1].axis('off')

            plt.suptitle(f'{nc_file.stem} - PCA Components as RGB')
            save_suffix = '_pca_rgb'

        else:  # single_channel mode
            ch = viz_config.get('single_channel', 500)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Get vmin/vmax from ground truth for consistent scaling
            gt_channel = gt[:, :, ch].cpu().numpy()
            vmin, vmax = gt_channel.min(), gt_channel.max()

            # Ground truth
            im1 = axes[0].imshow(gt_channel, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046)

            # Reconstruction with same scale
            im2 = axes[1].imshow(recon[:, :, ch].cpu().numpy(), cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            axes[1].set_title('Reconstruction')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046)

            plt.suptitle(f'{nc_file.stem} - Channel {ch}')
            save_suffix = f'_ch{ch}'

        plt.tight_layout()

        # Save
        save_path = output_dir / f'{nc_file.stem}{save_suffix}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved {save_path}")


if __name__ == '__main__':
    main()