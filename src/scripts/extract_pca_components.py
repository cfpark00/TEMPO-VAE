#!/usr/bin/env python3
"""
Extract PCA components from TEMPO spectral data.
"""

import argparse
import yaml
import sys
from pathlib import Path
import numpy as np
import torch
import netCDF4 as nc
from tqdm import tqdm
import shutil
from sklearn.decomposition import PCA
import pickle

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import init_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if 'input_dir' not in config:
        raise ValueError("FATAL: 'input_dir' is required in config")

    # Setup paths
    input_dir = Path(config['input_dir'])
    if not input_dir.exists():
        raise ValueError(f"FATAL: input_dir doesn't exist: {input_dir}")

    output_dir = init_directory(config['output_dir'], overwrite=args.overwrite)

    # Copy config
    shutil.copy2(args.config_path, output_dir / 'config.yaml')

    # Load normalization stats
    mean_path = Path(config['normalization']['mean_file'])
    std_path = Path(config['normalization']['std_file'])

    if not mean_path.exists():
        raise ValueError(f"FATAL: mean_file doesn't exist: {mean_path}")
    if not std_path.exists():
        raise ValueError(f"FATAL: std_file doesn't exist: {std_path}")

    mean_spectrum = torch.load(mean_path, weights_only=False)
    std_spectrum = torch.load(std_path, weights_only=False)

    print(f"Loaded normalization stats")
    print(f"Mean shape: {mean_spectrum.shape}")
    print(f"Std shape: {std_spectrum.shape}")

    # Get parameters
    params = config['processing']
    sample_params = config['sampling']
    pca_params = config['pca']

    # Set seed for reproducibility
    np.random.seed(sample_params['seed'])
    torch.manual_seed(sample_params['seed'])

    # Find input files
    nc_files = sorted(input_dir.glob('*.nc'))
    if not nc_files:
        raise ValueError(f"No .nc files found in {input_dir}")

    # Limit files
    max_files = sample_params['max_files']
    if args.debug:
        max_files = min(3, max_files)
    nc_files = nc_files[:max_files]

    print(f"Processing {len(nc_files)} files")
    print(f"Sampling {sample_params['pixels_per_file']} pixels per file")
    print(f"Total pixels: {len(nc_files) * sample_params['pixels_per_file']}")

    # Collect samples
    all_samples = []

    for nc_path in tqdm(nc_files, desc="Collecting samples"):
        try:
            # Read radiance data
            with nc.Dataset(nc_path) as f:
                rad = torch.tensor(np.array(f[params['band']]['radiance'][...]))

            # Get dimensions
            n_mirror, n_track, n_spectral = rad.shape

            # Apply same preprocessing as tiles
            # 1. Log transform
            log_rad = torch.log(torch.clamp(rad, params['min_radiance'], torch.inf))

            # 2. Normalize with global stats
            z_rad = (log_rad - mean_spectrum) / (std_spectrum + 1e-8)

            # 3. Clip
            z_rad = torch.clamp(z_rad, params['clip_min'], params['clip_max'])

            # Sample random pixels
            n_pixels = n_mirror * n_track
            n_samples = min(sample_params['pixels_per_file'], n_pixels)

            # Flatten spatial dimensions
            z_rad_flat = z_rad.reshape(n_pixels, n_spectral)

            # Random sampling without replacement
            indices = np.random.choice(n_pixels, n_samples, replace=False)
            samples = z_rad_flat[indices]

            all_samples.append(samples.numpy())

        except Exception as e:
            print(f"Error processing {nc_path}: {e}")
            continue

    # Concatenate all samples
    X = np.vstack(all_samples)
    print(f"\nCollected samples shape: {X.shape}")

    # Fit PCA
    print(f"\nFitting PCA with {pca_params['n_components']} components...")
    pca = PCA(n_components=pca_params['n_components'])
    pca.fit(X)

    # Print explained variance
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    # Save PCA model
    pca_path = output_dir / 'pca_model.pkl'
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"\nSaved PCA model to {pca_path}")

    # Save components as torch tensors
    components_path = output_dir / 'pca_components.pt'
    torch.save({
        'components': torch.tensor(pca.components_),  # [n_components, n_features]
        'mean': torch.tensor(pca.mean_),  # [n_features]
        'explained_variance': torch.tensor(pca.explained_variance_),
        'explained_variance_ratio': torch.tensor(pca.explained_variance_ratio_),
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }, components_path)
    print(f"Saved PCA components to {components_path}")

    # Save sample projections for visualization
    X_transformed = pca.transform(X)
    projections_path = output_dir / 'sample_projections.pt'
    torch.save(torch.tensor(X_transformed), projections_path)
    print(f"Saved sample projections to {projections_path}")

    # Save summary
    summary = {
        'n_files_processed': len(nc_files),
        'pixels_per_file': sample_params['pixels_per_file'],
        'total_samples': X.shape[0],
        'n_spectral_channels': X.shape[1],
        'n_components': pca_params['n_components'],
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_variance_explained': float(pca.explained_variance_ratio_.sum())
    }

    summary_path = output_dir / 'summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f)
    print(f"Saved summary to {summary_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()