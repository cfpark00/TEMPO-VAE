#!/usr/bin/env python3
"""
Compute global mean and std from TEMPO files for normalization.
"""

import argparse
import yaml
import sys
import numpy as np
import torch
import netCDF4 as nc
from pathlib import Path
from tqdm import tqdm
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import init_directory


def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required")
    if 'input_dir' not in config:
        raise ValueError("FATAL: 'input_dir' required")

    input_dir = Path(config['input_dir'])
    if not input_dir.exists():
        raise ValueError(f"FATAL: input_dir doesn't exist: {input_dir}")

    # Setup output
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Save config
    shutil.copy2(config_path, output_dir / 'config.yaml')

    # Find input files
    nc_files = sorted(input_dir.glob('**/*.nc'))
    if not nc_files:
        raise ValueError(f"No .nc files found in {input_dir}")

    # Limit files
    max_files = config.get('max_files', 10)
    if debug:
        max_files = min(3, max_files)

    n_files = min(max_files, len(nc_files))
    print(f"Computing statistics from {n_files} files (of {len(nc_files)} total)")

    # Processing params
    band = config.get('band', 'band_290_490_nm')
    min_radiance = config.get('min_radiance', 1.0)

    # Collect all log-radiance values
    all_log_rad = []

    for nc_path in tqdm(nc_files[:n_files], desc="Loading files"):
        try:
            with nc.Dataset(nc_path) as f:
                rad = np.array(f[band]['radiance'][...])

            # Log transform
            log_rad = np.log(np.clip(rad, min_radiance, None))

            # Flatten spatial dims, keep spectral
            log_rad_flat = log_rad.reshape(-1, log_rad.shape[-1])  # [n_pixels, n_spectral]
            all_log_rad.append(log_rad_flat)

        except Exception as e:
            print(f"  Error with {nc_path.name}: {e}")
            continue

    if not all_log_rad:
        raise ValueError("FATAL: No files could be loaded")

    # Concatenate all pixels
    all_log_rad = np.vstack(all_log_rad)
    print(f"Total pixels: {len(all_log_rad):,}")

    # Compute per-channel statistics
    mean_spectrum = all_log_rad.mean(axis=0).astype(np.float32)  # [1028]
    std_spectrum = all_log_rad.std(axis=0).astype(np.float32)    # [1028]

    print(f"\nStatistics computed:")
    print(f"  Mean shape: {mean_spectrum.shape}")
    print(f"  Std shape: {std_spectrum.shape}")
    print(f"  Mean range: [{mean_spectrum.min():.3f}, {mean_spectrum.max():.3f}]")
    print(f"  Std range: [{std_spectrum.min():.3f}, {std_spectrum.max():.3f}]")

    # Save as .pt files
    mean_path = output_dir / 'tempo_mean_spectrum.pt'
    std_path = output_dir / 'tempo_std_spectrum.pt'

    torch.save(torch.from_numpy(mean_spectrum).float(), mean_path)
    torch.save(torch.from_numpy(std_spectrum).float(), std_path)

    print(f"\nSaved to:")
    print(f"  {mean_path}")
    print(f"  {std_path}")

    # Save as text for inspection
    np.savetxt(output_dir / 'tempo_mean_spectrum.txt', mean_spectrum)
    np.savetxt(output_dir / 'tempo_std_spectrum.txt', std_spectrum)

    # Save manifest
    manifest = {
        'n_files_used': n_files,
        'total_pixels': len(all_log_rad),
        'band': band,
        'min_radiance': min_radiance,
        'mean_range': [float(mean_spectrum.min()), float(mean_spectrum.max())],
        'std_range': [float(std_spectrum.min()), float(std_spectrum.max())]
    }

    manifest_path = output_dir / 'manifest.yaml'
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f)

    print(f"\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)