#!/usr/bin/env python3
"""
Prepare ML-ready tiles from TEMPO radiance data.
"""

import argparse
import yaml
import sys
import numpy as np
import torch
import netCDF4 as nc
from pathlib import Path
import shutil
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import init_directory


def extract_tiles(z_rad, tile_size, n_tiles, seed=None):
    """Extract tiles with random augmentations (overlapping allowed)."""
    if seed is not None:
        np.random.seed(seed)

    n_mirror, n_track, n_spectral = z_rad.shape
    tile_mirror, tile_track = tile_size

    if n_mirror < tile_mirror or n_track < tile_track:
        return None

    tiles = []

    for _ in range(n_tiles):
        # Random position (overlapping allowed)
        i = np.random.randint(0, n_mirror - tile_mirror + 1)
        j = np.random.randint(0, n_track - tile_track + 1)

        # Extract tile
        tile = z_rad[i:i+tile_mirror, j:j+tile_track].clone()

        # Random augmentations
        # Random horizontal flip
        if np.random.rand() > 0.5:
            tile = torch.flip(tile, dims=[0])

        # Random vertical flip
        if np.random.rand() > 0.5:
            tile = torch.flip(tile, dims=[1])

        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)  # 0, 1, 2, or 3 rotations of 90 degrees
        if k > 0:
            tile = torch.rot90(tile, k, dims=[0, 1])

        tiles.append(tile)

    return torch.stack(tiles) if tiles else None


def process_file(nc_path, config, mean_spectrum=None, std_spectrum=None):
    """Process single TEMPO file into tiles."""
    params = config['processing']

    # Read radiance data
    with nc.Dataset(nc_path) as f:
        rad = torch.tensor(np.array(f[params['band']]['radiance'][...]))

    # Log transform
    log_rad = torch.log(torch.clamp(rad, params['min_radiance'], torch.inf))

    # Normalize with global or per-file statistics
    if mean_spectrum is not None and std_spectrum is not None:
        # Use global normalization
        z_rad = (log_rad - mean_spectrum) / (std_spectrum + 1e-8)
    else:
        # Fall back to per-file normalization
        mean = log_rad.mean(dim=(0, 1))
        std = log_rad.std(dim=(0, 1))
        z_rad = (log_rad - mean) / (std + 1e-8)

    # Clip
    z_rad = torch.clamp(z_rad, params['clip_min'], params['clip_max'])

    # Extract tiles
    tiles = extract_tiles(
        z_rad,
        params['tile_size'],
        params['tiles_per_file'],
        seed=None  # Random each time
    )

    return tiles


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

    # Load global normalization stats if provided
    mean_spectrum = None
    std_spectrum = None

    if 'normalization' in config:
        norm_config = config['normalization']

        if 'mean_file' in norm_config and 'std_file' in norm_config:
            # Load from files
            mean_path = Path(norm_config['mean_file'])
            std_path = Path(norm_config['std_file'])

            if not mean_path.exists():
                raise ValueError(f"FATAL: mean_file doesn't exist: {mean_path}")
            if not std_path.exists():
                raise ValueError(f"FATAL: std_file doesn't exist: {std_path}")

            mean_spectrum = torch.load(mean_path, weights_only=False)
            std_spectrum = torch.load(std_path, weights_only=False)
            print(f"Loaded global normalization from files")

        elif 'mean_spectrum' in norm_config and 'std_spectrum' in norm_config:
            # Use values from config
            mean_spectrum = torch.tensor(norm_config['mean_spectrum'])
            std_spectrum = torch.tensor(norm_config['std_spectrum'])
            print(f"Using global normalization from config")

    # Setup output
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Save config
    shutil.copy2(config_path, output_dir / 'config.yaml')

    # Find input files
    nc_files = sorted(input_dir.glob('**/*.nc'))
    if not nc_files:
        raise ValueError(f"No .nc files found in {input_dir}")

    # Limit files in debug mode
    if debug:
        nc_files = nc_files[:3]
        print(f"DEBUG: Processing only {len(nc_files)} files")

    print(f"Processing {len(nc_files)} TEMPO files")

    # Setup train/val split
    split_config = config.get('split', {})
    val_fraction = split_config.get('val_fraction', 0.2)
    seed = split_config.get('seed', 42)

    np.random.seed(seed)
    n_val = int(len(nc_files) * val_fraction)
    val_indices = set(np.random.choice(len(nc_files), n_val, replace=False))

    # Create split directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Process files
    train_count = 0
    val_count = 0
    failed = []

    # Track source files for each split
    train_sources = {}  # tile_file -> source_nc_file
    val_sources = {}    # tile_file -> source_nc_file

    for i, nc_path in enumerate(tqdm(nc_files, desc="Processing")):
        try:
            tiles = process_file(nc_path, config, mean_spectrum, std_spectrum)

            if tiles is None:
                failed.append(str(nc_path))
                continue

            # Save to appropriate split and track source
            if i in val_indices:
                tile_filename = f"{val_count:05d}.pt"
                save_path = val_dir / tile_filename
                torch.save(tiles, save_path)
                val_sources[tile_filename] = str(nc_path.relative_to(input_dir))
                val_count += 1
            else:
                tile_filename = f"{train_count:05d}.pt"
                save_path = train_dir / tile_filename
                torch.save(tiles, save_path)
                train_sources[tile_filename] = str(nc_path.relative_to(input_dir))
                train_count += 1

        except Exception as e:
            print(f"  Failed: {nc_path.name} - {e}")
            failed.append(str(nc_path))

    # Save normalization stats if global
    if mean_spectrum is not None and std_spectrum is not None:
        torch.save(mean_spectrum, output_dir / 'mean_spectrum.pt')
        torch.save(std_spectrum, output_dir / 'std_spectrum.pt')
        print(f"Saved normalization stats to output directory")

    # Save manifest
    manifest = {
        'input_files': len(nc_files),
        'train_files': train_count,
        'val_files': val_count,
        'failed': failed,
        'tile_shape': config['processing']['tile_size'] + [1028],  # [mirror, track, spectral]
        'tiles_per_file': config['processing']['tiles_per_file'],
        'global_normalization': mean_spectrum is not None
    }

    manifest_path = output_dir / 'manifest.yaml'
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f)

    # Save split information as JSON
    split_info = {
        'train_sources': train_sources,
        'val_sources': val_sources,
        'failed_files': failed,
        'split_config': {
            'val_fraction': val_fraction,
            'seed': seed,
            'n_train': train_count,
            'n_val': val_count,
            'n_failed': len(failed)
        }
    }

    split_info_path = output_dir / 'split_info.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved split information to {split_info_path}")

    print(f"\nDone:")
    print(f"  Train: {train_count} files")
    print(f"  Val: {val_count} files")
    print(f"  Failed: {len(failed)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)