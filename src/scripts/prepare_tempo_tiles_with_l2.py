#!/usr/bin/env python3
"""
Prepare ML-ready tiles from TEMPO radiance data WITH L2 components.
Saves spectral tiles alongside matched NO2, O3TOT, HCHO, CLDO4 tiles.
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


def extract_tiles_with_positions(z_rad, tile_size, n_tiles, seed=None):
    """Extract tiles and return their positions for L2 matching."""
    if seed is not None:
        np.random.seed(seed)

    n_mirror, n_track, n_spectral = z_rad.shape
    tile_mirror, tile_track = tile_size

    if n_mirror < tile_mirror or n_track < tile_track:
        return None, None

    tiles = []
    positions = []  # Store (i, j, flip_h, flip_v, rotation) for each tile

    for _ in range(n_tiles):
        # Random position
        i = np.random.randint(0, n_mirror - tile_mirror + 1)
        j = np.random.randint(0, n_track - tile_track + 1)

        # Extract tile
        tile = z_rad[i:i+tile_mirror, j:j+tile_track].clone()

        # Random augmentations
        flip_h = np.random.rand() > 0.5
        flip_v = np.random.rand() > 0.5
        rotation = np.random.randint(0, 4)

        if flip_h:
            tile = torch.flip(tile, dims=[0])
        if flip_v:
            tile = torch.flip(tile, dims=[1])
        if rotation > 0:
            tile = torch.rot90(tile, rotation, dims=[0, 1])

        tiles.append(tile)
        positions.append({
            'i': i,
            'j': j,
            'flip_h': flip_h,
            'flip_v': flip_v,
            'rotation': rotation
        })

    return torch.stack(tiles) if tiles else None, positions


def apply_augmentation_to_l2(tile, flip_h, flip_v, rotation):
    """Apply same augmentation to L2 component tile."""
    if flip_h:
        tile = torch.flip(tile, dims=[0])
    if flip_v:
        tile = torch.flip(tile, dims=[1])
    if rotation > 0:
        tile = torch.rot90(tile, rotation, dims=[0, 1])
    return tile


def load_l2_component(l1_filename, l2_config, component_name):
    """Load L2 component corresponding to L1 file."""
    # Simply replace L1 with L2 product name in filename
    product_name = l2_config['products'][component_name]
    l2_filename = l1_filename.replace("_RAD_L1_", f"_{product_name}_L2_")

    # Build L2 path
    l2_base_path = Path(l2_config['base_path'])
    product_subdir = l2_config['subdirs'][component_name]
    l2_path = l2_base_path / product_subdir / 'raw' / l2_filename

    if not l2_path.exists():
        return None

    # Read component field
    field_name = l2_config['fields'][component_name]

    try:
        with nc.Dataset(l2_path) as f:
            # Check for product group (L2 structure)
            if 'product' not in f.groups:
                return None

            if field_name not in f['product'].variables:
                return None

            data = np.array(f['product'][field_name][...])

            # Handle fill values (typically < -1e29)
            data = np.where(data < -1e29, np.nan, data)

            # Apply scaling - ensure data is float type
            scale = l2_config['scales'].get(component_name, 1.0)
            data = data.astype(np.float32) / float(scale)

            return torch.tensor(data, dtype=torch.float32)
    except Exception:
        return None


def normalize_l2_component(data, norm_type, stats=None):
    """Normalize L2 component data."""
    if norm_type == "zscore":
        if stats is None:
            valid_data = data[~torch.isnan(data)]
            if len(valid_data) == 0:
                return data, None
            mean = torch.mean(valid_data)
            std = torch.std(valid_data)
            stats = {"mean": mean.item(), "std": std.item()}
        normalized = (data - stats["mean"]) / (stats["std"] + 1e-8)

    elif norm_type == "minmax":
        if stats is None:
            valid_data = data[~torch.isnan(data)]
            if len(valid_data) == 0:
                return data, None
            vmin = torch.min(valid_data)
            vmax = torch.max(valid_data)
            stats = {"min": vmin.item(), "max": vmax.item()}
        normalized = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)

    elif norm_type == "asinh":
        if stats is None:
            valid_data = data[~torch.isnan(data)]
            if len(valid_data) == 0:
                return data, None
            median = torch.median(valid_data)
            mad = torch.median(torch.abs(valid_data - median))
            scale = 1.4826 * mad
            stats = {"scale": scale.item(), "median": median.item()}
        # Match linear probe: just divide by scale, no median subtraction
        normalized = torch.asinh(data / (stats["scale"] + 1e-8))

    elif norm_type == "logit":
        # For values in [0,1] like cloud fraction
        if stats is None:
            eps = 0.01
            stats = {"eps": eps}
        # Squeeze [0,1] to [eps, 1-eps] then apply logit
        eps = stats["eps"]
        squeezed = eps + (1 - 2*eps) * data
        # Handle NaN values
        squeezed = torch.where(torch.isnan(data), torch.nan, squeezed)
        normalized = torch.log(squeezed / (1 - squeezed))

    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

    return normalized, stats


def process_file_with_l2(nc_path, config, mean_spectrum=None, std_spectrum=None, l2_stats=None):
    """Process single TEMPO file into tiles with L2 components."""
    params = config['processing']

    # Read radiance data
    with nc.Dataset(nc_path) as f:
        rad = torch.tensor(np.array(f[params['band']]['radiance'][...]))

    # Get spatial dimensions
    n_mirror, n_track, _ = rad.shape

    # Log transform radiance
    log_rad = torch.log(torch.clamp(rad, params['min_radiance'], torch.inf))

    # Normalize with global or per-file statistics
    if mean_spectrum is not None and std_spectrum is not None:
        z_rad = (log_rad - mean_spectrum) / (std_spectrum + 1e-8)
    else:
        mean = log_rad.mean(dim=(0, 1))
        std = log_rad.std(dim=(0, 1))
        z_rad = (log_rad - mean) / (std + 1e-8)

    # Clip
    z_rad = torch.clamp(z_rad, params['clip_min'], params['clip_max'])

    # Load L2 components if configured
    l2_data = {}
    if 'l2' in config:
        for component in config['l2']['components']:
            l2_field = load_l2_component(nc_path.name, config['l2'], component)

            if l2_field is None:
                # FAIL IMMEDIATELY - NO FALLBACKS
                product_name = config['l2']['products'][component]
                l2_filename = nc_path.name.replace("_RAD_L1_", f"_{product_name}_L2_")
                l2_path = Path(config['l2']['base_path']) / config['l2']['subdirs'][component] / 'raw' / l2_filename
                raise ValueError(
                    f"FATAL: Failed to load L2 component {component} for {nc_path.name}\n"
                    f"Expected L2 file: {l2_path}\n"
                    f"L2 file exists: {l2_path.exists()}\n"
                    "NO SILENT FAILURES - FAIL IMMEDIATELY!"
                )

            # Check dimensions match
            if l2_field.shape[0] < n_mirror or l2_field.shape[1] < n_track:
                raise ValueError(
                    f"FATAL: {component} dimensions too small: {l2_field.shape} < L1 {n_mirror}x{n_track}"
                )

            # Crop to match L1 dimensions
            l2_field = l2_field[:n_mirror, :n_track]

            # Normalize
            norm_type = config['l2']['norm_types'].get(component, 'zscore')
            component_stats = l2_stats.get(component) if l2_stats else None
            l2_normalized, _ = normalize_l2_component(l2_field, norm_type, component_stats)

            l2_data[component] = l2_normalized

    # Extract tiles with positions
    tiles, positions = extract_tiles_with_positions(
        z_rad,
        params['tile_size'],
        params['tiles_per_file'],
        seed=None  # Random each time
    )

    if tiles is None:
        return None

    # Extract matching L2 tiles
    l2_tiles = {comp: [] for comp in l2_data.keys()}

    for pos in positions:
        i, j = pos['i'], pos['j']
        tile_mirror, tile_track = params['tile_size']

        for component, l2_field in l2_data.items():
            # Extract same region
            l2_tile = l2_field[i:i+tile_mirror, j:j+tile_track].clone()

            # Apply same augmentation
            l2_tile = apply_augmentation_to_l2(
                l2_tile,
                pos['flip_h'],
                pos['flip_v'],
                pos['rotation']
            )

            l2_tiles[component].append(l2_tile)

    # Stack L2 tiles
    for component in l2_tiles:
        if len(l2_tiles[component]) > 0:
            l2_tiles[component] = torch.stack(l2_tiles[component])
        else:
            # This should never happen since we fail if L2 is missing
            raise ValueError(f"FATAL: No L2 tiles extracted for {component}")

    # Return spectral tiles and L2 tiles
    return {
        'spectral': tiles,
        **{f'l2_{comp}': l2_tiles[comp] for comp in l2_tiles}
    }


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
            mean_path = Path(norm_config['mean_file'])
            std_path = Path(norm_config['std_file'])

            if mean_path.exists() and std_path.exists():
                mean_spectrum = torch.load(mean_path, weights_only=False)
                std_spectrum = torch.load(std_path, weights_only=False)
                print(f"Loaded global normalization from files")

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

    # Create L2 subdirectories if needed
    l2_components = config.get('l2', {}).get('components', [])
    for component in l2_components:
        (train_dir / f'l2_{component}').mkdir(exist_ok=True)
        (val_dir / f'l2_{component}').mkdir(exist_ok=True)

    # Process files
    train_count = 0
    val_count = 0
    failed = []

    # Track source files and L2 availability
    train_sources = {}
    val_sources = {}
    l2_availability = {comp: {'train': 0, 'val': 0} for comp in l2_components}

    # First pass: compute L2 stats if needed
    l2_stats = {}
    if l2_components and config.get('l2', {}).get('compute_global_stats', False):
        print("Computing global L2 statistics...")
        for component in l2_components:
            all_values = []
            for nc_path in tqdm(nc_files[:20], desc=f"Stats for {component}"):  # Sample first 20 files
                l2_field = load_l2_component(nc_path.name, config['l2'], component)
                if l2_field is not None:
                    valid = l2_field[~torch.isnan(l2_field)]
                    if len(valid) > 0:
                        all_values.append(valid)

            if all_values:
                all_values = torch.cat(all_values)
                norm_type = config['l2']['norm_types'].get(component, 'zscore')
                _, stats = normalize_l2_component(all_values, norm_type)
                l2_stats[component] = stats
                print(f"  {component}: {stats}")

    # Main processing
    for i, nc_path in enumerate(tqdm(nc_files, desc="Processing")):
        # NO TRY/EXCEPT - FAIL IMMEDIATELY ON ANY ERROR
        result = process_file_with_l2(nc_path, config, mean_spectrum, std_spectrum, l2_stats)

        if result is None or 'spectral' not in result:
            raise ValueError(f"FATAL: Failed to process {nc_path} - no spectral data returned")

        # Determine split
        if i in val_indices:
            tile_filename = f"{val_count:05d}.pt"
            save_dir = val_dir
            sources_dict = val_sources
            split_name = 'val'
            val_count += 1
        else:
            tile_filename = f"{train_count:05d}.pt"
            save_dir = train_dir
            sources_dict = train_sources
            split_name = 'train'
            train_count += 1

        # Save spectral tiles
        torch.save(result['spectral'], save_dir / tile_filename)
        sources_dict[tile_filename] = str(nc_path.relative_to(input_dir))

        # Save L2 tiles
        for component in l2_components:
            l2_key = f'l2_{component}'
            if l2_key in result:
                torch.save(result[l2_key], save_dir / f'l2_{component}' / tile_filename)
                l2_availability[component][split_name] += 1
            else:
                raise ValueError(f"FATAL: L2 component {component} missing from result for {nc_path}")

    # Save normalization stats
    if mean_spectrum is not None and std_spectrum is not None:
        torch.save(mean_spectrum, output_dir / 'mean_spectrum.pt')
        torch.save(std_spectrum, output_dir / 'std_spectrum.pt')

    if l2_stats:
        torch.save(l2_stats, output_dir / 'l2_stats.pt')
        print(f"Saved L2 normalization stats")

    # Save manifest
    manifest = {
        'input_files': len(nc_files),
        'train_files': train_count,
        'val_files': val_count,
        'failed': failed,
        'tile_shape': config['processing']['tile_size'] + [1028],
        'tiles_per_file': config['processing']['tiles_per_file'],
        'global_normalization': mean_spectrum is not None,
        'l2_components': l2_components,
        'l2_availability': l2_availability
    }

    with open(output_dir / 'manifest.yaml', 'w') as f:
        yaml.dump(manifest, f)

    # Save split information
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
        },
        'l2_availability': l2_availability
    }

    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\nDone:")
    print(f"  Train: {train_count} files")
    print(f"  Val: {val_count} files")
    print(f"  Failed: {len(failed)} files")

    if l2_components:
        print(f"\nL2 component availability:")
        total_l2_found = 0
        for comp in l2_components:
            train_found = l2_availability[comp]['train']
            val_found = l2_availability[comp]['val']
            total_found = train_found + val_found
            total_l2_found += total_found
            print(f"  {comp}: train={train_found}, val={val_found}")

        # FAIL LOUDLY if no L2 data was found
        if total_l2_found == 0:
            raise ValueError(
                "FATAL: No L2 data was found for ANY component! "
                "Check L2 paths in config:\n"
                f"  base_path: {config['l2']['base_path']}\n"
                f"  subdirs: {config['l2']['subdirs']}\n"
                "The script found 0 L2 files but continued silently - this is a research integrity violation!"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)