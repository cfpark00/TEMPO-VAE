#!/usr/bin/env python3
"""
Train linear probes from VAE latents to component fields (NO2, O3, HCHO, CLDO4).
"""

import argparse
import yaml
import json
import shutil
import torch
import torch.nn as nn
import numpy as np
import netCDF4 as nc
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model import get_model
from src.utils import init_directory


def load_l2_file(l1_filename, l2_base_path, product_name, product_subdir):
    """Load corresponding L2 file for given L1 filename."""
    # Replace L1 with L2 product name in filename
    l2_filename = l1_filename.replace("_RAD_L1_", f"_{product_name}_L2_")
    l2_path = l2_base_path / product_subdir / 'raw' / l2_filename

    if not l2_path.exists():
        print(f"Warning: L2 file not found: {l2_path}")
        return None

    return l2_path


def extract_component_field(l2_path, field_name, scale=1.0):
    """Extract specific field from L2 product."""
    with nc.Dataset(l2_path) as f:
        if 'product' not in f.groups:
            return None

        if field_name not in f['product'].variables:
            print(f"Warning: Field {field_name} not found in {l2_path}")
            return None

        data = np.array(f['product'][field_name][...])

        # Handle fill values (typically < -1e29)
        data = np.where(data < -1e29, np.nan, data)

        # Apply scaling - ensure data is float type
        data = data.astype(np.float32) / float(scale)

    return data


def normalize_component(data, norm_type, stats=None):
    """Normalize component field data."""
    if norm_type == "zscore":
        if stats is None:
            # Compute stats
            valid_data = data[~np.isnan(data)]
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            stats = {"mean": mean, "std": std}

        normalized = (data - stats["mean"]) / (stats["std"] + 1e-8)

    elif norm_type == "minmax":
        if stats is None:
            # Compute stats
            valid_data = data[~np.isnan(data)]
            vmin = np.min(valid_data)
            vmax = np.max(valid_data)
            stats = {"min": vmin, "max": vmax}

        normalized = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)

    elif norm_type == "asinh":
        if stats is None:
            # Compute stats - use MAD (Median Absolute Deviation) for robust scale
            valid_data = data[~np.isnan(data)]
            median = np.median(valid_data)
            mad = np.median(np.abs(valid_data - median))
            scale = 1.4826 * mad  # 1.4826 makes MAD consistent with std for normal dist
            stats = {"scale": scale, "median": median}

        normalized = np.arcsinh(data / (stats["scale"] + 1e-8))

    elif norm_type == "logit":
        from scipy.special import logit
        if stats is None:
            # Use fixed epsilon for squeeze transform
            eps = 0.01
            stats = {"eps": eps}

        # Squeeze [0,1] to [eps, 1-eps] then apply logit
        eps = stats["eps"]
        squeezed = eps + (1 - 2*eps) * data
        # Handle NaN values
        squeezed = np.where(np.isnan(data), np.nan, squeezed)
        normalized = logit(squeezed)

    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

    return normalized, stats


def process_file(l1_path, l2_base_path, l2_products, model, mean_spectrum, std_spectrum,
                 components_config, device):
    """Process one L1 file and extract latents + component fields."""

    # Load L1 radiance data
    with nc.Dataset(l1_path) as f:
        rad = torch.tensor(np.array(f['band_290_490_nm']['radiance'][...]))

    # Apply same normalization as training
    log_rad = torch.log(torch.clamp(rad, 1.0, torch.inf))
    z_rad = (log_rad - mean_spectrum) / (std_spectrum + 1e-8)
    z_rad = torch.clamp(z_rad, -10, 10)

    # Get largest 64 multiple that fits
    mirror_size, track_size, n_spectral = z_rad.shape
    mirror_crop = (mirror_size // 64) * 64
    track_crop = (track_size // 64) * 64
    z_rad_crop = z_rad[:mirror_crop, :track_crop, :]

    # Convert to channel-first for model: [1, 1028, H, W]
    input_tensor = z_rad_crop.permute(2, 0, 1).unsqueeze(0).to(device)

    # Get latent representation
    with torch.no_grad():
        # Get latent distribution using the model's get_latent method
        latent_dist = model.get_latent(input_tensor)  # Returns DiagonalGaussianDistribution
        # Use the mean of the distribution as our latent representation
        latent = latent_dist.mean  # [1, 32, H//4, W//4]

    # Assert latent dimensions are as expected
    assert latent.shape[0] == 1, f"Batch dimension should be 1, got {latent.shape[0]}"
    assert latent.shape[1] == 32, f"Latent channels should be 32, got {latent.shape[1]}"
    assert latent.shape[2] == mirror_crop // 4, f"Latent height should be {mirror_crop//4}, got {latent.shape[2]}"
    assert latent.shape[3] == track_crop // 4, f"Latent width should be {track_crop//4}, got {latent.shape[3]}"

    # Load and process component fields
    component_data = {}
    component_stats = {}

    for comp_name, comp_config in components_config.items():
        # Load L2 file
        l2_path = load_l2_file(
            l1_path.name,
            l2_base_path,
            comp_name,
            l2_products[comp_name]
        )

        if l2_path is None:
            component_data[comp_name] = None
            continue

        # Extract field
        field_data = extract_component_field(
            l2_path,
            comp_config["field"],
            comp_config["scale"]
        )

        if field_data is None:
            component_data[comp_name] = None
            continue

        # Assert original component field matches L1 dimensions before cropping
        assert field_data.ndim == 2, f"Component field {comp_name} must be 2D [H, W], got shape {field_data.shape}"
        assert field_data.shape[0] >= mirror_crop, f"Component {comp_name} height {field_data.shape[0]} < L1 crop {mirror_crop}"
        assert field_data.shape[1] >= track_crop, f"Component {comp_name} width {field_data.shape[1]} < L1 crop {track_crop}"

        # Crop to same size as L1
        field_data = field_data[:mirror_crop, :track_crop]

        # Normalize
        normalized, stats = normalize_component(
            field_data,
            comp_config["norm_type"]
        )

        # Average pool by 4x to match latent resolution
        # Reshape for pooling: [H, W] -> [H//4, 4, W//4, 4]
        h, w = normalized.shape
        pooled = normalized.reshape(h//4, 4, w//4, 4)
        # Suppress warning for all-NaN blocks (expected for cloud-covered regions)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')
            warnings.filterwarnings('ignore', r'invalid value encountered')
            pooled = np.nanmean(pooled, axis=(1, 3))  # Average over 4x4 blocks

        # Assert pooled component matches latent spatial dimensions
        latent_h, latent_w = latent.shape[2], latent.shape[3]
        assert pooled.shape == (latent_h, latent_w), \
            f"Pooled component {comp_name} shape {pooled.shape} != latent spatial shape ({latent_h}, {latent_w})"

        component_data[comp_name] = pooled
        component_stats[comp_name] = stats

    return latent.cpu(), component_data, component_stats


class LinearProbe(nn.Module):
    """Simple linear probe from latent channels to component."""
    def __init__(self, input_dim=32, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """MLP probe with configurable hidden layers."""
    def __init__(self, input_dim=32, hidden_dims=[512, 512], output_dim=1,
                 dropout=0.1, activation='relu'):
        super().__init__()

        # Build layers
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Add activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Add dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def train_probe(X_train, y_train, X_val, y_val, config):
    """Train a probe (linear or MLP)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create probe based on architecture type
    architecture = config.get('architecture', 'linear')

    if architecture == 'mlp':
        probe = MLPProbe(
            input_dim=32,
            hidden_dims=config.get('hidden_dims', [512, 512]),
            output_dim=1,
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'relu')
        ).to(device)
    else:  # linear
        probe = LinearProbe(input_dim=32, output_dim=1).to(device)

    # Use AdamW with weight_decay for proper L2 regularization
    weight_decay = config.get('weight_decay', 0.01)
    optimizer = torch.optim.AdamW(probe.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    # Get batch size
    batch_size = config.get('batch_size', 512)
    n_batches = (len(X_train) + batch_size - 1) // batch_size

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0

    for epoch in range(config['max_epochs']):
        # Train with mini-batches
        probe.train()
        epoch_train_loss = 0.0

        # Shuffle data for each epoch
        perm = torch.randperm(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))

            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            optimizer.zero_grad()
            y_pred = probe(X_batch)
            batch_loss = criterion(y_pred, y_batch)
            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item() * (end_idx - start_idx)

        # Average training loss for the epoch
        train_loss = epoch_train_loss / len(X_train)

        # Validate (can still do full batch for validation)
        probe.eval()
        with torch.no_grad():
            y_pred_val = probe(X_val)
            val_loss = criterion(y_pred_val, y_val).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = probe.state_dict().copy()
            best_epoch = epoch

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Load best model
    probe.load_state_dict(best_state)
    print(f"Best model from epoch {best_epoch} with val loss {best_val_loss:.4f}")

    return probe, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to analysis config')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    parser.add_argument('--debug', action='store_true', help='Process only 1 file')
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")

    # Setup output
    output_dir = init_directory(config['output_dir'], overwrite=args.overwrite)
    shutil.copy2(args.config_path, output_dir / 'config.yaml')

    # Create subdirectories
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    (output_dir / 'data_stats').mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load split info
    tiles_path = Path(config['data']['tiles_path'])
    split_info_path = tiles_path / 'split_info.json'
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    # Get validation L1 files (for probe analysis)
    l1_nc_path = Path(config['data']['l1_nc_path']) / 'raw'
    l2_base_path = Path(config['data']['l2_base_path'])

    train_files = list(split_info['val_sources'].values())
    if args.debug:
        train_files = train_files[:3]  # Use 3 files in debug mode for better probe training

    # Load normalization stats
    mean_spectrum = torch.load(tiles_path / 'mean_spectrum.pt')
    std_spectrum = torch.load(tiles_path / 'std_spectrum.pt')

    # Load model
    with open(config['model']['training_config_path'], 'r') as f:
        train_config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model params
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
    if 'model_state_dict' in checkpoint:
        # Use strict=False to ignore L2 head weights if present (for L2 supervised models)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Process all files and collect data
    all_latents = {}  # Will store latents per component (since each can have different valid pixels)
    all_components = {comp: [] for comp in config['components'].keys()}
    component_norm_stats = {comp: None for comp in config['components'].keys()}

    print(f"Processing {len(train_files)} validation files...")
    for filename in tqdm(train_files, desc="Processing files"):
        l1_path = l1_nc_path / filename

        if not l1_path.exists():
            print(f"Warning: L1 file not found: {l1_path}")
            continue

        # Process file
        latent, components, comp_stats = process_file(
            l1_path,
            l2_base_path,
            config['data']['l2_products'],
            model,
            mean_spectrum,
            std_spectrum,
            config['components'],
            device
        )

        # latent is [1, 32, H//4, W//4]
        latent = latent.squeeze(0)  # [32, H//4, W//4]
        h_lat, w_lat = latent.shape[1], latent.shape[2]

        # Flatten latent for sampling
        latent_flat = latent.reshape(32, -1).T  # [H*W, 32]

        # Process each component
        for comp_name, comp_data in components.items():
            if comp_data is not None:
                comp_flat = comp_data.flatten()  # [H*W]

                # Find valid (non-NaN) pixels
                valid_mask = ~np.isnan(comp_flat)
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) > 0:
                    # Sample from valid pixels only
                    n_pixels = config['probe']['n_pixels_per_file']
                    n_sample = min(n_pixels, len(valid_indices))
                    sample_indices = np.random.choice(valid_indices, n_sample, replace=False)

                    # Extract sampled pixels
                    sampled_latents = latent_flat[sample_indices].numpy()
                    sampled_comp = comp_flat[sample_indices]

                    # Store
                    all_components[comp_name].append(sampled_comp)
                    if comp_name not in all_latents:
                        all_latents[comp_name] = []
                    all_latents[comp_name].append(sampled_latents)

                # Store normalization stats from first file
                if component_norm_stats[comp_name] is None and comp_name in comp_stats:
                    component_norm_stats[comp_name] = comp_stats[comp_name]

    # Save normalization stats
    with open(output_dir / 'results' / 'component_norm_stats.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        stats_for_json = {}
        for comp, stats in component_norm_stats.items():
            if stats is not None:
                stats_for_json[comp] = {k: float(v) for k, v in stats.items()}
        json.dump(stats_for_json, f, indent=2)

    # Create data distribution histograms

    # First, create raw vs normalized comparison plot (like test_all_normalizations.py)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, comp_name in enumerate(config['components'].keys()):
        if comp_name not in all_components or len(all_components[comp_name]) == 0:
            continue

        # Get the raw data for this component (need to reload one file for raw stats)
        comp_config = config['components'][comp_name]
        sample_file = train_files[0]  # Use first file for raw data sample
        l1_path = l1_nc_path / sample_file
        l2_path = load_l2_file(
            l1_path.name,
            l2_base_path,
            comp_name,
            config['data']['l2_products'][comp_name]
        )

        if l2_path and l2_path.exists():
            # Extract raw field
            raw_data = extract_component_field(
                l2_path,
                comp_config["field"],
                comp_config["scale"]
            )

            if raw_data is not None:
                # Plot raw histogram (top row)
                ax_raw = axes[0, idx]
                valid_raw = raw_data[~np.isnan(raw_data)]
                ax_raw.hist(valid_raw, bins=100, alpha=0.7,
                           color=['red', 'blue', 'green', 'purple'][idx])
                ax_raw.set_title(f'{comp_name} - Raw', fontweight='bold')
                ax_raw.set_xlabel(f'Scale: {comp_config["scale"]}')
                ax_raw.set_ylabel('Count')
                ax_raw.set_yscale('log')
                ax_raw.grid(True, alpha=0.3)

                # Add raw stats
                raw_stats_text = (f'Mean: {np.mean(valid_raw):.2f}\n'
                                f'Std: {np.std(valid_raw):.2f}\n'
                                f'Min: {np.min(valid_raw):.2f}\n'
                                f'Max: {np.max(valid_raw):.2f}')
                ax_raw.text(0.02, 0.98, raw_stats_text,
                           transform=ax_raw.transAxes, fontsize=8,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot normalized histogram (bottom row) using collected data
        y_normalized = np.concatenate(all_components[comp_name])
        ax_norm = axes[1, idx]
        ax_norm.hist(y_normalized, bins=100, alpha=0.7,
                    color=['red', 'blue', 'green', 'purple'][idx])
        ax_norm.set_title(f'{comp_name} - {comp_config["norm_type"]}', fontweight='bold')
        ax_norm.set_xlabel('Normalized value')
        ax_norm.set_ylabel('Count')
        ax_norm.set_yscale('log')
        ax_norm.grid(True, alpha=0.3)

        # Add normalized stats
        norm_stats_text = (f'Mean: {np.mean(y_normalized):.3f}\n'
                          f'Std: {np.std(y_normalized):.3f}\n'
                          f'Min: {np.min(y_normalized):.3f}\n'
                          f'Max: {np.max(y_normalized):.3f}')
        ax_norm.text(0.02, 0.98, norm_stats_text,
                    transform=ax_norm.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.suptitle('Component Fields: Raw vs Normalized Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'data_stats' / 'all_normalizations_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Next, plot the shared input latent distributions (same for all components)
    if len(all_latents) > 0:
        # Get latents from any component (they're all the same pixels)
        first_comp = next(iter(all_latents.keys()))
        X = np.concatenate(all_latents[first_comp], axis=0)  # [N_total, 32]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot latent channel distributions (show a few representative channels)
        channels_to_show = [0, 8, 16, 24, 31]  # Sample 5 channels
        for ch in channels_to_show:
            axes[0].hist(X[:, ch], bins=50, alpha=0.5, label=f'Ch {ch}', density=True)
        axes[0].set_xlabel('Latent Values')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Input Latent Distributions (sample channels)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot all latent values combined
        axes[1].hist(X.flatten(), bins=100, alpha=0.7, color='blue', density=True)
        axes[1].set_xlabel('Latent Values (all channels)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('All Input Latent Values')
        axes[1].grid(True, alpha=0.3)

        # Add statistics
        axes[1].text(0.02, 0.98, f'Mean: {np.mean(X):.3f}\nStd: {np.std(X):.3f}\nMin: {np.min(X):.3f}\nMax: {np.max(X):.3f}',
                    transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Shared Input Latent Distributions (for all regressions)')
        plt.tight_layout()
        plt.savefig(output_dir / 'data_stats' / 'input_latent_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Then plot target distributions for each component
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, comp_name in enumerate(config['components'].keys()):
        if comp_name not in all_components or len(all_components[comp_name]) == 0:
            continue

        # Concatenate target data for this component
        y = np.concatenate(all_components[comp_name])  # [N_total]

        # Plot normalized component values (targets)
        ax = axes[idx] if idx < 4 else None
        if ax is not None:
            ax.hist(y, bins=50, alpha=0.7, color=['red', 'blue', 'green', 'purple'][idx], density=True)
            ax.set_xlabel('Normalized Values')
            ax.set_ylabel('Density')
            ax.set_title(f'{comp_name} Target Distribution')
            ax.grid(True, alpha=0.3)

            # Add statistics
            ax.text(0.02, 0.98, f'Mean: {np.mean(y):.3f}\nStd: {np.std(y):.3f}\nMin: {np.min(y):.3f}\nMax: {np.max(y):.3f}\nN: {len(y)}',
                        transform=ax.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Normalized Target Distributions (post-normalization)')
    plt.tight_layout()
    plt.savefig(output_dir / 'data_stats' / 'target_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Train probes for each component
    probes = {}
    results = {}

    for comp_name in config['components'].keys():
        if comp_name not in all_latents or len(all_components[comp_name]) == 0:
            print(f"Skipping {comp_name} - no valid data")
            continue

        print(f"\nTraining probe for {comp_name}...")

        # Concatenate data for this component
        X = np.concatenate(all_latents[comp_name], axis=0)  # [N_total, 32]
        y = np.concatenate(all_components[comp_name])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['probe']['test_split'],
            random_state=config['seed']
        )

        # Train probe
        probe, train_losses, val_losses = train_probe(
            X_train, y_train, X_test, y_test,
            config['probe']
        )

        probes[comp_name] = probe

        # Evaluate
        device = next(probe.parameters()).device
        X_test_torch = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            y_pred = probe(X_test_torch).cpu().numpy().squeeze()

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results[comp_name] = {
            'r2_score': float(r2),
            'mse': float(mse),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        print(f"{comp_name}: R² = {r2:.4f}, MSE = {mse:.4f}")

        # Save probe
        torch.save(probe.state_dict(),
                  output_dir / 'models' / f'probe_{comp_name}.pt')

        # Save predictions and test data for later replotting
        np.savez(output_dir / 'results' / f'predictions_{comp_name}.npz',
                 y_test=y_test,
                 y_pred=y_pred,
                 X_test=X_test)

        # Save training curves
        np.savez(output_dir / 'results' / f'training_curves_{comp_name}.npz',
                 train_losses=np.array(train_losses),
                 val_losses=np.array(val_losses))

        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Learning curves
        epochs = np.arange(1, len(train_losses) + 1)  # Start from 1 for log scale
        axes[0].plot(epochs, train_losses, label='Train', alpha=0.7)
        axes[0].plot(epochs, val_losses, label='Validation', alpha=0.7)

        # Add vertical line at best epoch
        best_epoch = np.argmin(val_losses) + 1  # +1 because epochs start at 1
        axes[0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
                       label=f'Best @ {best_epoch}')

        axes[0].set_xlabel('Epoch (log scale)')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title(f'{comp_name} - Learning Curves')
        axes[0].legend()
        axes[0].set_xscale('log')  # Log scale for x-axis
        axes[0].set_xlim(left=10)  # Start x-axis at epoch 10
        axes[0].set_yscale('log')

        # Predictions vs Ground Truth
        n_show = min(config['visualization']['n_examples'], len(y_test))
        axes[1].scatter(y_test[:n_show], y_pred[:n_show], alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', label='Perfect prediction')
        axes[1].set_xlabel('Ground Truth')
        axes[1].set_ylabel('Predicted')
        axes[1].set_title(f'{comp_name} - R² = {r2:.4f}')
        axes[1].legend()

        # Residuals
        residuals = y_test - y_pred
        axes[2].hist(residuals, bins=50, alpha=0.7)
        axes[2].set_xlabel('Residual (True - Predicted)')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'{comp_name} - Residual Distribution')
        axes[2].axvline(0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / 'figures' / f'probe_{comp_name}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    # Save results
    with open(output_dir / 'results' / 'probe_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary plot
    if len(results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        components = list(results.keys())
        r2_scores = [results[c]['r2_score'] for c in components]

        bars = ax.bar(components, r2_scores)
        ax.set_ylabel('R² Score')
        ax.set_title('Linear Probe Performance')
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / 'figures' / 'probe_summary.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Component R² scores: {results}")


if __name__ == '__main__':
    main()