#!/usr/bin/env python3
"""
Evaluate reconstruction error for all checkpoints in an experiment directory.
"""

import argparse
import yaml
import sys
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from glob import glob

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model import get_model
from src.utils import init_directory


def compute_metrics(gt, recon, metrics_list):
    """Compute specified metrics between ground truth and reconstruction."""
    results = {}

    # Flatten tensors for metric computation
    gt_flat = gt.flatten()
    recon_flat = recon.flatten()

    for metric in metrics_list:
        if metric == 'mse':
            results['mse'] = torch.mean((gt_flat - recon_flat) ** 2).item()
        elif metric == 'mae':
            results['mae'] = torch.mean(torch.abs(gt_flat - recon_flat)).item()
        elif metric == 'psnr':
            mse = torch.mean((gt_flat - recon_flat) ** 2).item()
            # PSNR for normalized data (assuming range [-10, 10] after clipping)
            max_val = 20.0  # Range of [-10, 10]
            results['psnr'] = 10 * np.log10(max_val**2 / (mse + 1e-10))

    return results


def evaluate_checkpoint(checkpoint_path, model_params, val_data, config, device):
    """Evaluate a single checkpoint on validation data."""

    # Load model
    model = get_model(model_params, device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('global_step', 0)
    else:
        model.load_state_dict(checkpoint)
        step = 0

    model.eval()

    # Get metrics config
    metrics_list = config['evaluation']['metrics']
    batch_size = config['evaluation']['batch_size']

    # Accumulate metrics across all validation samples
    all_metrics = {metric: [] for metric in metrics_list}

    # Process validation data in batches
    for batch_start in tqdm(range(0, len(val_data), batch_size),
                            desc=f"Evaluating {checkpoint_path.name}",
                            leave=False):
        batch_end = min(batch_start + batch_size, len(val_data))
        batch = val_data[batch_start:batch_end]

        # Stack batch
        batch_tensor = torch.stack(batch).to(device)

        # Forward pass
        with torch.no_grad():
            recon_batch = model(batch_tensor)

        # Compute metrics for each sample in batch
        for i in range(len(batch)):
            gt = batch_tensor[i]
            recon = recon_batch[i]

            sample_metrics = compute_metrics(gt, recon, metrics_list)
            for metric, value in sample_metrics.items():
                all_metrics[metric].append(value)

    # Average metrics across all samples
    avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

    return avg_metrics, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to evaluation config')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')
    parser.add_argument('--debug', action='store_true', help='Process only 1 checkpoint and 2 validation samples')
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config
    if 'exp_dir' not in config:
        raise ValueError("FATAL: 'exp_dir' is required in config")
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")

    # Setup paths
    exp_dir = Path(config['exp_dir'])
    if not exp_dir.exists():
        raise ValueError(f"FATAL: Experiment directory {exp_dir} does not exist")

    # Output dir is WITHIN exp_dir
    full_output_dir = exp_dir / Path(config['output_dir']).name

    # Initialize output directory with overwrite logic
    output_dir = init_directory(str(full_output_dir), overwrite=args.overwrite)

    # Create subdirectories
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)

    # Copy config to output directory
    shutil.copy2(args.config_path, output_dir / 'config.yaml')

    print(f"Evaluating checkpoints from: {exp_dir}")
    print(f"Output directory: {output_dir}")

    # Load training config from exp_dir
    training_config_path = exp_dir / config['model']['training_config_path']
    if not training_config_path.exists():
        raise ValueError(f"FATAL: Training config not found at {training_config_path}")

    with open(training_config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    # Find all checkpoints
    checkpoint_pattern = config['model']['checkpoint_pattern']
    checkpoint_paths = sorted(glob(str(exp_dir / checkpoint_pattern)))

    if not checkpoint_paths:
        raise ValueError(f"FATAL: No checkpoints found matching {checkpoint_pattern} in {exp_dir}")

    if args.debug:
        checkpoint_paths = checkpoint_paths[:1]

    print(f"Found {len(checkpoint_paths)} checkpoints to evaluate")

    # Load validation data from tiles
    val_dir = Path(config['data']['val_dir'])
    if not val_dir.exists():
        raise ValueError(f"FATAL: Validation directory {val_dir} does not exist")

    # Get list of validation tiles
    val_tiles = sorted(list(val_dir.glob('*.pt')))

    # Limit validation samples if specified
    max_val_samples = config['data'].get('max_val_samples')
    if max_val_samples is not None:
        val_tiles = val_tiles[:max_val_samples]

    if args.debug:
        val_tiles = val_tiles[:2]

    print(f"Loading {len(val_tiles)} validation tiles...")

    # Load validation data
    val_data = []
    for tile_path in tqdm(val_tiles, desc="Loading validation tiles"):
        # Load tile - shape is [num_patches, 64, 64, 1028]
        tile_batch = torch.load(tile_path, weights_only=True)
        # Process each 64x64 patch individually
        for i in range(tile_batch.shape[0]):
            # Get single patch [64, 64, 1028] and permute to [1028, 64, 64]
            patch = tile_batch[i].permute(2, 0, 1)
            val_data.append(patch)

    # Setup device and model params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Evaluate all checkpoints
    results = []
    for checkpoint_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
        checkpoint_path = Path(checkpoint_path)
        metrics, step = evaluate_checkpoint(
            checkpoint_path, model_params, val_data, config, device
        )

        # Add checkpoint info to results
        result_entry = {
            'checkpoint': checkpoint_path.name,
            'step': step,
            **metrics
        }
        results.append(result_entry)

        print(f"{checkpoint_path.name} - Step {step}: {metrics}")

    # Save results as JSON
    results_file = output_dir / 'results' / 'reconstruction_metrics.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

    # Plot metrics if requested
    if config['plotting']['plot_metrics'] and len(results) > 1:
        # Extract steps and metrics for plotting
        steps = [r['step'] for r in results]

        # Create figure with subplots for each metric
        metrics_to_plot = config['evaluation']['metrics']
        n_metrics = len(metrics_to_plot)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_to_plot):
            values = [r[metric] for r in results]
            axes[idx].plot(steps, values, 'o-', linewidth=2, markersize=6)
            axes[idx].set_xlabel('Training Step')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(f'{metric.upper()} vs Training Step')
            axes[idx].grid(True, alpha=0.3)

            # Highlight best value
            if metric in ['mse', 'mae']:
                best_idx = np.argmin(values)
            else:  # psnr
                best_idx = np.argmax(values)

            axes[idx].plot(steps[best_idx], values[best_idx], 'r*',
                          markersize=15, label=f'Best: {values[best_idx]:.4f}')
            axes[idx].legend()

        plt.suptitle(f'Reconstruction Metrics - {exp_dir.name}')
        plt.tight_layout()

        # Save plot
        plot_file = output_dir / 'figures' / 'metrics_vs_step.png'
        plt.savefig(plot_file, dpi=config['plotting']['dpi'], bbox_inches='tight')
        plt.close()
        print(f"Saved metrics plot to {plot_file}")

        # Also create a summary plot showing best checkpoint performance
        fig, ax = plt.subplots(figsize=(8, 5))

        # Find best checkpoint for each metric and plot as bar chart
        best_results = {}
        for metric in metrics_to_plot:
            values = [r[metric] for r in results]
            if metric in ['mse', 'mae']:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            best_results[metric] = {
                'value': values[best_idx],
                'step': steps[best_idx],
                'checkpoint': results[best_idx]['checkpoint']
            }

        # Create bar plot
        metrics_names = list(best_results.keys())
        metric_values = [best_results[m]['value'] for m in metrics_names]
        metric_steps = [best_results[m]['step'] for m in metrics_names]

        bars = ax.bar(range(len(metrics_names)), metric_values)
        ax.set_xticks(range(len(metrics_names)))
        ax.set_xticklabels([m.upper() for m in metrics_names])
        ax.set_ylabel('Metric Value')
        ax.set_title('Best Checkpoint Performance by Metric')

        # Add value labels on bars
        for bar, value, step in zip(bars, metric_values, metric_steps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}\n(step {step})',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save summary plot
        summary_file = output_dir / 'figures' / 'best_metrics_summary.png'
        plt.savefig(summary_file, dpi=config['plotting']['dpi'], bbox_inches='tight')
        plt.close()
        print(f"Saved summary plot to {summary_file}")

    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()