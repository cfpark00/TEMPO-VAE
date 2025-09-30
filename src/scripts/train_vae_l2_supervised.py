#!/usr/bin/env python3
"""
Train VAE with multi-task L2 product supervision on TEMPO spectral tiles.
Follows exact same structure as train_vae.py
"""

import argparse
import yaml
import sys
from pathlib import Path
import torch
import shutil
from datetime import datetime

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import init_directory
from src.model import get_model
from src.tempo_data_with_l2 import TEMPODataLoaderWithL2
from src.model_with_l2 import VAEWithL2Supervision
from src.train_utils import seed_all, get_device
from tqdm import tqdm
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def validate_config(config):
    """Validate configuration."""
    # Required fields
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")

    if 'data' not in config:
        raise ValueError("FATAL: 'data' section is required in config")

    if 'base_dir' not in config['data']:
        raise ValueError("FATAL: 'data.base_dir' is required in config")

    if 'model' not in config:
        raise ValueError("FATAL: 'model' section is required in config")

    if 'training' not in config:
        raise ValueError("FATAL: 'training' section is required in config")

    if 'l2_supervision' not in config:
        raise ValueError("FATAL: 'l2_supervision' section is required in config")

    # Check data directories exist
    base_dir = Path(config['data']['base_dir'])
    if not base_dir.exists():
        raise ValueError(f"FATAL: Data directory doesn't exist: {base_dir}")

    train_dir = base_dir / 'train'
    if not train_dir.exists():
        raise ValueError(f"FATAL: Training directory doesn't exist: {train_dir}")

    val_dir = base_dir / 'val'
    if not val_dir.exists():
        raise ValueError(f"FATAL: Validation directory doesn't exist: {val_dir}")


class L2SupervisedTrainer:
    """Trainer for VAE with L2 supervision - matches Trainer class interface."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: Path,
        save_every: int = 1000,
        val_every: int = 100,
        log_every: int = 10,
        plot_every: int = 50,
        l2_weights: dict = None
    ):
        """Initialize trainer matching original Trainer interface."""
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.save_every = save_every
        self.val_every = val_every
        self.log_every = log_every
        self.plot_every = plot_every
        self.l2_weights = l2_weights or {'NO2': 0.1, 'O3TOT': 0.1, 'HCHO': 0.1, 'CLDO4': 0.1}

        # Create directories
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.summary_dir = self.output_dir / 'summary'
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage (same as original)
        self.train_metrics = []
        self.val_metrics = []
        self.step = 0

        # For plotting (same as original)
        self.plot_steps = []
        self.plot_losses = []
        self.plot_nll = []
        self.plot_kl = []
        self.plot_val_losses = []
        self.plot_pixel_mse = []
        self.plot_l2_losses = {'NO2': [], 'O3TOT': [], 'HCHO': [], 'CLDO4': []}

    def save_checkpoint(self, step=None):
        """Save model checkpoint - same as original."""
        if step is None:
            step = self.step

        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }

        ckpt_path = self.ckpt_dir / f'ckpt_step={step:06d}.pt'
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str):
        """Load model checkpoint - same as original."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        print(f"Loaded checkpoint from step {self.step}")

    def train_step(self, batch):
        """Single training step - adapted for L2 supervision."""
        self.model.train()

        # Debug: print data statistics on first step
        if self.step == 0:
            spectral = batch['spectral']
            print(f"Batch stats - min: {spectral.min():.3f}, max: {spectral.max():.3f}, "
                  f"mean: {spectral.mean():.3f}, std: {spectral.std():.3f}")

        # Forward pass with L2 supervision
        loss, metrics = self.model.compute_loss(
            batch,
            kl_weight=self.model.vae.kl_weight,
            l2_weights=self.l2_weights
        )

        # Also compute pixel MSE for tracking (same as original)
        with torch.no_grad():
            output = self.model(batch['spectral'])
            pixel_mse = torch.mean((batch['spectral'] - output['reconstruction']) ** 2).item()
            metrics['pixel_mse'] = pixel_mse

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (same as original)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Convert metrics to float
        metrics = {k: v if isinstance(v, float) else v for k, v in metrics.items()}

        return metrics

    def validate(self, val_loader, n_batches=10):
        """Validate model - same as original."""
        self.model.eval()

        val_metrics = {}
        n_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss, metrics = self.model.compute_loss(
                    batch,
                    kl_weight=self.model.vae.kl_weight,
                    l2_weights=self.l2_weights
                )

                # Accumulate metrics
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v * batch['spectral'].shape[0]
                n_samples += batch['spectral'].shape[0]

        # Average metrics
        val_metrics = {f'val_{k}': v / n_samples for k, v in val_metrics.items()}

        return val_metrics

    def train(self, train_loader, val_loader=None, n_steps=10000):
        """Train model - same structure as original."""
        pbar = tqdm(total=n_steps, desc="Training")
        train_iter = iter(train_loader)

        running_metrics = {}

        while self.step < n_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Training step
            metrics = self.train_step(batch)
            self.step += 1

            # Update running average (same as original)
            alpha = 0.99 if running_metrics else 0.0
            for k, v in metrics.items():
                running_metrics[k] = alpha * running_metrics.get(k, 0) + (1 - alpha) * v

            # Log metrics
            if self.step % self.log_every == 0:
                self.train_metrics.append({
                    'step': self.step,
                    **running_metrics
                })

                # Store for plotting (matching original metric names)
                self.plot_steps.append(self.step)
                self.plot_losses.append(running_metrics.get('loss', 0))
                self.plot_nll.append(running_metrics.get('nll_loss', 0))
                self.plot_kl.append(running_metrics.get('kl_loss', 0))
                self.plot_pixel_mse.append(running_metrics.get('pixel_mse', 0))

                # Store L2 losses
                for product in ['NO2', 'O3TOT', 'HCHO', 'CLDO4']:
                    key = f'{product}_loss'
                    if key in running_metrics:
                        self.plot_l2_losses[product].append(running_metrics[key])

                # Update progress bar
                pbar.set_postfix(running_metrics)

            # Update plots
            if self.step % self.plot_every == 0 and self.step > 0:
                self.update_plots()

            # Validation
            if val_loader is not None and self.step % self.val_every == 0:
                val_metrics = self.validate(val_loader)
                self.val_metrics.append({
                    'step': self.step,
                    **val_metrics
                })

                # Log validation metrics
                tqdm.write(f"Step {self.step}: " +
                          ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

            # Save checkpoint
            if self.step % self.save_every == 0:
                ckpt_path = self.save_checkpoint()
                tqdm.write(f"Saved checkpoint: {ckpt_path}")

                # Save reconstructions when saving checkpoint
                self.save_reconstructions(batch, self.step)
                tqdm.write(f"Saved reconstructions at step {self.step}")

            pbar.update(1)

        pbar.close()

        # Final checkpoint
        ckpt_path = self.save_checkpoint()
        print(f"Training complete. Final checkpoint: {ckpt_path}")

        # Save metrics
        self.save_metrics()

    def save_metrics(self):
        """Save training and validation metrics - same as original."""
        metrics_path = self.output_dir / 'metrics.json'

        metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {metrics_path}")

    def save_reconstructions(self, batch, step):
        """Save reconstruction comparison images with L2 predictions."""
        self.model.eval()

        with torch.no_grad():
            # Get first 8 samples
            spectral = batch['spectral'][:8]
            output = self.model(spectral)

            # Calculate reconstruction MSE
            mse = torch.mean((spectral - output['reconstruction']) ** 2, dim=(1, 2, 3))

            # Plot comparisons (extended to show L2 predictions)
            fig, axes = plt.subplots(4, 8, figsize=(24, 12))

            for i in range(min(4, spectral.shape[0])):
                # Original - take 3 channels for RGB visualization
                orig = spectral[i, [100, 500, 900], :, :].cpu().numpy()
                orig = np.transpose(orig, (1, 2, 0))
                orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)

                # Reconstruction
                rec = output['reconstruction'][i, [100, 500, 900], :, :].cpu().numpy()
                rec = np.transpose(rec, (1, 2, 0))
                rec = (rec - rec.min()) / (rec.max() - rec.min() + 1e-8)

                # Difference
                diff = np.abs(orig - rec)

                # Plot original
                axes[i, 0].imshow(orig)
                axes[i, 0].set_title(f'Original {i}')
                axes[i, 0].axis('off')

                # Plot reconstruction
                axes[i, 1].imshow(rec)
                axes[i, 1].set_title(f'Recon {i}')
                axes[i, 1].axis('off')

                # Plot difference
                axes[i, 2].imshow(diff, cmap='hot')
                axes[i, 2].set_title(f'|Diff| (MSE={mse[i]:.4f})')
                axes[i, 2].axis('off')

                # Plot spectral slice comparison
                mid_y = spectral.shape[2] // 2
                mid_x = spectral.shape[3] // 2
                orig_spectrum = spectral[i, :, mid_y, mid_x].cpu().numpy()
                rec_spectrum = output['reconstruction'][i, :, mid_y, mid_x].cpu().numpy()

                axes[i, 3].plot(orig_spectrum, 'b-', alpha=0.7, label='Original')
                axes[i, 3].plot(rec_spectrum, 'r-', alpha=0.7, label='Recon')
                axes[i, 3].set_xlabel('Channel')
                axes[i, 3].set_ylabel('Value')
                axes[i, 3].set_title(f'Spectrum')
                axes[i, 3].legend()
                axes[i, 3].grid(True, alpha=0.3)

                # Plot L2 predictions vs targets
                for j, product in enumerate(['NO2', 'O3TOT', 'HCHO', 'CLDO4']):
                    col_idx = j + 4

                    # Prediction (upsampled for visualization)
                    pred = output['l2_predictions'][product][i, 0].cpu()
                    pred_up = torch.nn.functional.interpolate(
                        pred.unsqueeze(0).unsqueeze(0),
                        size=(64, 64),
                        mode='nearest'
                    )[0, 0].numpy()

                    # Target
                    target = batch[product][i].cpu().numpy()

                    # Stack for comparison
                    vmin = np.nanmin(target)
                    vmax = np.nanmax(target)

                    axes[i, col_idx].imshow(pred_up, cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[i, col_idx].set_title(f'{product} Pred')
                    axes[i, col_idx].axis('off')

            plt.suptitle(f'Reconstructions at Step {step}')
            plt.tight_layout()

            # Save to figures directory
            figures_dir = self.output_dir / 'figures'
            figures_dir.mkdir(exist_ok=True)
            plt.savefig(figures_dir / f'reconstructions_step_{step:06d}.png', dpi=100, bbox_inches='tight')
            plt.close()

        self.model.train()

    def update_plots(self):
        """Update live plots - extended version of original with L2 losses."""
        if len(self.plot_steps) < 2:
            return

        # Determine if we should use log scale (same as original)
        max_step = max(self.plot_steps)
        use_log = max_step >= 100

        if use_log:
            # Filter for steps >= 100 for log-log plot
            idx_start = next((i for i, s in enumerate(self.plot_steps) if s >= 100), 0)
            steps = self.plot_steps[idx_start:]
            losses = self.plot_losses[idx_start:]
            nll = self.plot_nll[idx_start:]
            kl = self.plot_kl[idx_start:]
            pixel_mse = self.plot_pixel_mse[idx_start:]
            l2_losses = {k: v[idx_start:] for k, v in self.plot_l2_losses.items()}
        else:
            steps = self.plot_steps
            losses = self.plot_losses
            nll = self.plot_nll
            kl = self.plot_kl
            pixel_mse = self.plot_pixel_mse
            l2_losses = self.plot_l2_losses

        if len(steps) < 2:
            return

        # Create figure with 3 subplots (extended from original 2)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Total loss and components (same as original)
        ax1.plot(steps, losses, 'b-', label='Total Loss', linewidth=2)
        ax1.plot(steps, nll, 'g-', label='Recon Loss', alpha=0.7)
        ax1.plot(steps, kl, 'r-', label='KL Loss', alpha=0.7)

        if use_log:
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlim(left=100)
            ax1.set_title('Training Loss (log-log scale)')
        else:
            ax1.set_title('Training Loss')

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Pixel MSE (same as original)
        ax2.plot(steps, pixel_mse, 'purple', label='Pixel MSE', linewidth=2)

        if use_log:
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_xlim(left=100)
            ax2.set_title('Pixel MSE (log-log scale)')
        else:
            ax2.set_title('Pixel MSE')

        ax2.set_xlabel('Step')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: L2 losses (new)
        colors = {'NO2': 'blue', 'O3TOT': 'green', 'HCHO': 'orange', 'CLDO4': 'red'}
        for product, color in colors.items():
            if product in l2_losses and len(l2_losses[product]) > 0:
                ax3.plot(steps, l2_losses[product], color=color, label=f'{product} Loss', alpha=0.7)

        if use_log:
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            ax3.set_xlim(left=100)
            ax3.set_title('L2 Supervision Losses (log-log scale)')
        else:
            ax3.set_title('L2 Supervision Losses')

        ax3.set_xlabel('Step')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f'Training Progress - Step {self.step}')
        plt.tight_layout()

        # Save plot
        plot_path = self.summary_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()


def main(config_path, overwrite=False, debug=False):
    """Main training function - matches train_vae.py structure exactly."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate
    validate_config(config)

    # Setup output directory
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Create subdirectories (same as original)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy2(config_path, output_dir / 'config.yaml')

    # Set seed
    seed = config.get('seed', 42)
    seed_all(seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Debug mode adjustments (same as original)
    if debug:
        print("DEBUG MODE: Reduced training steps and data")
        config['training']['n_steps'] = min(200, config['training'].get('n_steps', 10000))
        config['training']['save_every'] = 50
        config['training']['val_every'] = 25
        config['training']['plot_every'] = 20

    # Create data loaders with L2 (matching original's interface exactly)
    print("\nLoading training data...")
    train_loader = TEMPODataLoaderWithL2.get_dataloader(
        data_dir=config['data']['base_dir'],
        split='train',
        batch_size=config['data'].get('batch_size', 16),
        num_workers=config['data'].get('num_workers', 4),
        min_buffer_size=config['data'].get('min_buffer_size', 200),
        verbose=True
    )

    print("\nLoading validation data...")
    val_loader = TEMPODataLoaderWithL2.get_dataloader(
        data_dir=config['data']['base_dir'],
        split='val',
        batch_size=config['data'].get('batch_size', 16),
        num_workers=config['data'].get('val_num_workers', 1),
        min_buffer_size=config['data'].get('val_min_buffer_size', 100),
        verbose=True
    )

    # Create model (using get_model like original)
    print("\nInitializing model...")

    # Build model params exactly like original
    model_params = {
        "architecture_type": "vae",
        "architecture_params": {
            "enc_dec_params": config.get('model', {})
        },
        "optimizer_type": "AdamW",
        "optimizer_params": {
            "lr": config.get('optimizer', {}).get('lr', 0.0001),
            "weight_decay": config.get('optimizer', {}).get('weight_decay', 0.05),
            "betas": config.get('optimizer', {}).get('betas', [0.9, 0.95])
        }
    }

    # Get base VAE model
    base_model = get_model(model_params, device)

    # Wrap with L2 supervision
    model = VAEWithL2Supervision(
        base_vae=base_model.vae,
        latent_channels=config['model']['z_channels'],
        mlp_hidden=config['l2_supervision']['mlp_hidden']
    ).to(device)

    # Create NEW optimizer for the full model (base + L2 heads)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        betas=config['optimizer'].get('betas', [0.9, 0.95]),
        weight_decay=config['optimizer'].get('weight_decay', 0.05)
    )

    # Count parameters (same as original)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create trainer
    train_config = config['training']
    trainer = L2SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        save_every=train_config.get('save_every', 1000),
        val_every=train_config.get('val_every', 100),
        log_every=train_config.get('log_every', 10),
        plot_every=train_config.get('plot_every', 50),
        l2_weights=config['l2_supervision']['weights']
    )

    # Load checkpoint if resuming (same as original)
    if 'resume_from' in train_config:
        ckpt_path = train_config['resume_from']
        print(f"\nResuming from checkpoint: {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)

    # Training (same as original)
    print(f"\nStarting training for {train_config['n_steps']} steps...")
    print(f"Output directory: {output_dir}")

    start_time = datetime.now()
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_steps=train_config['n_steps']
    )
    end_time = datetime.now()

    # Log training time (same as original)
    duration = end_time - start_time
    print(f"\nTraining completed in {duration}")

    # Save final info (same as original)
    info = {
        'seed': seed,
        'device': str(device),
        'n_params': n_params,
        'training_time': str(duration),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
    }

    info_path = output_dir / 'training_info.yaml'
    with open(info_path, 'w') as f:
        yaml.dump(info, f)

    print(f"Training info saved to {info_path}")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE with L2 supervision on TEMPO tiles")
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output directory')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with reduced steps')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)