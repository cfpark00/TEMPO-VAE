"""
Training utilities for VAE.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def seed_all(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Get available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        # Get GPU with most free memory
        free_memory = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            free_mem = torch.cuda.mem_get_info()[0]
            free_memory.append((i, free_mem))

        if free_memory:
            best_gpu = max(free_memory, key=lambda x: x[1])[0]
            return torch.device(f'cuda:{best_gpu}')

    return torch.device('cpu')


def get_sqrt_schedule(n_steps: int, n_saves: int = 100) -> List[int]:
    """
    Generate save steps on sqrt schedule for better coverage.

    Args:
        n_steps: Total number of training steps
        n_saves: Number of checkpoints to save

    Returns:
        List of step numbers to save at
    """
    # Generate sqrt-spaced points between 0 and 1
    sqrt_points = np.sqrt(np.linspace(0, 1, n_saves))

    # Convert to step numbers
    save_steps = (sqrt_points * n_steps).astype(int)

    # Remove duplicates and ensure we save at the end
    save_steps = sorted(list(set(save_steps)))
    if n_steps not in save_steps:
        save_steps.append(n_steps)

    return save_steps


class Trainer:
    """Simple trainer for VAE."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: Path,
        save_every: int = 1000,
        val_every: int = 100,
        log_every: int = 10,
        plot_every: int = 50
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            device: Device to train on
            output_dir: Output directory for checkpoints
            save_every: Save checkpoint every N steps
            val_every: Validate every N steps
            log_every: Log metrics every N steps
            plot_every: Update plots every N steps
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.save_every = save_every
        self.val_every = val_every
        self.log_every = log_every
        self.plot_every = plot_every

        # Create directories
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.summary_dir = self.output_dir / 'summary'
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.train_metrics = []
        self.val_metrics = []
        self.step = 0

        # For plotting
        self.plot_steps = []
        self.plot_losses = []
        self.plot_nll = []
        self.plot_kl = []
        self.plot_val_losses = []
        self.plot_pixel_mse = []  # Track pixel-level MSE

    def save_checkpoint(self, step: Optional[int] = None):
        """Save model checkpoint."""
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
        """Load model checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        print(f"Loaded checkpoint from step {self.step}")

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move batch to device
        batch = batch.to(self.device, dtype=torch.float32)

        # Debug: print data statistics on first step
        if self.step == 0:
            print(f"Batch stats - min: {batch.min():.3f}, max: {batch.max():.3f}, "
                  f"mean: {batch.mean():.3f}, std: {batch.std():.3f}")

        # Forward pass
        loss, metrics = self.model.get_loss(batch)

        # Also compute pixel MSE for tracking
        with torch.no_grad():
            recon = self.model.forward(batch)
            pixel_mse = torch.mean((batch - recon) ** 2).item()
            metrics['pixel_mse'] = pixel_mse

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Convert metrics to float
        metrics = {k: v.item() if torch.is_tensor(v) else v
                  for k, v in metrics.items()}

        return metrics

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        n_batches: int = 10
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        val_metrics = {}
        n_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break

                batch = batch.to(self.device, dtype=torch.float32)
                loss, metrics = self.model.get_loss(batch)

                # Accumulate metrics
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v.item() * batch.shape[0]
                n_samples += batch.shape[0]

        # Average metrics
        val_metrics = {f'val_{k}': v / n_samples for k, v in val_metrics.items()}

        return val_metrics

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        n_steps: int = 10000
    ):
        """
        Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            n_steps: Number of training steps
        """
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

            # Training step
            metrics = self.train_step(batch)
            self.step += 1

            # Update running average
            alpha = 0.99 if running_metrics else 0.0
            for k, v in metrics.items():
                running_metrics[k] = alpha * running_metrics.get(k, 0) + (1 - alpha) * v

            # Log metrics
            if self.step % self.log_every == 0:
                self.train_metrics.append({
                    'step': self.step,
                    **running_metrics
                })

                # Store for plotting
                self.plot_steps.append(self.step)
                self.plot_losses.append(running_metrics.get('loss', 0))
                self.plot_nll.append(running_metrics.get('nll_loss', 0))
                self.plot_kl.append(running_metrics.get('kl_loss', 0))
                self.plot_pixel_mse.append(running_metrics.get('pixel_mse', 0))

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
        """Save training and validation metrics."""
        metrics_path = self.output_dir / 'metrics.json'

        metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {metrics_path}")

    def save_reconstructions(self, batch, step):
        """Save reconstruction comparison images."""
        self.model.eval()

        with torch.no_grad():
            # Get batch and reconstruct
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)  # Add batch dim if needed

            batch = batch[:8].to(self.device, dtype=torch.float32)  # Take first 8 samples
            recon = self.model.forward(batch)

            # Calculate reconstruction MSE
            mse = torch.mean((batch - recon) ** 2, dim=(1, 2, 3))  # Per-sample MSE

            # Plot comparisons (show 3 spectral channels as RGB)
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))

            for i in range(min(4, batch.shape[0])):
                # Original - take 3 channels for RGB visualization
                orig = batch[i, [100, 500, 900], :, :].cpu().numpy()  # Pick 3 spectral channels
                orig = np.transpose(orig, (1, 2, 0))
                orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)  # Normalize to [0,1]

                # Reconstruction
                rec = recon[i, [100, 500, 900], :, :].cpu().numpy()
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
                mid_y = batch.shape[2] // 2
                mid_x = batch.shape[3] // 2
                orig_spectrum = batch[i, :, mid_y, mid_x].cpu().numpy()
                rec_spectrum = recon[i, :, mid_y, mid_x].cpu().numpy()

                axes[i, 3].plot(orig_spectrum, 'b-', alpha=0.7, label='Original')
                axes[i, 3].plot(rec_spectrum, 'r-', alpha=0.7, label='Recon')
                axes[i, 3].set_xlabel('Spectral Channel')
                axes[i, 3].set_ylabel('Value')
                axes[i, 3].set_title(f'Spectrum at ({mid_y},{mid_x})')
                axes[i, 3].legend()
                axes[i, 3].grid(True, alpha=0.3)

            plt.suptitle(f'Reconstructions at Step {step}')
            plt.tight_layout()

            # Save to figures directory
            figures_dir = self.output_dir / 'figures'
            figures_dir.mkdir(exist_ok=True)
            plt.savefig(figures_dir / f'reconstructions_step_{step:06d}.png', dpi=100, bbox_inches='tight')
            plt.close()

        self.model.train()

    def update_plots(self):
        """Update live plots with log-log scale and xmin=100 (or linear if < 100 steps)."""
        if len(self.plot_steps) < 2:
            return  # Need at least 2 points to plot

        # Determine if we should use log scale
        max_step = max(self.plot_steps)
        use_log = max_step >= 100

        if use_log:
            # Filter for steps >= 100 for log-log plot
            valid_idx = [i for i, s in enumerate(self.plot_steps) if s >= 100]
            if len(valid_idx) < 2:
                # Fall back to linear if not enough points
                use_log = False
                valid_idx = list(range(len(self.plot_steps)))
        else:
            # Use all points for linear plot
            valid_idx = list(range(len(self.plot_steps)))

        steps = [self.plot_steps[i] for i in valid_idx]
        losses = [self.plot_losses[i] for i in valid_idx]
        nll = [self.plot_nll[i] for i in valid_idx]
        kl = [self.plot_kl[i] for i in valid_idx]
        pixel_mse = [self.plot_pixel_mse[i] for i in valid_idx] if self.plot_pixel_mse else []

        # Also filter validation metrics
        val_steps = []
        val_losses = []
        for metric in self.val_metrics:
            if (not use_log or metric['step'] >= 100) and 'val_loss' in metric:
                val_steps.append(metric['step'])
                val_losses.append(metric['val_loss'])

        # Plot 1: Total Loss
        plt.figure(figsize=(10, 6))
        if use_log:
            plt.loglog(steps, losses, 'b-', label='Train Loss', alpha=0.7)
            if val_steps:
                plt.loglog(val_steps, val_losses, 'r^', label='Val Loss', markersize=8)
            plt.title('Total Loss (log-log scale)')
            plt.xlim(left=100)
        else:
            plt.plot(steps, losses, 'b-', label='Train Loss', alpha=0.7)
            if val_steps:
                plt.plot(val_steps, val_losses, 'r^', label='Val Loss', markersize=8)
            plt.title('Total Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both' if use_log else 'major')
        plt.tight_layout()
        plt.savefig(self.summary_dir / 'loss.png', dpi=100)
        plt.close()

        # Plot 2: Pixel MSE - THIS IS THE ACTUAL RECONSTRUCTION ERROR
        plt.figure(figsize=(10, 6))
        if pixel_mse:  # Only plot if we have data
            if use_log:
                plt.loglog(steps, pixel_mse, 'g-', alpha=0.7)
                plt.title('Pixel MSE (Reconstruction Error) - log-log scale')
                plt.xlim(left=100)
            else:
                plt.plot(steps, pixel_mse, 'g-', alpha=0.7)
                plt.title('Pixel MSE (Reconstruction Error)')
            plt.xlabel('Step')
            plt.ylabel('Mean Squared Error')
            plt.grid(True, alpha=0.3, which='both' if use_log else 'major')
        plt.tight_layout()
        plt.savefig(self.summary_dir / 'recons_err.png', dpi=100)
        plt.close()

        # Plot 3: KL Divergence
        plt.figure(figsize=(10, 6))
        if use_log:
            plt.loglog(steps, kl, 'm-', alpha=0.7)
            plt.title('KL Divergence (log-log scale)')
            plt.xlim(left=100)
        else:
            plt.plot(steps, kl, 'm-', alpha=0.7)
            plt.title('KL Divergence')
        plt.xlabel('Step')
        plt.ylabel('KL Loss')
        plt.grid(True, alpha=0.3, which='both' if use_log else 'major')
        plt.tight_layout()
        plt.savefig(self.summary_dir / 'kl.png', dpi=100)
        plt.close()