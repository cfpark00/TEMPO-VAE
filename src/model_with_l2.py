"""
VAE model with L2 product prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class L2PredictionHead(nn.Module):
    """Conv1D MLP for predicting ALL L2 products from VAE latents."""

    def __init__(self, latent_channels: int = 32, hidden_dims: list = [512, 512], n_outputs: int = 4):
        super().__init__()

        layers = []
        in_channels = latent_channels

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
            ])
            in_channels = hidden_dim

        # Output layer - 4 channels for 4 L2 products
        layers.append(nn.Conv2d(in_channels, n_outputs, kernel_size=1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor [B, latent_channels, H/4, W/4]

        Returns:
            L2 predictions [B, 4, H/4, W/4]
        """
        return self.mlp(z)


class VAEWithL2Supervision(nn.Module):
    """VAE with multi-task L2 product supervision."""

    def __init__(self, base_vae, latent_channels: int = 32, mlp_hidden: list = [512, 512]):
        super().__init__()
        self.vae = base_vae

        # Single prediction head for all 4 L2 products
        self.l2_head = L2PredictionHead(latent_channels, mlp_hidden, n_outputs=4)

        # Product order for indexing
        self.l2_products = ['NO2', 'O3TOT', 'HCHO', 'CLDO4']

        # Average pooling to downsample L2 targets from 64x64 to 16x16
        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with reconstruction and L2 predictions.

        Args:
            x: Input spectral tiles [B, 1028, 64, 64]

        Returns:
            Dictionary with:
                - 'reconstruction': Reconstructed spectral data
                - 'posterior': Distribution object
                - 'z': Sampled latent
                - 'l2_predictions': Dict of L2 predictions
        """
        # Use EXACT same VAE forward as original
        posterior = self.vae.encode(x)
        z = posterior.sample()
        reconstruction = self.vae.decode(z)

        # Predict ALL L2 products with single head
        l2_all = self.l2_head(z)  # [B, 4, 16, 16]

        # Split into individual products
        l2_predictions = {}
        for i, product in enumerate(self.l2_products):
            l2_predictions[product] = l2_all[:, i:i+1, :, :]  # Keep channel dim

        return {
            'reconstruction': reconstruction,
            'posterior': posterior,
            'z': z,
            'l2_predictions': l2_predictions
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        kl_weight: float = 1e-6,
        l2_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined VAE and L2 supervision loss.
        EXACTLY matches original VAE loss computation plus L2.

        Args:
            batch: Dictionary with 'spectral' and L2 product tensors
            kl_weight: Weight for KL divergence loss
            l2_weights: Weights for each L2 product loss

        Returns:
            Total loss and metrics dictionary
        """
        if l2_weights is None:
            l2_weights = {
                'NO2': 0.1,
                'O3TOT': 0.1,
                'HCHO': 0.1,
                'CLDO4': 0.1
            }

        x = batch['spectral']

        # Use EXACT same loss computation as original get_loss()
        reconstruction, posterior = self.vae(x)

        # Get z for L2 predictions (reuse the same sample)
        z = posterior.sample()

        # Reconstruction loss (EXACTLY as in original)
        if self.vae.nll_loss_type == "l1":
            rec_loss = F.l1_loss(x, reconstruction, reduction="none")
        elif self.vae.nll_loss_type == "l2":
            rec_loss = F.mse_loss(x, reconstruction, reduction="none")
        else:
            raise ValueError("nll_loss_type must be l1 or l2")

        nll_loss = rec_loss / torch.exp(self.vae.logvar) + self.vae.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # KL loss (EXACTLY as in original)
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = self.vae.kl_weight * kl_loss

        # L2 supervision losses - predict from the SAME z sample
        l2_all = self.l2_head(z)  # [B, 4, 16, 16]

        l2_losses = {}
        total_l2_loss = 0.0

        for i, product in enumerate(self.l2_products):
            if product in batch:
                # Downsample target to match latent resolution
                target = batch[product].unsqueeze(1)  # Add channel dim [B, 1, 64, 64]
                target_downsampled = self.downsample(target)  # [B, 1, 16, 16]

                # Handle NaN values - only compute loss on valid pixels
                pred = l2_all[:, i:i+1, :, :]  # [B, 1, 16, 16]
                valid_mask = ~torch.isnan(target_downsampled)

                if valid_mask.sum() > 0:
                    l2_loss = F.mse_loss(
                        pred[valid_mask],
                        target_downsampled[valid_mask],
                        reduction='mean'
                    )
                    l2_losses[f'{product}_loss'] = l2_loss.item()
                    total_l2_loss += l2_weights[product] * l2_loss

        # Total loss (VAE loss is already weighted)
        vae_loss = nll_loss + kl_loss  # kl_loss already includes kl_weight
        total_loss = vae_loss + total_l2_loss

        # Collect metrics (EXACTLY matching original names)
        metrics = {
            'loss': total_loss.item(),  # Match original name
            'nll_loss': nll_loss.item(),  # Match original name
            'kl_loss': kl_loss.item(),  # This is already weighted
            **l2_losses
        }

        return total_loss, metrics