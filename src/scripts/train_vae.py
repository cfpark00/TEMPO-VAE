#!/usr/bin/env python3
"""
Train VAE on TEMPO spectral tiles.
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
from src.model import get_model  # Single file with exact mltools architecture
from src.tempo_data import TEMPODataLoader
from src.train_utils import Trainer, seed_all, get_device


def validate_config(config):
    """Validate configuration."""
    # Required fields
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")

    if 'data' not in config:
        raise ValueError("FATAL: 'data' section is required in config")

    if 'train_dir' not in config['data']:
        raise ValueError("FATAL: 'data.train_dir' is required in config")

    if 'model' not in config:
        raise ValueError("FATAL: 'model' section is required in config")

    if 'training' not in config:
        raise ValueError("FATAL: 'training' section is required in config")

    # Check data directories exist
    train_dir = Path(config['data']['train_dir'])
    if not train_dir.exists():
        raise ValueError(f"FATAL: Training directory doesn't exist: {train_dir}")

    if 'val_dir' in config['data']:
        val_dir = Path(config['data']['val_dir'])
        if not val_dir.exists():
            raise ValueError(f"FATAL: Validation directory doesn't exist: {val_dir}")


def main(config_path, overwrite=False, debug=False):
    """Main training function."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate
    validate_config(config)

    # Setup output directory
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Create subdirectories
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

    # Debug mode adjustments
    if debug:
        print("DEBUG MODE: Reduced training steps and data")
        config['training']['n_steps'] = min(200, config['training'].get('n_steps', 10000))  # More steps for plots
        config['data']['min_buffer_size'] = min(10, config['data'].get('min_buffer_size', 200))
        config['training']['save_every'] = 50
        config['training']['val_every'] = 25
        config['training']['plot_every'] = 20  # More frequent plots in debug

    # Create data loaders
    print("\nLoading training data...")
    train_loader = TEMPODataLoader.get_dataloader(
        data_dir=config['data']['train_dir'],
        batch_size=config['data'].get('batch_size', 16),
        num_workers=config['data'].get('num_workers', 4),
        min_buffer_size=config['data'].get('min_buffer_size', 200),
        verbose=True
    )

    val_loader = None
    if 'val_dir' in config['data']:
        print("\nLoading validation data...")
        val_loader = TEMPODataLoader.get_dataloader(
            data_dir=config['data']['val_dir'],
            batch_size=config['data'].get('batch_size', 16),
            num_workers=config['data'].get('val_num_workers', 1),
            min_buffer_size=config['data'].get('val_min_buffer_size', 100),
            verbose=True
        )

    # Create model
    print("\nInitializing model...")

    # Build model params exactly like old config
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

    model = get_model(model_params, device)

    # Model already on device and has optimizer attached

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create trainer
    train_config = config['training']
    trainer = Trainer(
        model=model,
        optimizer=model.optimizer,  # Use optimizer attached to model
        device=device,
        output_dir=output_dir,
        save_every=train_config.get('save_every', 1000),
        val_every=train_config.get('val_every', 100),
        log_every=train_config.get('log_every', 10),
        plot_every=train_config.get('plot_every', 50)
    )

    # Load checkpoint if resuming
    if 'resume_from' in train_config:
        ckpt_path = train_config['resume_from']
        print(f"\nResuming from checkpoint: {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)

    # Training
    print(f"\nStarting training for {train_config['n_steps']} steps...")
    print(f"Output directory: {output_dir}")

    start_time = datetime.now()
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_steps=train_config['n_steps']
    )
    end_time = datetime.now()

    # Log training time
    duration = end_time - start_time
    print(f"\nTraining completed in {duration}")

    # Save final info
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
    parser = argparse.ArgumentParser(description="Train VAE on TEMPO tiles")
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output directory')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with reduced steps')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)