#!/bin/bash

# Train VAE on TEMPO spectral tiles

# Run with uv environment
uv run python src/scripts/train_vae.py configs/training/train_vae_default.yaml "$@"