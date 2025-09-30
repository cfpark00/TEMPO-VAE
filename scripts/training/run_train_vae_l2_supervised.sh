#!/bin/bash

# Train VAE with multi-task L2 supervision

uv run python src/scripts/train_vae_l2_supervised.py configs/training/train_vae_l2_supervised.yaml "$@"