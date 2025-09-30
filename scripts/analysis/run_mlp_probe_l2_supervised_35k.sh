#!/bin/bash

# Run MLP probe analysis on L2 supervised VAE checkpoint at step 35000
uv run python src/scripts/linear_probe_analysis.py configs/analysis/mlp_probe_l2_supervised_35k.yaml --overwrite