#!/bin/bash
# Evaluate reconstruction error across all checkpoints in an experiment

uv run python src/scripts/evaluate_reconstruction.py configs/analysis/evaluate_reconstruction.yaml "$@"