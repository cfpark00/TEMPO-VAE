#!/bin/bash
# Extract PCA components from TEMPO spectral data

uv run python src/scripts/extract_pca_components.py \
    configs/analysis/extract_pca_components.yaml \
    --overwrite