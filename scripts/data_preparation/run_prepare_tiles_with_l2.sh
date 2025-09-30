#!/bin/bash
# Prepare TEMPO tiles with L2 components (NO2, O3TOT, HCHO, CLDO4)

uv run python src/scripts/prepare_tempo_tiles_with_l2.py configs/data_preparation/prepare_tiles_with_l2.yaml "$@"