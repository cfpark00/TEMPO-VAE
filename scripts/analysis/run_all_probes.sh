#!/bin/bash
# Run all probe analyses (linear and MLP for both base and L2-supervised models)

set -e  # Exit on error

echo "=========================================="
echo "Running all probe analyses"
echo "=========================================="

# Base model - Linear probe
echo ""
echo "1/4: Running linear probe on base model..."
uv run python src/scripts/linear_probe_analysis.py \
    configs/analysis/linear_probe.yaml \
    --overwrite

# Base model - MLP probe
echo ""
echo "2/4: Running MLP probe on base model..."
uv run python src/scripts/linear_probe_analysis.py \
    configs/analysis/mlp_probe.yaml \
    --overwrite

# L2 supervised model - Linear probe
echo ""
echo "3/4: Running linear probe on L2-supervised model..."
uv run python src/scripts/linear_probe_analysis.py \
    configs/analysis/linear_probe_l2_supervised.yaml \
    --overwrite

# L2 supervised model - MLP probe
echo ""
echo "4/4: Running MLP probe on L2-supervised model..."
uv run python src/scripts/linear_probe_analysis.py \
    configs/analysis/mlp_probe_l2_supervised.yaml \
    --overwrite

echo ""
echo "=========================================="
echo "All probe analyses complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - data/analysis/linear_probe_analysis/"
echo "  - data/analysis/mlp_probe_analysis/"
echo "  - data/analysis/linear_probe_analysis_l2_supervised/"
echo "  - data/analysis/mlp_probe_l2_supervised/"