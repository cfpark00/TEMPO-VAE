# Linear Probe Analysis Development
**Date:** 2025-09-18
**Time:** 17:21
**Developer:** Claude

## Summary
Developed comprehensive linear probe and MLP probe analysis tools to map VAE latent representations to TEMPO L2 component fields (NO2, O3TOT, HCHO, CLDO4).

## Major Tasks Completed

### 1. Linear Probe Analysis Pipeline
- Created `src/scripts/linear_probe_analysis.py` - Main analysis script
- Implemented data pipeline:
  - Loads L1 radiance files and corresponding L2 product files
  - Normalizes L1 data (log transform, z-score, clipping) matching VAE training
  - Extracts VAE latent representations (32 channels, H//4 × W//4)
  - Processes L2 component fields with appropriate normalization
  - 4x average pooling to match latent resolution

### 2. Advanced Normalization Strategies
- **NO2 & HCHO**: Implemented robust asinh transform with MAD-based scaling
  - Handles negative values from retrieval noise
  - Robust to extreme outliers using Median Absolute Deviation
  - Range: approximately [-5, 5] after normalization

- **O3TOT**: Standard z-score normalization (already Gaussian-like)

- **CLDO4**: Logit transform with squeeze
  - Maps [0,1] → [0.01, 0.99] → logit → [-4.6, 4.6]
  - Perfectly invertible
  - Spreads out bimodal cloud fraction distribution

### 3. MLP Probe Extension
- Added `MLPProbe` class with configurable architecture:
  - 2 hidden layers (512 units each)
  - ReLU activation, dropout (0.1)
  - L2 regularization via AdamW
- Updated training to support both linear and MLP architectures

### 4. Training Improvements
- Implemented mini-batch training (batch_size=512)
  - Prevents OOM with large datasets
  - Data shuffling each epoch
- Tracks and saves best model based on validation loss
- Removed early stopping, runs full epochs
- Learning curves visualization:
  - Log-scale x-axis starting at epoch 10
  - Vertical line marking best epoch
  - Both train and validation losses

### 5. Data Analysis Visualizations
Created comprehensive plots saved in `data_stats/`:
- `all_normalizations_comparison.png` - Raw vs normalized distributions
- `input_latent_distributions.png` - Shared input latent features
- `target_distributions.png` - 2x2 grid of normalized targets

Per-component analysis in `figures/`:
- Learning curves with best epoch indicator
- Predictions vs ground truth scatter plots
- Residual distributions
- R² score summary bar chart

### 6. Configuration Files
- `configs/analysis/linear_probe.yaml` - Linear probe configuration
- `configs/analysis/mlp_probe.yaml` - MLP probe configuration
- Both use same data paths and normalization settings
- Configurable: epochs, learning rate, batch size, weight decay

### 7. Execution Scripts
- `scripts/analysis/run_linear_probe.sh` - Minimal bash wrapper
- `scripts/analysis/run_mlp_probe.sh` - Minimal bash wrapper
- Support for `--debug` (3 files) and `--overwrite` flags

## Technical Details

### Shape Assertions
Added comprehensive shape checks:
- Component fields must match L1 dimensions [H, W]
- VAE latents verified as [1, 32, H//4, W//4]
- Pooled components must match latent spatial dimensions

### Sampling Strategy
- Sample 1000-2000 pixels per file from valid (non-NaN) regions
- Separate sampling per component (different valid masks)
- Maintains matched latent-component pairs

### Normalization Parameters Saved
- Component normalization stats saved to `results/component_norm_stats.json`
- Ensures reproducibility for inference

## Key Insights

### Component Field Statistics (Raw)
- **NO2**: Heavy-tailed, range -114 to 236 (×10¹⁵ molec/cm²)
- **O3TOT**: Gaussian-like, range 265-438 DU
- **HCHO**: Heavy-tailed, range -9 to 6 (×10¹⁶ molec/cm²)
- **CLDO4**: Bimodal, bounded [0, 1]

### Normalization Impact
- Robust MAD-based asinh transform critical for NO2/HCHO
- Logit transform effectively spreads cloud fraction
- All normalized ranges now comparable (~[-5, 5])

## Files Created/Modified

### Created
- `/src/scripts/linear_probe_analysis.py`
- `/configs/analysis/linear_probe.yaml`
- `/configs/analysis/mlp_probe.yaml`
- `/scripts/analysis/run_linear_probe.sh`
- `/scripts/analysis/run_mlp_probe.sh`
- `/scratch/check_component_stats.py`
- `/scratch/plot_raw_histograms.py`
- `/scratch/list_no2_fields.py`
- `/scratch/test_no2_normalization.py`
- `/scratch/test_cloud_logit.py`
- `/scratch/test_all_normalizations.py`

### Modified
- `CLAUDE.md` - Already contained project guidelines (read-only)
- Config parameters adjusted (epochs, pixels per file, batch size)

## Next Steps
- Run full analysis on all 35 training files
- Compare linear vs MLP probe performance
- Consider using true TEMPO validation split for final evaluation
- Potentially add more complex architectures (attention, deeper networks)