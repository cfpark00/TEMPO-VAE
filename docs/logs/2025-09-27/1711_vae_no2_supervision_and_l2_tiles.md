# VAE NO2 Supervision and L2 Tile Preparation Development
**Date:** 2025-09-27
**Time:** 17:11
**Developer:** Claude

## Summary
Extended the VAE training infrastructure to support NO2 latent supervision, created tile preparation pipeline that includes L2 atmospheric products alongside spectral data, and wrote comprehensive research documentation. Fixed critical silent failure issues in data processing.

## Major Tasks Completed

### 1. Reconstruction Evaluation Infrastructure
- Created `src/scripts/evaluate_reconstruction.py` to evaluate VAE reconstruction error across all checkpoints
- Loads validation tiles directly (not NC files) for efficiency
- Computes MSE, MAE, PSNR metrics for each checkpoint
- Generates plots showing metrics evolution over training steps
- Identifies best checkpoint for each metric
- Config: `configs/analysis/evaluate_reconstruction.yaml`

### 2. VAE Model Extension for NO2 Supervision
- Modified `AutoencoderKL` class to support NO2 prediction:
  - Added `no2_weight` and `no2_mlp_hidden` parameters
  - Implemented MLP probe: latent (32 channels) → 512 → 512 → 1 (NO2)
  - Added `predict_no2()` method for inference
  - Used 1x1 convolutions to maintain spatial dimensions
- Updated `get_model()` to pass NO2 parameters from config
- Maintains backward compatibility - NO2 features only activate when configured

### 3. NO2 Supervised Training Script
- Created `src/scripts/train_vae_no2_supervised.py`:
  - Extends base trainer with `NO2SupervisedTrainer` class
  - Loads NO2 L2 files matching L1 tiles (would need proper tile-L2 pairing)
  - Applies asinh normalization matching linear probe
  - Downsamples NO2 by 4x to match latent resolution (64→16)
  - Computes MSE loss between predicted and actual NO2
  - Joint optimization: reconstruction + KL + weighted NO2 loss
- Config: `configs/training/train_vae_no2_latent_supervised.yaml`:
  - Minimal configuration with `no2_weight: 0.1`
  - MLP architecture [512, 512] matching probe analysis
  - L2 paths and product specifications

### 4. L2 Tile Preparation Pipeline
- Created `src/scripts/prepare_tempo_tiles_with_l2.py`:
  - Extracts tiles from L1 and ALL L2 products simultaneously
  - Maintains exact spatial correspondence between L1 and L2
  - Applies same augmentations (flips, rotations) to all components
  - Component-specific normalization:
    - NO2/HCHO: asinh transform (handles negatives and heavy tails)
    - O3TOT: z-score normalization
    - CLDO4: logit transform with epsilon squeeze
  - Saves at full 64x64 resolution (downsampling happens during training)
  - Directory structure:
    ```
    train/
    ├── 00000.pt         # Spectral tiles [64, 64, 64, 1028]
    ├── l2_NO2/00000.pt  # NO2 tiles [64, 64, 64]
    ├── l2_O3TOT/00000.pt
    ├── l2_HCHO/00000.pt
    └── l2_CLDO4/00000.pt
    ```

### 5. Research Integrity Fixes
**Critical issue**: Original script had multiple silent failures - would continue with 0 L2 files found!

Fixed by:
- Removing ALL try/except blocks that hide errors
- Adding immediate failure when L2 files can't be loaded
- Explicit error messages showing expected file paths
- Final validation that raises error if no L2 data found
- Fixed config paths (was looking in non-existent `/n/home12/cfpark00/tempo_data/`)
- Fixed product names in filename replacement ("NO2" not "NO2-MINDS")
- Fixed normalization to exactly match linear probe analysis

**Key principle enforced**: FAIL FAST - no silent failures, no fallbacks, no warnings that continue execution

### 6. Comprehensive Research Documentation
Created `/n/home12/cfpark00/TEMPO/docs/research_report.md`:
- Complete project overview and TEMPO satellite description
- Research objectives and methodology
- Detailed technical implementation
- Experimental results (Linear R²: 0.12-0.74, MLP R²: 0.21-0.92)
- Key findings: Excellent cloud detection (R²=0.92), good ozone (R²=0.75), NO2 challenges
- Repository structure and development philosophy
- Reproducibility information

### 7. Visualization Tools
Created `scratch/tiles_with_l2_check/visualize_tiles_with_l2.py`:
- Grid visualization showing L1 PCA RGB alongside L2 components
- Rows: Different tiles from various files
- Columns: PCA RGB (L1), NO2, O3TOT, HCHO, CLDO4
- Component-specific colormaps
- Handles missing data gracefully

## Technical Details

### Normalization Matching
Ensured perfect consistency with linear probe analysis:
- asinh: `asinh(data / scale)` without median subtraction
- logit: Squeeze to [0.01, 0.99] then apply logit
- All stats computed from first 20 files for consistency

### Shape Management
- Tiles: [batch, 64, 64, 1028] for spectral
- L2 tiles: [batch, 64, 64] for each component
- Latent: [batch, 32, 16, 16] after 4x downsampling
- NO2 prediction: [batch, 1, 16, 16] from MLP probe

### Failed Approaches (Important Lessons)
1. Initial NO2 training script assumed dataloader would return source filenames - it doesn't
2. Tried to load L2 on-the-fly during training - impossible without tile position info
3. Silent failures everywhere - now replaced with immediate crashes

## Files Created/Modified

### Created
- `/src/scripts/evaluate_reconstruction.py`
- `/src/scripts/train_vae_no2_supervised.py`
- `/src/scripts/prepare_tempo_tiles_with_l2.py`
- `/configs/analysis/evaluate_reconstruction.yaml`
- `/configs/training/train_vae_no2_latent_supervised.yaml`
- `/configs/data_preparation/prepare_tiles_with_l2.yaml`
- `/scripts/analysis/run_evaluate_reconstruction.sh`
- `/scripts/training/run_train_vae_no2_supervised.sh`
- `/scripts/data_preparation/run_prepare_tiles_with_l2.sh`
- `/docs/research_report.md`
- `/scratch/tiles_with_l2_check/visualize_tiles_with_l2.py`

### Modified
- `/src/model.py` - Added NO2 supervision to AutoencoderKL
- `/src/scripts/prepare_tempo_tiles_with_l2.py` - Multiple fixes for research integrity

## Key Insights

### Data Pipeline Requirements
- Proper L1-L2 pairing requires preprocessing tiles together
- Cannot retrofit L2 supervision onto existing tiles without position info
- Spatial correspondence must be maintained exactly

### Research Integrity
- Silent failures are "criminal" to research (user's words - and they're right!)
- Better to crash immediately than run for hours with wrong behavior
- Every fallback must be replaced with explicit failure
- Clear error messages save debugging time

### Model Architecture
- 1x1 convolutions work well for maintaining spatial dims in probe
- MLP probe [512, 512] significantly outperforms linear probe
- Joint training with auxiliary losses requires careful weight balancing

## Next Steps
- Run full L2 tile preparation on all data
- Train VAE with NO2 supervision using properly paired tiles
- Extend to multi-task learning with all L2 products
- Complete AAAI 2026 paper writing

## Status
Successfully created infrastructure for supervised VAE training with atmospheric products. Data pipeline now properly pairs L1 and L2 tiles with fail-fast validation. Ready for full supervised training experiments.