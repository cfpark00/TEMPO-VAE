# Development Log: TEMPO Data Pipeline Setup
**Date:** 2025-01-17
**Time:** 00:06
**Developer:** Claude

## Summary
Established complete data processing pipeline for TEMPO satellite spectral radiance data, from downloading raw data to preparing ML-ready tiles with proper normalization.

## Major Tasks Completed

### 1. Data Download Pipeline
- Created minimal download script (`src/scripts/download_tempo_data.py`) - 95 lines
- Simplified config to only essential fields (`configs/data/download_tempo_jan_2025_LA.yaml`)
- Added NO2 L2 retrieval download capability (`download_tempo_no2_data.py`)
- Downloads from URL list at `/n/home12/cfpark00/TEMPO/data/jan_2025_files.txt`

### 2. Data Preparation Pipeline
- **Compute Global Statistics** (`src/scripts/compute_tempo_stats.py`)
  - Computes mean/std per spectral channel across multiple files
  - Saves as .pt files for consistent normalization
  - Config-driven with proper fail-fast validation

- **Tile Extraction** (`src/scripts/prepare_tempo_tiles.py`)
  - Extracts 64×64 tiles from TEMPO radiance data
  - Uses global normalization (preserves denormalization ability)
  - Added augmentations: random flips and 90° rotations
  - Changed from non-overlapping to overlapping tiles for better data utilization
  - Automatic 80/20 train/val split

### 3. Repository Organization
- Followed strict repo conventions from `docs/repo_usage.md`
- All scripts in `src/scripts/`
- Configs in `configs/data/` and `configs/data_preparation/`
- Bash wrappers in `scripts/`
- Data outputs to `data/` directory

## Key Design Decisions

1. **Global Normalization**: Computing statistics from sample of files and applying consistently across all data, rather than per-file normalization. This preserves the ability to denormalize reconstructed data.

2. **Overlapping Tiles with Augmentation**: Allowing tiles to overlap and adding random augmentations increases data diversity significantly.

3. **Fail-Fast Philosophy**: All scripts validate configs immediately and fail with clear error messages rather than silently falling back to defaults.

## File Structure Changes
```
TEMPO/
├── src/scripts/
│   ├── download_tempo_data.py          # RAD L1 download
│   ├── download_tempo_no2_data.py      # NO2 L2 download
│   ├── compute_tempo_stats.py          # Global statistics
│   └── prepare_tempo_tiles.py          # Tile extraction
├── configs/
│   ├── data/
│   │   ├── download_tempo_jan_2025_LA.yaml
│   │   └── download_tempo_no2_jan_2025_LA.yaml
│   └── data_preparation/
│       ├── compute_stats.yaml
│       └── prepare_tiles.yaml
├── scripts/
│   ├── data/
│   │   ├── download_tempo_jan_2025_LA.sh
│   │   └── download_tempo_no2_jan_2025_LA.sh
│   └── data_preparation/
│       ├── compute_stats.sh
│       └── prepare_tiles.sh
└── data/                                # Output directory
    ├── tempo_jan2025_LA/               # Downloaded RAD data
    ├── tempo_jan2025_LA_NO2/           # Downloaded NO2 data
    ├── tempo_stats/                    # Global normalization
    └── tempo_jan2025_LA_tiles/        # ML-ready tiles
```

## Workflow
1. Download TEMPO data: `bash scripts/data/download_tempo_jan_2025_LA.sh`
2. Compute global stats: `bash scripts/data_preparation/compute_stats.sh`
3. Prepare tiles: `bash scripts/data_preparation/prepare_tiles.sh --overwrite`

## Technical Details
- Tile size: 64×64 spatial, 1028 spectral channels
- Tiles per file: 64 (with overlaps allowed)
- Normalization: Log transform → z-score with global mean/std → clip to [-10, 10]
- Output format: PyTorch .pt files, shape [n_tiles, 64, 64, 1028]

## Notes for Future Development
- Global normalization stats saved with tiles for reconstruction
- Config uses 64×64 tiles and 64 tiles/file (changed from 32×64 and 20 tiles)
- Augmentations help with data diversity for VAE training
- All scripts follow fail-fast philosophy from repo conventions