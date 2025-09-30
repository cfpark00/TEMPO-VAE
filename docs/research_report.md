# TEMPO Satellite Data Analysis Research Report

## Project Overview

This document comprehensively describes the TEMPO satellite data analysis research project conducted in September 2025. The project focuses on developing machine learning approaches, specifically Variational Autoencoders (VAEs), to analyze hyperspectral radiance data from NASA's TEMPO (Tropospheric Emissions: Monitoring of Pollution) satellite and map learned representations to atmospheric trace gas concentrations.

## 1. TEMPO Satellite and Data

### 1.1 TEMPO Mission
TEMPO is a NASA satellite instrument that measures atmospheric pollution over North America with unprecedented temporal and spatial resolution. It provides hourly measurements during daylight hours, enabling tracking of air quality dynamics throughout the day.

### 1.2 Data Characteristics
- **Spectral Resolution**: 1028 channels covering 290-490 nm (UV-visible spectrum)
- **Spatial Resolution**: Approximately 2.1 km × 4.4 km at nadir
- **Temporal Resolution**: Hourly during daylight hours
- **Coverage**: North America (Mexico to Canada, Atlantic to Pacific)
- **Study Region**: Los Angeles area, January 2025

### 1.3 Data Products
The research uses two levels of TEMPO data:

**L1 Data (Radiance):**
- Raw spectral measurements from the satellite
- File format: `TEMPO_RAD_L1_V03_YYYYMMDDTHHMMSSZ_SXXXGYY.nc`
- Band: `band_290_490_nm` containing 1028 spectral channels
- Dimensions: Variable, typically ~130 × 2048 × 1028 (mirror × track × spectral)

**L2 Data (Atmospheric Products):**
- Retrieved atmospheric gas concentrations from L1 radiance
- Products analyzed:
  - **NO2** (Nitrogen Dioxide): `vertical_column_troposphere` in molecules/cm²
  - **O3TOT** (Total Ozone): `column_amount_o3` in Dobson Units
  - **HCHO** (Formaldehyde): `vertical_column` in molecules/cm²
  - **CLDO4** (Cloud Fraction): `cloud_fraction` (0-1 range)

## 2. Research Objectives

### 2.1 Primary Goals
1. **Representation Learning**: Train a VAE to learn compressed representations of TEMPO hyperspectral radiance data
2. **Atmospheric Retrieval**: Map learned latent representations to atmospheric gas concentrations
3. **Method Validation**: Compare linear vs. nonlinear (MLP) probes for extracting atmospheric information from latent space
4. **Scientific Understanding**: Identify which atmospheric signals are best captured by unsupervised representation learning

### 2.2 Research Questions
- Can VAE latent representations effectively encode atmospheric composition information?
- Which atmospheric products are most accurately predicted from learned representations?
- Does nonlinear mapping (MLP) significantly improve over linear probing?
- What is the optimal normalization strategy for different atmospheric products?

## 3. Methodology

### 3.1 Data Processing Pipeline

#### 3.1.1 Spectral Data Normalization
The radiance data undergoes a three-step normalization process:
1. **Log Transform**: `log(clamp(radiance, min=1.0, max=∞))`
2. **Z-Score Normalization**: Using global mean and standard deviation computed from training data
3. **Clipping**: Values clipped to [-10, 10] range

#### 3.1.2 Tile Extraction
- **Tile Size**: 64 × 64 spatial pixels × 1028 spectral channels
- **Tiles Per File**: 64 tiles with overlapping extraction
- **Augmentation**: Random flips (horizontal/vertical) and 90° rotations
- **Train/Val Split**: 70/30 split with seed=42 for reproducibility

#### 3.1.3 L2 Product Normalization
Each atmospheric product uses tailored normalization:
- **NO2**: asinh transform with MAD-based scaling: `asinh(data / scale)`
  - Scale = 1.4826 × MAD (Median Absolute Deviation)
  - Handles negative values and heavy-tailed distributions
- **O3TOT**: Standard z-score normalization
- **HCHO**: asinh transform (same as NO2)
- **CLDO4**: Logit transform with squeeze: `logit(0.01 + 0.98 × data)`

### 3.2 Model Architecture

#### 3.2.1 Variational Autoencoder
- **Architecture**: Exact implementation matching mltools specifications
- **Parameters**: 27.3 million
- **Encoder**:
  - Input: [1028, 64, 64]
  - Channel progression: 512 → 256 → 128
  - Latent: 32 channels at 16×16 spatial resolution (4× downsampling)
- **Decoder**: Mirror architecture with transposed convolutions
- **Attention**: Middle block attention with 4 heads
- **Activation**: GELU
- **Normalization**: Group normalization (8 groups)

#### 3.2.2 Training Configuration
- **Optimizer**: AdamW (lr=0.0001, weight_decay=0.05, betas=[0.9, 0.95])
- **Loss Components**:
  - Reconstruction: L1 loss
  - KL Divergence: Weight = 1e-6
- **Training Duration**: 200,000 steps (~41 hours)
- **Batch Size**: 32
- **Checkpointing**: Every 5,000 steps

### 3.3 Probe Analysis

#### 3.3.1 Linear Probe
- Direct linear mapping: Latent channels (32) → Component field (1)
- Per-pixel regression at latent resolution (16×16)
- L2 regularization via weight decay

#### 3.3.2 MLP Probe
- Architecture: 32 → 512 → 512 → 1
- Activation: ReLU with dropout (0.1)
- Training: 2000 epochs, batch size 512
- Learning rate: 0.001 with AdamW optimizer

#### 3.3.3 Evaluation Strategy
- Sample 1000 pixels per file from valid (non-NaN) regions
- 80/20 train/test split
- Metrics: R² score, MSE, learning curves
- Best model selection based on validation loss

## 4. Results

### 4.1 VAE Training Performance
- **Final Reconstruction Loss**: Converged after ~150,000 steps
- **Training Time**: 1 day, 17 hours on GPU
- **Checkpoints Generated**: 39 checkpoints (every 5,000 steps)

### 4.2 Linear Probe Results
| Component | R² Score | Interpretation |
|-----------|----------|----------------|
| NO2       | 0.119    | Poor - high noise, complex retrieval |
| O3TOT     | 0.530    | Moderate - captures broad patterns |
| HCHO      | 0.507    | Moderate - similar to O3TOT |
| CLDO4     | 0.737    | Good - cloud patterns well represented |

### 4.3 MLP Probe Results
| Component | R² Score | Improvement over Linear |
|-----------|----------|-------------------------|
| NO2       | 0.209    | +75% (still challenging) |
| O3TOT     | 0.754    | +42% (strong performance) |
| HCHO      | 0.498    | -2% (marginal change) |
| CLDO4     | 0.923    | +25% (excellent) |

### 4.4 Key Findings
1. **Cloud Detection Excellence**: CLDO4 achieves R²=0.92 with MLP probe, indicating VAE latents strongly encode cloud information
2. **Ozone Retrieval Success**: O3TOT shows good performance (R²=0.75), suggesting stratospheric signals are well captured
3. **NO2 Challenges**: Despite improvements, NO2 remains difficult due to:
   - Low signal-to-noise ratio in raw measurements
   - Complex atmospheric chemistry and transport
   - Retrieval sensitivity to surface and aerosol properties
4. **Nonlinearity Benefits**: MLP probes significantly improve performance for most products except HCHO

## 5. Technical Infrastructure

### 5.1 Repository Structure
```
/n/home12/cfpark00/TEMPO/
├── src/                    # Python source code
│   ├── model.py           # VAE implementation
│   ├── tempo_data.py      # Data loading utilities
│   ├── train_utils.py     # Training infrastructure
│   └── scripts/           # Orchestration scripts
├── configs/               # YAML configuration files
│   ├── data_preparation/  # Data processing configs
│   ├── training/         # Model training configs
│   └── analysis/         # Analysis configs
├── scripts/              # Bash execution wrappers
├── data/                 # Experiment outputs (gitignored)
├── docs/                 # Documentation and logs
└── paper/               # AAAI 2026 paper draft
```

### 5.2 Key Components

#### 5.2.1 Data Processing (`src/scripts/`)
- `prepare_tempo_tiles.py`: Extract and normalize spectral tiles
- `prepare_tempo_tiles_with_l2.py`: Joint extraction with L2 products
- `compute_tempo_stats.py`: Calculate global normalization parameters

#### 5.2.2 Training Scripts
- `train_vae.py`: Standard VAE training
- `train_vae_no2_supervised.py`: VAE with NO2 latent supervision (experimental)

#### 5.2.3 Analysis Scripts
- `linear_probe_analysis.py`: Linear mapping from latents to L2 products
- `analyze_reconstruction.py`: VAE reconstruction quality assessment
- `evaluate_reconstruction.py`: Multi-checkpoint evaluation

### 5.3 Development Philosophy
The codebase follows strict research integrity principles:
- **Fail-fast validation**: Immediate errors on missing configs or invalid states
- **No silent failures**: Explicit crashes rather than hidden fallbacks
- **Full reproducibility**: Configs saved with outputs, fixed random seeds
- **Clear separation**: Implementation (HOW) vs. orchestration (WHAT/WHEN)

## 6. Current Status and Next Steps

### 6.1 Completed Work
- ✅ Data download and preprocessing pipeline
- ✅ Global normalization statistics computation
- ✅ VAE training (27M parameters, 200k steps)
- ✅ Linear probe analysis across all L2 products
- ✅ MLP probe analysis with improved results
- ✅ Comprehensive evaluation and visualization tools
- ✅ Tile preparation with L2 components (ready for supervised training)

### 6.2 Ongoing Development
- Implementation of NO2-supervised VAE training
- Integration of multiple L2 products for multi-task learning
- Paper writing for AAAI 2026 submission

### 6.3 Future Directions
1. **Multi-task Learning**: Joint training with all L2 products
2. **Attention Mechanisms**: Explore self-attention for better spatial context
3. **Temporal Modeling**: Leverage TEMPO's hourly resolution
4. **Physical Constraints**: Incorporate atmospheric physics into model
5. **Uncertainty Quantification**: Bayesian approaches for retrieval confidence

## 7. Scientific Impact

This research demonstrates that unsupervised representation learning can effectively extract atmospheric information from hyperspectral satellite data. The strong performance on cloud detection (R²=0.92) and ozone retrieval (R²=0.75) validates the approach, while challenges with NO2 highlight areas needing targeted improvements. The work provides a foundation for applying deep learning to satellite-based atmospheric monitoring, with potential applications in air quality forecasting, emission source identification, and climate studies.

## 8. Data and Code Availability

### 8.1 Data Sources
- **TEMPO L1/L2 Data**: Available from NASA Earthdata
- **Processed Tiles**: Generated using provided scripts
- **Trained Models**: Checkpoints saved in `data/vae_training/`

### 8.2 Reproducibility
All experiments can be reproduced using:
1. Configuration files in `configs/`
2. Execution scripts in `scripts/`
3. Fixed random seeds (seed=42)
4. Saved normalization statistics

## Appendix A: File Naming Conventions

### TEMPO Files
- L1: `TEMPO_RAD_L1_V03_YYYYMMDDTHHMMSSZ_SXXXGYY.nc`
- L2: `TEMPO_{PRODUCT}_L2_VXX_XX_YYYYMMDDTHHMMSSZ_SXXXGYY.nc`

### Processed Data
- Tiles: `NNNNN.pt` (5-digit zero-padded index)
- L2 tiles: `l2_{component}/NNNNN.pt`
- Stats: `{component}_stats.pt`, `mean_spectrum.pt`, `std_spectrum.pt`

## Appendix B: Configuration Parameters

### VAE Training
- Batch size: 32
- Learning rate: 1e-4
- KL weight: 1e-6
- Training steps: 200,000
- Validation frequency: Every 50 steps
- Checkpoint frequency: Every 5,000 steps

### Probe Training
- Epochs: 2,000
- Batch size: 512
- Learning rate: 1e-3
- Weight decay: 1e-2
- Architecture: Linear or MLP [512, 512]

### Data Processing
- Tile size: 64×64×1028
- Tiles per file: 64
- Train/val split: 70/30
- Random seed: 42
- Normalization clip: [-10, 10]

---

*Document Generated: September 2025*
*Research Location: Harvard University / Finkbeiner Lab*
*Primary Researcher: cfpark00*
*Development Support: Claude (Anthropic)*