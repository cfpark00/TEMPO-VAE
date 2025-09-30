# Spectral Reconstruction Analysis and Figure 3 Creation

**Date:** September 29, 2025, 21:37
**Session Duration:** ~2 hours
**Main Focus:** Extract validation spectral reconstructions, create channel-wise error analysis figure, integrate into paper

## Summary

Created comprehensive channel-wise reconstruction error analysis for the VAE model. Extracted 61,440 validation spectra, computed RMSE per spectral channel for both normalized and physical radiance units, and generated Figure 3 for the paper showing reconstruction quality across the 290-490nm wavelength range.

## Tasks Completed

### 1. Spectral Data Extraction
- Created `/n/home12/cfpark00/TEMPO/scratch/spectral_reconstruction_data/extract_spectra.py`
- Loaded unsupervised VAE checkpoint (step 200,000)
- Processed all 960 validation tiles (15 files × 64 tiles/file)
- Sampled 64 random spectra per tile → 61,440 total spectra
- Implemented denormalization function to convert back to physical radiance units
- Saved both normalized and physical radiance versions for original and reconstructed spectra
- Output: `validation_spectra.pt` (1010.6 MB)

### 2. Physical Unit Verification
- Verified physical radiance values are in expected range (10¹⁰ to 3×10¹³)
- Confirmed units: photons s⁻¹ cm⁻² nm⁻¹ sr⁻¹
- Sanity checked against mean radiance spectrum

### 3. Channel-wise Error Analysis Plot
- Created `/n/home12/cfpark00/TEMPO/scratch/spectral_reconstruction_data/plot_channel_errors.py`
- Generated vertical 2-panel figure:
  - Panel (a): RMSE for normalized spectra (log scale)
  - Panel (b): RMSE for physical radiance with mean spectrum overlay (log scale)
- Both panels show wavelength-dependent reconstruction quality
- Added ±1 std shaded regions
- Final figure size: 8×6 inches
- Removed grid lines per user request
- Changed from MAE to MSE metric
- Removed "(log scale)" text from y-axis labels

### 4. Key Results
- **Normalized RMSE**: Mean = 0.0326, Range = [0.0085, 0.6935]
- **Physical RMSE**: Mean = 4.45×10¹¹, Range = [9.46×10⁹, 1.73×10¹²]
- **Average radiance**: Mean = 1.93×10¹³, Range = [4.99×10⁹, 3.30×10¹³]
- **Relative RMSE**: Mean = 8.28%
- Reconstruction errors are 1-2 orders of magnitude smaller than signal

### 5. Paper Integration
- Copied figure to `paper/figures/channel_wise_reconstruction_errors.png`
- Added Figure 3 to main.tex with comprehensive caption:
  - Describes both panels (normalized and physical)
  - Mentions 61,440 validation spectra from 960 tiles
  - Notes error magnitude relative to signal
  - Includes information about log scale and std shading
- Set LaTeX width to `0.5\textwidth` for proper sizing
- Successfully compiled paper (12 pages, no float overflow warnings)

## File Structure Changes

New directory and files created:
```
scratch/spectral_reconstruction_data/
├── extract_spectra.py          # Script to extract validation spectra
├── plot_channel_errors.py      # Script to generate Figure 3
├── validation_spectra.pt       # Extracted spectral data (1.0 GB)
└── channel_wise_errors.png     # Generated figure
```

Paper figure added:
```
paper/figures/channel_wise_reconstruction_errors.png
```

## Technical Details

### Denormalization Pipeline
The denormalization reverses the normalization applied during data preparation:
1. Reverse z-score: `log_rad = normalized * (std + 1e-8) + mean`
2. Reverse log transform: `radiance = exp(log_rad)`

### RMSE Computation
- Computed per channel over all 61,440 spectra
- Standard deviation propagated through RMSE calculation
- Used for uncertainty visualization (shaded regions)

### Figure Sizing Resolution
Initial attempt used `figsize=(6, 8)` which caused LaTeX float overflow (173.5pt).
Then tried `figsize=(3.5, 4)` which was too small.
Final solution: `figsize=(8, 6)` with LaTeX `width=0.5\textwidth` for proper scaling.

## Paper Status

Figure 3 successfully integrated. Caption emphasizes:
- Wavelength-dependent reconstruction quality
- Comparison between normalized and physical units
- Error magnitude relative to signal (1-2 orders of magnitude smaller)
- Log scale visualization for full dynamic range

## Next Steps

- Figure 4 (probing results) remains as placeholder
- May need to create visualization for linear vs MLP probe performance
- Consider additional quantitative analysis plots if needed

## Notes

- Python figure size (inches) and LaTeX width multiplier must be coordinated
- Log scale is appropriate for showing full dynamic range of both error and signal
- Physical units confirmed from NASA TEMPO documentation
- Relative RMSE ~8% indicates good reconstruction quality for compression ratio of 514×