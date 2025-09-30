# Development Log - 2025-09-29 14:58
## Topic: Probe Analysis Setup and NO2 Data Quality Investigation

### Summary
Set up MLP probe analysis for L2 supervised VAE checkpoint and investigated NO2 data quality issues, discovering extreme outliers in training data.

### Tasks Completed

#### 1. Probe Analysis Configuration
- **Issue**: User wanted to run probe analysis on intermediate checkpoint (step 35000) from ongoing L2 supervised training
- **Solution**:
  - Created new config: `configs/analysis/mlp_probe_l2_supervised_35k.yaml`
  - Created corresponding bash script: `scripts/analysis/run_mlp_probe_l2_supervised_35k.sh`
  - Fixed `linear_probe_analysis.py` to handle L2 supervised model checkpoints with `strict=False` loading

- **Key Finding**: L2 supervised models have additional MLP heads that are "read-only" from latents - they don't affect reconstruction, only predict L2 products as an auxiliary task

#### 2. NO2 Data Quality Analysis
- **Created analysis scripts in `scratch/no2_analysis/`**:
  - `analyze_no2_stats.py`: Analyzes statistics across all 50 NO2 files
  - `visualize_outlier.py`: Deep dive into outlier file
  - `check_asinh_effect.py`: Examines how asinh normalization handles outliers

- **Key Findings**:
  - File 18 (`TEMPO_NO2_L2_V03_20250128T181454Z_S007G09.nc`) is a severe outlier
  - Contains physically impossible NO2 values: max of 148,613 × 10¹⁵ molec/cm²
  - This file is in the TRAINING set (tile 00012.pt)
  - 99.7% of pixels are normal, but 18 pixels have extreme values

#### 3. Normalization Impact Assessment
- **asinh normalization is handling outliers well**:
  - Original space: outlier is 148,613× larger than normal
  - After asinh: outlier is only 11.6× larger than normal
  - This compression prevents outliers from dominating the loss function

### Technical Details

#### Model Architecture Insights
The L2 supervised VAE (`VAEWithL2Supervision`) architecture:
```python
# Standard VAE path (unchanged)
posterior = self.vae.encode(x)
z = posterior.sample()
reconstruction = self.vae.decode(z)

# Additional L2 prediction (read-only from z)
l2_predictions = self.l2_head(z)
```
The L2 heads don't modify latents, allowing base VAE weights to be used independently.

#### Data Quality Statistics
- NO2 data across 50 files:
  - Mean: 0.60 × 10¹⁵ molec/cm²
  - Valid data fraction: 85% average
  - Outlier file has retrieval failures creating unphysical values

### Files Modified
- `src/scripts/linear_probe_analysis.py`: Added `strict=False` for checkpoint loading
- Created 3 analysis scripts in `scratch/no2_analysis/`
- New configs and scripts in appropriate directories

### Next Steps Recommendations
1. Consider adding data quality filters during tile preparation
2. Monitor probe analysis results to see if outliers affect learned representations
3. The asinh normalization choice appears optimal for handling these issues

### Notes
- Training can continue while probe analysis runs (read-only access)
- The extreme outliers are likely L2 retrieval failures, not measurement issues
- Current normalization strategy (asinh) is robust to these outliers