# Probe Analysis Figures Generation and Paper Integration
**Date:** 2025-09-29 21:48
**Session Duration:** ~30 minutes
**Main Goal:** Generate comprehensive probe comparison figures and integrate them into paper

## Tasks Completed

### 1. Created Probe Figure Generation Script
- **Location:** `scratch/probing_figures/generate_probe_comparison_figures.py`
- **Functionality:**
  - Loads probe results from 4 analysis directories (linear/MLP × unsupervised/supervised)
  - Generates learning curve plots (2×2 grid for 4 products)
  - Generates scatter plots (2×2 grid for predicted vs ground truth)
  - Automated figure generation for both model types

### 2. Generated 8 Complete Figures
**Output directory:** `scratch/probing_figures/`

**Unsupervised Model (Base VAE):**
1. `unsupervised_model_linear_learning_curves.png` - Learning curves for linear probes
2. `unsupervised_model_mlp_learning_curves.png` - Learning curves for MLP probes
3. `unsupervised_model_linear_scatter.png` - Scatter plots for linear probe predictions
4. `unsupervised_model_mlp_scatter.png` - Scatter plots for MLP probe predictions

**L2-Supervised Model:**
5. `l2_supervised_model_linear_learning_curves.png` - Learning curves for linear probes
6. `l2_supervised_model_mlp_learning_curves.png` - Learning curves for MLP probes
7. `l2_supervised_model_linear_scatter.png` - Scatter plots for linear probe predictions
8. `l2_supervised_model_mlp_scatter.png` - Scatter plots for MLP probe predictions

### 3. Key Results from Analysis
**Performance Summary (R² scores):**

| Product | Base Linear | Base MLP | Supervised Linear | Supervised MLP |
|---------|-------------|----------|-------------------|----------------|
| NO₂     | 0.122       | 0.203    | 0.151             | 0.227          |
| O₃      | 0.545       | 0.811    | 0.521             | 0.815          |
| HCHO    | 0.474       | 0.511    | 0.469             | 0.500          |
| Cloud   | 0.785       | 0.930    | 0.778             | 0.922          |

**Key Findings:**
- MLP probes consistently outperform linear probes (especially O₃: 0.545→0.811, Cloud: 0.785→0.930)
- Minimal difference between unsupervised and supervised models
- NO₂ remains challenging across all methods (R² ≈ 0.20-0.23)
- HCHO shows little improvement from linear to MLP (~0.5 regardless)

### 4. Paper Integration

#### Main Text Figures (2 full-width figures)
**Added to `paper/main.tex` lines 339-360:**

1. **Figure: Unsupervised VAE Probing Results**
   - Files: `unsupervised_linear_scatter.png` + `unsupervised_mlp_scatter.png`
   - Layout: Side-by-side scatter plots (linear left, MLP right)
   - Location: Main text Section 4 (Results)

2. **Figure: L2-Supervised VAE Probing Results**
   - Files: `supervised_linear_scatter.png` + `supervised_mlp_scatter.png`
   - Layout: Side-by-side scatter plots (linear left, MLP right)
   - Location: Main text Section 4 (Results)

#### Appendix Figures (2 full-width figures)
**Added to Appendix "Additional Results" section, lines 592-608:**

3. **Figure: Unsupervised VAE Probe Training Dynamics**
   - Files: `unsupervised_linear_learning_curves.png` + `unsupervised_mlp_learning_curves.png`
   - Layout: Side-by-side learning curves (linear left, MLP right)
   - Spacing: 1cm horizontal space, 0.46 textwidth each panel

4. **Figure: L2-Supervised VAE Probe Training Dynamics**
   - Files: `supervised_linear_learning_curves.png` + `supervised_mlp_learning_curves.png`
   - Layout: Side-by-side learning curves (linear left, MLP right)
   - Spacing: 1cm horizontal space, 0.46 textwidth each panel

### 5. Added Detailed Text and Captions

#### Appendix Subsection: "Probe Training Dynamics"
- Comprehensive explanation of convergence behavior
- Linear probes: fast convergence (100 epochs)
- MLP probes: slower convergence (up to 2000 epochs)
- Product-specific patterns (NO₂/HCHO plateau then improve, O₃/Cloud steady)
- Key insight: unsupervised and supervised models perform nearly identically

#### Figure Captions
All captions include:
- Full description of what's shown
- Panel organization explanation
- Performance metrics (R² scores)
- Key insights and comparisons
- Cross-references between figures

### 6. File Organization
**Created new directory:** `paper/figures/probe_learning_curves/`

**Copied files:**
- `unsupervised_linear_learning_curves.png`
- `unsupervised_mlp_learning_curves.png`
- `supervised_linear_learning_curves.png`
- `supervised_mlp_learning_curves.png`
- `unsupervised_linear_scatter.png`
- `unsupervised_mlp_scatter.png`
- `supervised_linear_scatter.png`
- `supervised_mlp_scatter.png`

## Design Decisions

### Grouping Strategy
Initially grouped by probe type (linear vs MLP), but changed to **group by model** (unsupervised vs supervised) based on user feedback:
- Better for comparing linear vs MLP within same model
- More logical flow for assessing impact of supervision
- Each figure shows complete story for one model

### Figure Layout
- Used `figure*` (full-width) for all probe figures
- Side-by-side panels with explicit spacing (`\hspace{1cm}`)
- Reduced width (0.46 textwidth) to accommodate spacing
- Consistent color scheme across all figures (NO₂: purple, O₃: blue, HCHO: teal, Cloud: orange)

## Technical Notes

### Bug Fix
Initial script had incorrect npz key (`y_true` → `y_test`) when loading predictions. Fixed in line 46 of generation script.

### LaTeX Spacing Issue
Initial attempt with 0.48 textwidth + 0.5cm spacing was too tight. Final solution: 0.46 textwidth + 1cm spacing provides proper visual separation.

## Files Modified
1. `scratch/probing_figures/generate_probe_comparison_figures.py` (created)
2. `paper/main.tex` (lines 339-360, 583-608)
3. Created `paper/figures/probe_learning_curves/` directory
4. Copied 8 PNG files to paper/figures/

## Next Steps (if needed)
- Compile LaTeX to verify figure rendering
- Potentially adjust figure sizes based on compiled output
- Consider adding summary table comparing all probe results
- May need to add references to these figures in main text narrative

## Data Sources Used
All analysis used existing probe results from:
- `/n/home12/cfpark00/TEMPO/data/analysis/linear_probe_analysis/`
- `/n/home12/cfpark00/TEMPO/data/analysis/mlp_probe_analysis/`
- `/n/home12/cfpark00/TEMPO/data/analysis/linear_probe_analysis_l2_supervised/`
- `/n/home12/cfpark00/TEMPO/data/analysis/mlp_probe_l2_supervised/`

No model re-training was performed - all figures generated from pre-computed results.