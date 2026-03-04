# Task-002: Visualization Fixes & Interpretative Scaffolding

**Branch:** `feat/viz-clarity-improvements`
**Status:** `COMPLETED`
**Created by:** GEMINI
**Assigned to:** CLAUDE

## Objective
Implement high and medium priority clarity fixes across the 11 visualization panels in `scripts/visualize_hyena_kernels.py`. These fixes address numerical instability in ratio heatmaps, unreadable RGB-PCA projections, lack of frequency axes in FFTs, and generic outlier issues in global norm ranges.

## Scope (files allowed to modify)
- `scripts/visualize_hyena_kernels.py`

## Acceptance Criteria
- [x] **Panel 1 & 2:** `plot_kernel_norm_heatmaps` and `plot_raw_vs_masked` use 99th percentile clipping for `vmax` instead of absolute max.
- [x] **Panel 4:** `plot_spectral_analysis` clips or masks out the DC component and adds relative frequency axis labels ([-0.5, 0.5] cycles/pixel).
- [x] **Panel 6:** `plot_activation_maps` uses $log_2(\text{post}/\text{pre})$ for the ratio and centers the `RdBu_r` colormap at 0.
- [x] **Panel 9:** `plot_kernel_pca` is rewritten to use K-Means clustering (e.g., k=4) instead of PCA to RGB mapping, assigning a distinct categorical color to each spatial position based on its dominant mixing mode.
- [x] **Panel 10:** `plot_kernel_slices` selects off-diagonal channel pairs dynamically based on highest spatial energy/variance rather than hardcoded indices `(0, hd//4)`.

## Relevant Invariants
- INV-5: Public API signatures (`plot_*` function arguments) must not change without operator approval because they are called by the Gradio app. 
- PREF-3: Visualization tools use scientific colormaps.
- PREF-5: One function = one clear responsibility.

## Context / References
- See `audit_report.md` for a complete breakdown of why these changes are necessary.
- **Note on Panel 9:** You may need to import `sklearn.cluster.KMeans` or implement a simple pytorch-based greedy clustering if you don't want to add a dependency. See `RECAP_STATE` for dependency rules (INV-4).
