# EMNIST Spatial Recall 3D - Simple Copy - Experiment Tracker

## Overview

3D spatial recall task where a 2D image is placed on a depth slice of a 3D volume.
Fixed placement at front-top-left corner, must recall at back-bottom-right corner.

**Canvas dimensions:** 8 × 64 × 64 (D × H × W)
**Target size:** 16 × 16

______________________________________________________________________

## Experiments

### Baseline Experiments

| Wandb ID | Model    | Hidden Dim | Val Loss | Notes            |
| -------- | -------- | ---------- | -------- | ---------------- |
| TBD      | Hyena XS | 160        | TBD      | Initial baseline |

______________________________________________________________________

## Notes

- 3D implementation created as extension of 2D spatial recall
- Uses depth-first ordering for sequence flattening
- Readout region is at the last depth slice, bottom-right corner
