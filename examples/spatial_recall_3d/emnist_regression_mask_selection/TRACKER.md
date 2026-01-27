# EMNIST Spatial Recall 3D - Mask Selection - Experiment Tracker

## Overview

3D spatial recall task with multiple items (target + distractors):

- Multiple 2D images placed on depth slices of a 3D volume
- Mask channel indicates which item is the target
- Must recall target at back-bottom-right corner

**Canvas dimensions:** 8 × 64 × 64 (D × H × W)
**Target size:** 16 × 16
**Number of items:** 4 (1 target + 3 distractors)

______________________________________________________________________

## Experiments

### Baseline Experiments

| Wandb ID | Model    | Hidden Dim | Val Loss | Notes            |
| -------- | -------- | ---------- | -------- | ---------------- |
| TBD      | Hyena XS | 160        | TBD      | Initial baseline |

______________________________________________________________________

## Notes

- Items can be placed on any depth slice (except overlapping with readout)
- Mask channel marks the target location in 3D space
- Model must learn to attend to the correct item based on mask
