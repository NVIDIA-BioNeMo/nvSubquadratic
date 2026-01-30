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

| Wandb ID | Model    | Hidden Dim | Batch Size | GPUs | Val Loss | Notes                           |
| -------- | -------- | ---------- | ---------- | ---- | -------- | ------------------------------- |
| eu3z4nwd | Hyena XS | 160        | 64 (16×4)  | 4    | 0.0487   | DDP, bf16-mixed, 50k iterations |
| 835xlrmz | Hyena XS | 160        | 64 (16×4)  | 1    | running  | Resumed from 31k (grad accum)   |
| n6k6b1u2 | Mamba XS | 96         | 64 (16×4)  | 4    | 0.522    | DDP, bf16-mixed, 50k iterations |
| 0nbqdnkh | Attn XS  | 160        | 64 (16×4)  | 4    | 0.884    | DDP, bf16-mixed, 50k iterations |

### Medium (M) Size Experiments

| Wandb ID | Model   | Hidden Dim | Batch Size | GPUs | Val Loss | Notes                            |
| -------- | ------- | ---------- | ---------- | ---- | -------- | -------------------------------- |
| hqjb6zi7 | Attn M  | 384        | 64 (8×8)   | 8    | running  | DDP, bf16-mixed, 50k iterations  |
| p45sg1q8 | Mamba M | 256        | 64 (8×8)   | 8    | running  | DDP, bf16-mixed, 50k iterations  |
| j3e6p849 | Hyena M | 416        | 64 (8×8)   | 8    | running  | DDP, bf16-mixed, chunked_fftconv |

______________________________________________________________________

## Notes

- Items can be placed on any depth slice (except overlapping with readout)
- Mask channel marks the target location in 3D space
- Model must learn to attend to the correct item based on mask
