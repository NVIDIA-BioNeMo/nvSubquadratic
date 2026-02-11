# Multi-Head Hyena Experiments - EMNIST Spatial Recall 2D (Color Conditioning)

## Overview

**Goal**: Test multi-head convolutions with dense \[head_dim × head_dim\] channel mixing within each head.

**Hypothesis**: Dense within-head channel mixing may improve feature learning vs depthwise (Hyena) while being more efficient than full dense (Attention).

**Task**: EMNIST Spatial Recall 2D Regression with Color Conditioning

- **Input**: 64×64 canvas with 4 EMNIST digits in RGB (3 channels) with colored bounding boxes
- **Target**: 16×16 RGB region containing the digit colored with its frame color

______________________________________________________________________

## Baselines

### Depthwise Hyena Baselines

| Config                 | Architecture    | Hidden | omega_0 | gc  | Norm      | Val Loss      | WandB ID | Notes                      |
| ---------------------- | --------------- | ------ | ------- | --- | --------- | ------------- | -------- | -------------------------- |
| ccnn_hyena_m           | Hyena Depthwise | 416    | 10      | 10  | LayerNorm | 0.0028        | tqbnoevm | Original baseline          |
| ccnn_hyena_m           | Hyena Depthwise | 416    | 10      | 1   | LayerNorm | 0.0050        | wupmxhxt | Debug with LayerStats      |
| ccnn_hyena_m           | Hyena Depthwise | 416    | 1       | 1   | LayerNorm | 0.0023        | tvysdwo0 | omega_0=1.0                |
| ccnn_hyena_m_groupnorm | Hyena Depthwise | 416    | 1       | 1   | GN(1)     | **0.0020** ⭐ | xhlhhvmf | **NEW SOTA** GN(1)+omega=1 |

**Key Finding**: GroupNorm(1) + omega_0=1.0 achieves **0.0020**, 29% better than original baseline!

______________________________________________________________________

## Multi-Head Experiments

### Initial Experiments (head_dim=16)

| Config                 | Hidden | Heads | head_dim | Val Loss   | WandB ID | Status  | Notes                             |
| ---------------------- | ------ | ----- | -------- | ---------- | -------- | ------- | --------------------------------- |
| ccnn_hyena_multihead_m | 384    | 24    | 16       | **0.0047** | 5ivig6hf | ✅ Done | v1: no init scaling               |
| ccnn_hyena_multihead_m | 384    | 24    | 16       | 0.0060     | ulec6rce | ✅ Done | v2: with 1/sqrt(head_dim) scaling |

**Observation**: head_dim=16 works but ~1.7x worse than depthwise baseline (0.0028).

______________________________________________________________________

### Stability Experiments (head_dim=32)

Testing larger head_dim=32 for potentially better feature learning.

| Config                 | Hidden | Heads | head_dim | Val Loss | WandB ID | Status      | Notes                           |
| ---------------------- | ------ | ----- | -------- | -------- | -------- | ----------- | ------------------------------- |
| ccnn_hyena_multihead_m | 352    | 11    | 32       | 0.0913   | 3vqbuyb4 | ❌ Unstable | v3: massive training spikes     |
| ccnn_hyena_multihead_m | 352    | 11    | 32       | -        | u5ijga2a | ✅ Done     | v4: + grad_clip=1.0             |
| ccnn_hyena_multihead_m | 352    | 11    | 32       | -        | 7c302k4a | ✅ Done     | v5: + larger SIREN (mlp=64)     |
| ccnn_hyena_multihead_m | 352    | 11    | 32       | 0.0946   | v2bo61em | ✅ Done     | v6: gc=1.0 + LayerStatsCallback |
| ccnn_hyena_multihead_m | 352    | 11    | 32       | 0.0306   | 15q4jlf7 | ✅ Done     | v7: seed=42, gc=1.0             |
| ccnn_hyena_multihead_m | 352    | 11    | 32       | -        | xdw5gozm | ❌ Cancel   | v8: lr=5e-5 (too slow)          |
| ccnn_hyena_multihead_m | 352    | 11    | 32       | 0.0750   | q1e0mgyq | ✅ Done     | v9: gc=0.5, still unstable      |
| ccnn_hyena_multihead_m | 352    | 11    | 32       | 0.0060   | xfjnfn9z | ✅ Done     | v10: **omega_0=1.0**, gc=1.0 ⭐ |

**Key Finding**: omega_0=1.0 is critical for stability with head_dim=32!

**Root Cause Analysis**:

- SIREN outputs 27x more values with head_dim=32 (11K vs 416 for depthwise)
- High omega_0 (10.0) creates high-frequency kernels that amplify instabilities
- Lowering omega_0 to 1.0 produces smoother kernels → stable training

______________________________________________________________________

### Normalization Experiments (omega_0=1.0, gc=1.0)

Testing different normalization strategies after fixing stability with omega_0=1.0:

| Config                           | Norm          | qk_norm | Val Loss      | WandB ID | Status  | Notes                          |
| -------------------------------- | ------------- | ------- | ------------- | -------- | ------- | ------------------------------ |
| ccnn_hyena_multihead_m           | LayerNorm     | True    | 0.0060        | xfjnfn9z | ✅ Done | v10: baseline with omega_0=1.0 |
| ccnn_hyena_multihead_m           | Identity      | False   | 0.0036        | m3r7p1os | ✅ Done | v11: no norms                  |
| ccnn_hyena_multihead_m_groupnorm | GroupNorm(11) | True    | **0.0031** ⭐ | q0n1mpoa | ✅ Done | v12: per-head norm             |
| ccnn_hyena_multihead_m_groupnorm | GroupNorm(11) | False   | 0.0059        | txca97up | ✅ Done | v13: per-head norm, no qk_norm |

**Key Findings**:

- **GroupNorm(11) + qk_norm** is best multi-head config (0.0031)
- Removing qk_norm hurts GroupNorm (0.0059 vs 0.0031)
- No norms (Identity) works surprisingly well (0.0036)
- LayerNorm is worst for multi-head (0.0060)

______________________________________________________________________

## Summary: Best Results

| Rank | Model                          | Val Loss      | Gap to Best | Notes               |
| ---- | ------------------------------ | ------------- | ----------- | ------------------- |
| 1    | Hyena Depthwise GN(1)+omega=1  | **0.0020** ⭐ | -           | NEW SOTA            |
| 2    | Hyena Depthwise omega=1        | 0.0023        | 1.15x       |                     |
| 3    | Hyena Depthwise (original)     | 0.0028        | 1.40x       | baseline            |
| 4    | **Hyena Multi-Head GN(11)+qk** | **0.0031** ⭐ | 1.55x       | **BEST MULTI-HEAD** |
| 5    | Hyena Multi-Head (no norms)    | 0.0036        | 1.80x       |                     |
| 6    | Hyena Multi-Head (hd=16)       | 0.0047        | 2.35x       |                     |

______________________________________________________________________

## Conclusions

1. **omega_0=1.0** is essential for stable multi-head training with larger head_dim
1. **GroupNorm(num_heads) + qk_norm** is the optimal normalization for multi-head
1. Multi-head with hd=32 (0.0031) approaches but doesn't beat depthwise (0.0020)
1. **Gap is only 1.55x** - multi-head is viable if dense channel mixing is needed
1. Original depthwise can be improved 29% with GN(1) + omega_0=1.0

______________________________________________________________________

**Last Updated**: 2026-01-28
**Status**: ✅ Complete. Multi-head stability solved. Best config: GN(11) + qk_norm + omega_0=1.0
