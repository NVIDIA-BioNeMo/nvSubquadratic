# Gaussian Mask Ablation Results

**Dataset:** `euler_multi_quadrants_periodicBC` (2D, 512×512, 5 fields, periodic BC)
**Model:** Hyena with `GaussianModulationND` mask on CKConv global convolution
**Architecture:** 12 blocks, 384 hidden channels, patch_size=16 (32×32 effective resolution)
**Training:** 110,000 iterations, cosine schedule with 5% warmup, bf16-mixed precision
**Baseline hyperparameters:** lr=1e-3, wd=1e-5, ω₀=30, `init_extent`=1.0

All runs use `GaussianModulationND` with attenuation-based initialization:

- `min_attenuation_at_step = 0.1` (mask value at ±1 grid step from center)
- `max_attenuation_at_limit = 0.95` (mask value at grid boundary)
- `init_extent` controls initial effective receptive field as a fraction of the full grid

______________________________________________________________________

## 1. Init Extent Ablation

Sweeps `init_extent` ∈ {0.25, 0.5, 0.75, 1.0} with lr=1e-3, wd=1e-5, ω₀=30.

| init_extent | test/loss    | val/best_loss | test/NRMSE  | val/NRMSE   | wandb                                                                         |
| ----------- | ------------ | ------------- | ----------- | ----------- | ----------------------------------------------------------------------------- |
| 0.25        | 0.002858     | 0.002952      | 0.02968     | 0.02924     | [jv1nf89k](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jv1nf89k) |
| 0.50        | 0.002853     | 0.002957      | 0.02980     | 0.02937     | [o1fnxgok](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o1fnxgok) |
| **0.75**    | **0.002815** | **0.002896**  | **0.02953** | **0.02910** | [o22jw4uh](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o22jw4uh) |
| 1.00        | 0.002858     | 0.002958      | 0.02977     | 0.02935     | [dbj7jsme](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dbj7jsme) |

**Takeaway:** `init_extent` has minimal impact on final performance. All values converge to similar test loss (~0.00285). `init_extent=0.75` is marginally best, but the differences are within noise.

______________________________________________________________________

## 2. ω₀ (SIREN frequency) Ablation

Sweeps ω₀ ∈ {1, 10, 30, 100} with init_extent=0.5, lr=1e-3, wd=1e-5.

| ω₀     | test/loss    | val/best_loss | test/NRMSE  | val/NRMSE   | wandb                                                                         |
| ------ | ------------ | ------------- | ----------- | ----------- | ----------------------------------------------------------------------------- |
| 1      | 0.002998     | 0.003070      | 0.03044     | 0.03000     | [ypss8ikh](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ypss8ikh) |
| 10     | 0.002951     | 0.003036      | 0.03024     | 0.02980     | [42svy7wf](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/42svy7wf) |
| **30** | **0.002853** | **0.002957**  | **0.02980** | **0.02937** | [o1fnxgok](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o1fnxgok) |
| 100    | 0.002943     | 0.003035      | 0.03014     | 0.02970     | [ljyptr1z](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ljyptr1z) |

**Takeaway:** ω₀=30 (default) is the best. Lower frequencies (1, 10) underperform, likely because the SIREN kernel lacks expressiveness at low frequencies. ω₀=100 is slightly worse, suggesting diminishing returns or optimization difficulty at very high frequencies. The spread is ~5% in test loss between worst (ω₀=1) and best (ω₀=30).

______________________________________________________________________

## 3. Learning Rate Ablation

Sweeps lr ∈ {3e-4, 1e-3, 3e-3, 4e-3} with init_extent=0.5, wd=1e-5, ω₀=30.

| lr       | test/loss    | val/best_loss | test/NRMSE  | val/NRMSE   | wandb                                                                         |
| -------- | ------------ | ------------- | ----------- | ----------- | ----------------------------------------------------------------------------- |
| 3e-4     | 0.003322     | 0.003447      | 0.03221     | 0.03176     | [qdsiw9q7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/qdsiw9q7) |
| **1e-3** | **0.002853** | **0.002957**  | **0.02980** | **0.02937** | [o1fnxgok](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o1fnxgok) |
| 3e-3     | 0.003464     | 0.003519      | 0.03322     | 0.03273     | [uy7m5qtw](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/uy7m5qtw) |
| 4e-3     | 0.003755     | 0.003807      | 0.03463     | 0.03410     | [qo09bade](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/qo09bade) |

**Takeaway:** lr=1e-3 (default) is clearly optimal. Both lower (3e-4) and higher (3e-3, 4e-3) learning rates perform significantly worse. The model is sensitive to lr — going from 1e-3 to 4e-3 increases test loss by 32%.

______________________________________________________________________

## 4. Weight Decay Ablation

Sweeps wd ∈ {0, 1e-6, 1e-5, 1e-4} with init_extent=0.5, lr=1e-3, ω₀=30.

| wd    | test/loss    | val/best_loss | test/NRMSE  | val/NRMSE   | wandb                                                                         |
| ----- | ------------ | ------------- | ----------- | ----------- | ----------------------------------------------------------------------------- |
| **0** | **0.002844** | **0.002938**  | **0.02968** | **0.02925** | [n9dyc97u](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n9dyc97u) |
| 1e-6  | *(running)*  | —             | —           | —           | —                                                                             |
| 1e-5  | 0.002853     | 0.002957      | 0.02980     | 0.02937     | [o1fnxgok](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o1fnxgok) |
| 1e-4  | 0.002899     | 0.002986      | 0.02998     | 0.02955     | [1j7c7skl](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/1j7c7skl) |

**Takeaway:** Weight decay has a small but consistent effect. wd=0 is marginally best, wd=1e-5 (default) is very close, and wd=1e-4 is slightly worse. The model is not very sensitive to weight decay in this range.

______________________________________________________________________

## Summary: All Runs Ranked by Test Loss

| Rank | Run    | init_extent | ω₀  | lr   | wd   | test/loss    | test/NRMSE  |
| ---- | ------ | ----------- | --- | ---- | ---- | ------------ | ----------- |
| 1    | e075   | 0.75        | 30  | 1e-3 | 1e-5 | **0.002815** | **0.02953** |
| 2    | wd0    | 0.50        | 30  | 1e-3 | 0    | 0.002844     | 0.02968     |
| 3    | e050   | 0.50        | 30  | 1e-3 | 1e-5 | 0.002853     | 0.02980     |
| 4    | e025   | 0.25        | 30  | 1e-3 | 1e-5 | 0.002858     | 0.02968     |
| 5    | e100   | 1.00        | 30  | 1e-3 | 1e-5 | 0.002858     | 0.02977     |
| 6    | wd1e4  | 0.50        | 30  | 1e-3 | 1e-4 | 0.002899     | 0.02998     |
| 7    | w0-100 | 0.50        | 100 | 1e-3 | 1e-5 | 0.002943     | 0.03014     |
| 8    | w0-10  | 0.50        | 10  | 1e-3 | 1e-5 | 0.002951     | 0.03024     |
| 9    | w0-1   | 0.50        | 1   | 1e-3 | 1e-5 | 0.002998     | 0.03044     |
| 10   | lr3e4  | 0.50        | 30  | 3e-4 | 1e-5 | 0.003322     | 0.03221     |
| 11   | lr3e3  | 0.50        | 30  | 3e-3 | 1e-5 | 0.003464     | 0.03322     |
| 12   | lr4e3  | 0.50        | 30  | 4e-3 | 1e-5 | 0.003755     | 0.03463     |
| 13   | wd1e6  | 0.50        | 30  | 1e-3 | 1e-6 | *(running)*  | —           |

______________________________________________________________________

## 5. Mask Parameter Convergence Analysis

Extracted from the final checkpoints of the init_extent ablation runs.
Each mask has `std_param` of shape `[2, 384]` (2 spatial dims × 384 channels) per block (12 blocks total = 9,216 parameters per run).

**Grid geometry:** 31 points in \[-1, 1\], step = 0.0667, half-grid = 15 steps from center to boundary.
**Clamp bounds:** min_std = 0.031 (mask = 0.1 at ±1 step), max_std = 3.12 (mask = 0.95 at boundary).

### Per-block median σ and effective receptive field

ERF = number of grid steps from center where 1D mask > 0.5. The full half-grid is 15 steps.

| Block | init_extent=0.25 | init_extent=0.50 | init_extent=0.75 | init_extent=1.00 |
| :---: | :--------------: | :--------------: | :--------------: | :--------------: |
|   0   |  0.115 (2.0/15)  |  0.114 (2.0/15)  |  0.109 (1.9/15)  |  0.113 (2.0/15)  |
|   1   |  0.079 (1.4/15)  |  0.105 (1.9/15)  |  0.106 (1.9/15)  |  0.112 (2.0/15)  |
|   2   |  0.090 (1.6/15)  |  0.094 (1.7/15)  |  0.102 (1.8/15)  |  0.110 (1.9/15)  |
|   3   |  0.093 (1.6/15)  |  0.096 (1.7/15)  |  0.104 (1.8/15)  |  0.097 (1.7/15)  |
|   4   |  0.083 (1.5/15)  |  0.095 (1.7/15)  |  0.100 (1.8/15)  |  0.105 (1.8/15)  |
|   5   |  0.081 (1.4/15)  |  0.089 (1.6/15)  |  0.099 (1.7/15)  |  0.106 (1.9/15)  |
|   6   |  0.092 (1.6/15)  |  0.102 (1.8/15)  |  0.102 (1.8/15)  |  0.105 (1.9/15)  |
|   7   |  0.083 (1.5/15)  |  0.092 (1.6/15)  |  0.098 (1.7/15)  |  0.097 (1.7/15)  |
|   8   |  0.091 (1.6/15)  |  0.098 (1.7/15)  |  0.106 (1.9/15)  |  0.108 (1.9/15)  |
|   9   |  0.080 (1.4/15)  |  0.089 (1.6/15)  |  0.093 (1.6/15)  |  0.095 (1.7/15)  |
|  10   |  0.079 (1.4/15)  |  0.090 (1.6/15)  |  0.093 (1.6/15)  |  0.096 (1.7/15)  |
|  11   |  0.077 (1.4/15)  |  0.088 (1.6/15)  |  0.093 (1.6/15)  |  0.094 (1.7/15)  |

### Overall σ distribution

|            | 0.25  | 0.50  | 0.75  | 1.00  |
| ---------: | :---: | :---: | :---: | :---: |
|        min | 0.031 | 0.031 | 0.031 | 0.031 |
|        p10 | 0.039 | 0.041 | 0.042 | 0.041 |
|     median | 0.086 | 0.096 | 0.101 | 0.103 |
|        p90 | 0.151 | 0.161 | 0.170 | 0.182 |
|        max | 0.876 | 0.846 | 0.842 | 0.878 |
|       mean | 0.094 | 0.101 | 0.108 | 0.116 |
| % at clamp | 3.0%  | 2.6%  | 2.4%  | 2.5%  |

### Init σ vs converged σ

| init_extent | init median | converged median |     ratio      |
| :---------: | :---------: | :--------------: | :------------: |
|    0.25     |    0.060    |      0.086       |  1.43× (grew)  |
|    0.50     |    0.085    |      0.096       |  1.13× (grew)  |
|    0.75     |    0.104    |      0.101       | 0.97× (stayed) |
|    1.00     |    0.120    |      0.103       | 0.86× (shrank) |

### Anisotropy (e075 — best run)

| Block | med σ (d0) | med σ (d1) | d0/d1 | corr(d0,d1) |
| :---: | :--------: | :--------: | :---: | :---------: |
|   0   |   0.112    |   0.108    | 1.03  |    0.85     |
|   1   |   0.105    |   0.109    | 0.97  |    0.37     |
|   2   |   0.098    |   0.106    | 0.92  |    0.48     |
|   3   |   0.103    |   0.105    | 0.98  |    0.18     |
|   4   |   0.102    |   0.098    | 1.04  |    0.42     |
|   5   |   0.096    |   0.100    | 0.97  |    0.23     |
|   6   |   0.105    |   0.098    | 1.07  |    0.27     |
|   7   |   0.104    |   0.091    | 1.13  |    0.37     |
|   8   |   0.115    |   0.098    | 1.17  |    0.05     |
|   9   |   0.096    |   0.088    | 1.09  |    0.36     |
|  10   |   0.089    |   0.098    | 0.91  |    0.32     |
|  11   |   0.098    |   0.090    | 1.09  |    0.27     |

### Mask convergence insights

1. **All init_extents converge to a common operating range** (median σ ≈ 0.086–0.103). Smaller inits grow, larger inits shrink, meeting near σ ≈ 0.095–0.10. This explains why `init_extent` has negligible impact on final performance.

1. **The effective receptive field is very compact** — the median channel's mask at 50% threshold covers only ~1.5–2.0 grid steps out of 15 (the half-grid). The mask makes the SIREN kernels highly local.

1. **Block 0 is consistently the widest** (σ ≈ 0.11, ERF ≈ 2.0 steps) across all init_extents. Deeper blocks prefer narrower masks (σ ≈ 0.08–0.10). This layer-wise hierarchy suggests early layers need broader spatial context.

1. **Very few channels saturate at the minimum clamp** (~2.5–3.0%), and the maximum clamp is never approached (max observed σ ≈ 0.88 vs clamp at 3.12). The learned widths genuinely prefer the compact range.

1. **Negligible anisotropy**: d0/d1 ratios range from 0.91–1.17 with no consistent bias. Correlations between dimensions vary widely (0.05–0.85), with block 0 showing the strongest coupling. The dataset's spatial symmetry is reflected in the learned masks.

______________________________________________________________________

## Key Findings

1. **Learning rate is the most sensitive hyperparameter.** lr=1e-3 is clearly optimal; deviating by 3× in either direction degrades test loss by 16-32%.
1. **ω₀=30 is the sweet spot for SIREN frequency.** Lower frequencies lack expressiveness; ω₀=100 shows slight degradation, possibly due to optimization difficulty.
1. **Init extent has negligible impact.** The Gaussian mask's initial receptive field converges to similar final performance regardless of initialization (0.25–1.0). Mask parameter analysis confirms all runs converge to a common σ ≈ 0.095–0.10 operating point.
1. **Weight decay is not critical.** wd=0 and wd=1e-5 perform nearly identically. Higher wd=1e-4 marginally hurts. The model does not strongly benefit from regularization at this scale.
1. **The learned masks are highly local** — the effective receptive field covers only ~10–13% of the half-grid per dimension. Block 0 is consistently the widest (broadest spatial context).
1. **Best configuration:** init_extent=0.75, ω₀=30, lr=1e-3, wd=1e-5 (test loss = 0.002815, NRMSE = 0.02953). However, the margin over the default (init_extent=1.0) is small.
