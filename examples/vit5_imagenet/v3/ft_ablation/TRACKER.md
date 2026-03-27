# v3 FiLM Finetuning Ablation

## Objective

Finetune the v3 FiLM pretrained checkpoint (Hyena CLS-row gated + FiLM
conditioning with `film_after_pos_embed`) and ablate key hyperparameters.
Building on v2 findings: cosine schedule, dp=0.15 baseline.

## Pretrained Checkpoint

| Run ID     | Architecture                                                                     | Top-1 (test) | Notes                                  |
| ---------- | -------------------------------------------------------------------------------- | ------------ | -------------------------------------- |
| `e1du5yky` | FiLM CLS-row gated (LAMB, direct/identity, film_after_pos_embed, global wd=0.05) | 81.5%        | 3 FiLM layers (2 hidden + 1 pos_embed) |

## Infrastructure

- **Nodes**: b65c909e-06, b65c909e-08 (4 jobs x 2 GPUs each = 8 slots)
- **Compile mode**: `max-autotune-no-cudagraphs` (CUDA graphs incompatible with pre-training validation + EMA weight swap)
- **Triton cache**: per-job (`TRITON_CACHE_DIR=/home/dwromero/.triton/cache_${SLURM_JOB_ID}`)
- **Containers**: isolated (no `--container-name`, prevents CUDA graph state corruption)
- **Iter speed**: ~3.6 it/s steady state (2502 steps/epoch, ~6h total per 25-epoch run)

## Callbacks

- `FiLMMonitorCallback`: logs gamma/beta stats, weight delta from init, input dependence, sin disruption (every 50 steps)
- `IterationSpeedCallback`: logs `perf/iter_per_sec`, `perf/samples_per_sec`, `perf/batch_time_ms` (every 10 steps)
- `LabeledEMAWeightAveraging`: decay=0.99996

______________________________________________________________________

## Wave 1 — Low LR baseline sweep (lr=3e-5)

### Fixed Recipe

- **LR**: 3e-5 (AdamW), **Scheduler**: Cosine, 5-epoch warmup (20% of 25 epochs)
- **Mixup / CutMix**: OFF, **Augmentation**: RandAugment (rand-m9-mstd0.5-inc1)
- **EMA**: 0.99996, **Loss**: SoftTargetCE, smoothing=0.1
- **Batch size**: 256/GPU x 2 GPUs = 512 effective, **Epochs**: 25, **Precision**: bf16-mixed

### Ablation Plan

| #   | Config         | LR       | WD       | DP       | FiLM WD   | FFT Backend  |
| --- | -------------- | -------- | -------- | -------- | --------- | ------------ |
| 1   | `baseline`     | 3e-5     | 0.05     | 0.15     | global    | torch_fft    |
| 2   | `film_wd_001`  | 3e-5     | 0.05     | 0.15     | **0.01**  | torch_fft    |
| 3   | `film_wd_0`    | 3e-5     | 0.05     | 0.15     | **0.0**   | torch_fft    |
| 4   | `subq_ops`     | 3e-5     | 0.05     | 0.15     | global    | **subq_ops** |
| 5   | `lr5e5`        | **5e-5** | 0.05     | 0.15     | global    | torch_fft    |
| 6   | `dp01_wd01`    | 3e-5     | **0.10** | **0.10** | global    | torch_fft    |
| 7   | `dp02`         | 3e-5     | 0.05     | **0.20** | global    | torch_fft    |
| 8   | `dp01_film001` | 3e-5     | 0.05     | **0.10** | **0.001** | torch_fft    |

### Results (cancelled at epoch ~13/25 — heavy overfitting across all runs)

| #   | Config       | SLURM | W&B ID   | Best Val Acc (EMA) | Best Epoch | Val Loss Trend | Notes                                                 |
| --- | ------------ | ----- | -------- | ------------------ | ---------- | -------------- | ----------------------------------------------------- |
| 1   | baseline     | 36753 | wz82vpat | 0.8168             | ~8         | ↑ after ep 8   | +0.17% over pretrain                                  |
| 2   | film_wd_001  | 36754 | vvciuyfw | 0.8167             | ~9         | ↑ after ep 9   | Identical to baseline                                 |
| 3   | film_wd_0    | 36755 | fevs91ob | 0.8169             | ~9         | ↑ after ep 9   | Identical to baseline                                 |
| 4   | subq_ops     | 36756 | 4trjdpps | 0.8168             | ~8         | ↑ after ep 8   | Same speed as torch_fft (FFT not bottleneck at 15x14) |
| 5   | lr5e5        | 36757 | nmmcvrwx | 0.8166             | ~8         | ↑ after ep 8   | Slightly worse; higher LR didn't help alone           |
| 6   | dp01_wd01    | 36758 | j29020if | 0.8168             | ~7         | ↑ after ep 7   | Lower reg → earlier overfitting                       |
| 7   | dp02         | 36759 | b76xq1ih | 0.8167             | ~9         | ↑ after ep 9   | Higher dp delayed overfitting slightly                |
| 8   | dp01_film001 | 36760 | qxqim5qy | 0.8167             | ~8         | ↑ after ep 8   | No benefit from relaxed FiLM WD                       |

### Wave 1 Conclusions

1. **LR=3e-5 is too conservative.** Val acc barely moved from pretrained 0.815 → peak ~0.817 (+0.2%). The model stays in a shallow local optimum near the pretrained weights.
1. **All regularization knobs are ineffective** at this LR — FiLM WD, drop path, global WD make no meaningful difference because the model isn't moving far enough.
1. **Overfitting starts at epoch 7-9** for all runs — val loss bottoms then rises while train loss keeps decreasing.
1. **subq_ops = no-op at 15x14 resolution** — FFT is not the bottleneck. Would matter at higher resolution.
1. **Need higher LR** to push the model further from the pretrained basin, with compensating regularization to prevent overfitting.

______________________________________________________________________

## Wave 2 — Higher LR + augmentation exploration (lr=1e-4 center)

### Motivation

Wave 1 showed that lr=3e-5 is too low to meaningfully improve over the pretrained checkpoint.
Wave 2 centers on **lr=1e-4** (3x higher) with **wd=0.1, dp=0.2** as the baseline regularization
(aligned with Mamba® / DeiT-III finetuning recipes). Key new axis: **data augmentation** —
the pretrained model saw mixup=0.8 + cutmix=1.0 + three-augment for 800 epochs; removing
all of that during finetuning likely contributed to overfitting.

### Fixed Recipe (same as wave 1 except LR/WD/DP/augmentation)

- **Scheduler**: Cosine, 5-epoch warmup (20% of 25 epochs)
- **EMA**: 0.99996, **Loss**: SoftTargetCE, smoothing=0.1
- **Batch size**: 256/GPU x 2 GPUs = 512 effective, **Epochs**: 25, **Precision**: bf16-mixed

### Ablation Plan

| #   | Config             | LR       | WD      | DP      | FiLM WD       | Mixup   | CutMix  | Three-aug | Rationale                          |
| --- | ------------------ | -------- | ------- | ------- | ------------- | ------- | ------- | --------- | ---------------------------------- |
| 1   | `lr1e4`            | **1e-4** | 0.05    | 0.15    | global        | 0       | 0       | No        | Higher LR baseline (min reg)       |
| 2   | `lr1e4_dp02_wd01`  | **1e-4** | **0.1** | **0.2** | global        | 0       | 0       | No        | Mamba®-style reg center            |
| 3   | `lr3e4_dp03_wd01`  | **3e-4** | **0.1** | **0.3** | global        | 0       | 0       | No        | Aggressive LR, heavy reg           |
| 4   | `lr1e4_mixup`      | **1e-4** | 0.1     | 0.2     | global        | **0.8** | **1.0** | No        | Re-enable pretrain-level mix       |
| 5   | `lr1e4_3aug`       | **1e-4** | 0.1     | 0.2     | global        | 0       | 0       | **Yes**   | Pretrain augmentation style        |
| 6   | `lr1e4_mixup_3aug` | **1e-4** | 0.1     | 0.2     | global        | **0.8** | **1.0** | **Yes**   | Full pretrain pipeline             |
| 7   | `lr1e4_film0`      | **1e-4** | 0.1     | 0.2     | **0 (no WD)** | 0       | 0       | No        | Free FiLM conditioning             |
| 8   | `lr1e4_wd03`       | **1e-4** | **0.3** | 0.2     | **0 (no WD)** | 0       | 0       | No        | Very strong backbone WD, free FiLM |

### Results

| #   | Config           | SLURM | W&B ID | Best Val Acc (EMA) | Best Epoch | Notes                                                                        |
| --- | ---------------- | ----- | ------ | ------------------ | ---------- | ---------------------------------------------------------------------------- |
| 1   | lr1e4            | 36767 | 5sel   | 0.817 (ep 3)       | 3          | **Cancelled ep 14**: val/acc=0.814, val_loss=0.731 rising (overfit, low reg) |
| 2   | lr1e4_dp02_wd01  | 36768 |        |                    |            |                                                                              |
| 3   | lr3e4_dp03_wd01  | 36769 | kxhi   | 0.817 (ep 2)       | 2          | **Cancelled ep 9**: val/acc=0.807, diverged (LR=3e-4 too high)               |
| 4   | lr1e4_mixup      | 36770 | ffh3   | 0.816 (ep 4)       | 4          | **Cancelled ep 9**: val/acc=0.814, val_loss=0.759 rising (mixup too heavy)   |
| 5   | lr1e4_3aug       | 36776 | jhty   | 0.816 (ep 5)       | 5          | **Cancelled ep 12**: val/acc=0.813, declining (wd=0.1 not enough)            |
| 6   | lr1e4_mixup_3aug | 36777 | k9nf   | 0.816 (ep 3)       | 3          | **Cancelled ep 8**: val/acc=0.814, val_loss rising (too much aug)            |
| 7   | lr1e4_film0      | 36773 |        |                    |            |                                                                              |
| 8   | lr1e4_wd03       | 36775 |        |                    |            |                                                                              |

### Wave 2 Replacements (mid-flight, replacing diverged/stagnant runs)

| #   | Config                  | LR         | WD      | DP      | FiLM WD       | Aug       | SLURM | Rationale                                                            |
| --- | ----------------------- | ---------- | ------- | ------- | ------------- | --------- | ----- | -------------------------------------------------------------------- |
| 3b  | `lr2e4_wd03`            | **2e-4**   | **0.3** | 0.2     | **0 (no WD)** | none      | 36786 | **Cancelled ep 11**: val/acc=0.809, overfit fast (LR still too high) |
| 1b  | `lr1e4_wd03_3aug`       | 1e-4       | **0.3** | 0.2     | **0 (no WD)** | three-aug | 36789 | Winning recipe + three-augment                                       |
| 4b  | `lr1e4_wd03_dp03`       | 1e-4       | **0.3** | **0.3** | **0 (no WD)** | none      | 36788 | Winning wd=0.3 recipe + stronger drop path                           |
| 5b  | `lr15e4_wd03`           | **1.5e-4** | **0.3** | 0.2     | **0 (no WD)** | none      | 36790 | LR bracket between 1e-4 and 2e-4                                     |
| 3c  | `lr15e4_wd03_dp03`      | **1.5e-4** | **0.3** | **0.3** | **0 (no WD)** | none      | 36793 | Best of both dp03 + lr1.5e-4                                         |
| 6b  | `lr1e4_3aug_wd02_film0` | 1e-4       | **0.2** | 0.2     | **0 (no WD)** | three-aug | 36787 | Free FiLM + moderate WD + light augmentation                         |

### Wave 2 Late Replacements (max-regularization exploration)

| #    | Config                 | LR         | WD      | DP      | FiLM WD       | Aug       | Epochs | SLURM | Rationale                                            |
| ---- | ---------------------- | ---------- | ------- | ------- | ------------- | --------- | ------ | ----- | ---------------------------------------------------- |
| 1c   | `lr1e4_wd03_dp03_3aug` | 1e-4       | **0.3** | **0.3** | **0 (no WD)** | three-aug | 25     | 36795 | Maximum regularization: all best strategies combined |
| 5c   | `lr15e4_wd03_3aug`     | **1.5e-4** | **0.3** | 0.2     | **0 (no WD)** | three-aug | 25     | 36797 | Faster LR + three-augment                            |
| 10ep | `lr1e4_wd03_10ep`      | 1e-4       | **0.3** | 0.2     | **0 (no WD)** | none      | **10** | 36800 | **Shorter training to capture peak**                 |

### 10-Epoch Run Result (key validation)

| Epoch      | val/acc_ema | val/loss_ema |
| ---------- | ----------- | ------------ |
| 0          | 0.8154      | 0.7364       |
| 1          | 0.8161      | 0.7276       |
| 3          | **0.8169**  | 0.7162       |
| 5          | **0.8169**  | 0.7126       |
| 6          | 0.8164      | **0.7122**   |
| 7          | 0.8157      | 0.7129       |
| 10 (final) | **0.8149**  | 0.7145       |

**Final vs Peak**: 0.8149 vs 0.8169 (−0.2pp). Compare with 25-epoch: ~0.809 final vs 0.817 peak (−0.8pp).
The 10-epoch run preserves **0.6pp more accuracy** at the end than the 25-epoch version.

### Wave 2 Conclusions (FINAL)

**Universal pattern**: All runs peak at val/acc_ema ≈ 0.817 around epoch 2-5, then overfit.
The difference is how long runs sustain near-peak accuracy:

| Regularization Level      | Runs                        | Peak Epoch | Overfit Start | val/acc at ep 12 | Final (ep 25) |
| ------------------------- | --------------------------- | ---------- | ------------- | ---------------- | ------------- |
| Low (wd≤0.1, dp≤0.2)      | lr1e4, lr1e4_dp02           | ep 3-4     | ep 7-8        | 0.813-0.814      | ~0.812        |
| Medium (wd=0.1-0.2 + aug) | lr1e4_3aug, 3aug_wd02_film0 | ep 5-6     | ep 10-12      | 0.812-0.813      | ~0.809        |
| High (wd=0.3 + free FiLM) | lr1e4_wd03                  | ep 5-7     | ep 10-12      | 0.813            | ~0.809        |
| **High + dp/aug**         | **wd03_dp03, wd03_3aug**    | ep 5-7     | **ep 12-14**  | **0.813-0.815**  | ~0.805        |
| Max reg (all combined)    | maxreg                      | ep 5-8     | ep 14-16      | 0.811            | ~0.798        |
| Too high LR               | lr2e4, lr3e4, lr15e4_dp03   | ep 2-3     | ep 6-8        | 0.809-0.811      | 0.794-0.809   |
| Too much aug              | lr1e4_mix, mixup_3aug       | ep 3-5     | —             | 0.814-0.815      | — (cancelled) |
| **10 epochs only**        | **lr1e4_wd03_10ep**         | ep 3-5     | **—**         | —                | **0.8149**    |

**Key findings**:

1. **Ceiling is 0.817 val/acc_ema** (+0.2pp over pretrained 0.815) — no recipe broke this
1. **10 epochs is the optimal training length**: final EMA model at 0.8149, only 0.2pp below peak
1. **Best sustained accuracy**: wd=0.3 + dp=0.2 + free FiLM for 10 epochs
1. **Free FiLM** consistently helps (or at least doesn't hurt) across all recipes
1. **LR=1e-4 is optimal** — 1.5e-4 is borderline, 2e-4+ overfits too fast
1. **Mixup/CutMix too heavy** for finetuning — prevents convergence in 25 epochs
1. **Three-augment is the most effective data regularizer** — lighter than mixup, extends the peak window by ~2 epochs
1. **More regularization delays but never prevents overfitting** — even the maximum combo (wd=0.3 + dp=0.3 + three-augment) collapses to 0.798 by epoch 23
1. **dp=0.3 is slightly too much with lr=1.5e-4** — lr15e4_wd03_dp03 collapsed to 0.794

**Recommended final recipe**: `lr=1e-4, wd=0.3, dp=0.2, film_wd=0 (free FiLM), 10 epochs, cosine schedule`

______________________________________________________________________

## Code Changes

### `film.py` — FiLM parameterization + per-parameter WD

- Added `film_parameterization` arg: `"direct"` (default, backward-compat) or `"residual"` (1+gamma)
- Added `no_weight_decay` arg: `False`=global WD, `True`=no WD, `float`=custom WD
  via `param._weight_decay` attribute
- Added `init_type`: `"identity"` (default) or `"small_random"`
- Added `gamma_max`: optional tanh bound on gamma deviation (requires residual)

### `kernels_nd.py` — `film_after_pos_embed`

- Added `film_after_pos_embed` flag to `SIRENKernelND`: first FiLM pair modulates
  the positional embedding output (after sine), remaining pairs modulate hidden layers.
  Requires `embedding_dim == mlp_hidden_dim` and `num_film_layers = num_layers`.

### `base_lightning_wrapper.py` — `_build_param_groups`

- Refactored optimizer construction to use `_build_param_groups()` which supports:
  - `param._no_weight_decay = True` -> WD=0
  - `param._weight_decay = <float>` -> custom WD group
  - Otherwise -> global WD from optimizer config

### Callbacks (new)

- `experiments/callbacks/film_monitor.py` — `FiLMMonitorCallback`: gamma/beta stats,
  weight delta from init, input dependence, sin disruption (ported from muon-optimizer branch
  commit 7bad43f, extended with delta tracking + film_after_pos_embed-aware analysis)
- `experiments/callbacks/iteration_speed.py` — `IterationSpeedCallback`: sliding-window
  iter/sec, samples/sec, batch_time_ms logging

______________________________________________________________________

## Wave 3 — Structural finetuning (LLRD + head re-init)

### Motivation

Wave 2 showed all recipes converge to the same 0.817 ceiling and overfit. The problem
isn't regularization strength — it's that all layers update at the same rate, corrupting
lower-level features that are already well-learned from 800 epochs of pretraining.

Wave 3 introduces **layer-wise learning rate decay (LLRD)** and **head re-initialization**
to break the ceiling:

- **LLRD**: scales LR per layer depth — head gets full LR, embedding layer gets `lr * decay^13`.
  Protects lower features while letting upper layers + head adapt freely.
- **Head re-init**: drops pretrained `out_proj` weights and re-learns the classification
  head from scratch. The pretrained head may be locally optimal for the pretrained features
  but suboptimal after finetuning shifts.

### Fixed Recipe (from Wave 2 winner)

- **LR**: 1e-4 (AdamW), **WD**: 0.3, **DP**: 0.2, **FiLM WD**: 0 (free)
- **Scheduler**: Cosine, 5-epoch warmup (20% of 10 epochs → 50% warmup for 10ep configs)
- **Epochs**: 10 (from Wave 2 conclusion: 25 epochs always overfits)
- **EMA**: 0.99996, **Loss**: SoftTargetCE, smoothing=0.1

### New code

- `_build_param_groups()` extended with `layer_decay` + `num_blocks` params for LLRD
- `DropKeysFromCheckpoint` callback: removes matching key prefixes from state_dict (for head re-init)
- `ExperimentConfig.layer_decay` / `num_blocks` fields for config-level LLRD control

### Ablation Plan

| #   | Config                | LLRD     | Re-init Head | LR       | Aug           | SLURM | Node | Rationale                         |
| --- | --------------------- | -------- | ------------ | -------- | ------------- | ----- | ---- | --------------------------------- |
| 1   | `llrd075`             | **0.75** | No           | 1e-4     | none          | 36843 | 06   | Standard LLRD (DeiT-III default)  |
| 2   | `llrd065`             | **0.65** | No           | 1e-4     | none          | 36844 | 06   | Aggressive LLRD — more protection |
| 3   | `llrd075_reinit`      | **0.75** | **Yes**      | 1e-4     | none          | 36845 | 06   | LLRD + fresh head                 |
| 4   | `reinit_head`         | None     | **Yes**      | 1e-4     | none          | 36846 | 06   | Head re-init only (control)       |
| 5   | `llrd075_3aug`        | **0.75** | No           | 1e-4     | **three-aug** | 36847 | 08   | LLRD + data regularization        |
| 6   | `llrd075_lr3e4`       | **0.75** | No           | **3e-4** | none          | 36848 | 08   | LLRD taming higher LR             |
| 7   | `llrd085`             | **0.85** | No           | 1e-4     | none          | 36849 | 08   | Mild LLRD bracket                 |
| 8   | `llrd075_reinit_3aug` | **0.75** | **Yes**      | 1e-4     | **three-aug** | 36850 | 08   | Maximum structural + data combo   |

### Results

| #   | Config              | SLURM | W&B ID   | Peak Val Acc (EMA) | Final val/acc_ema | Final val/loss_ema | Notes                              |
| --- | ------------------- | ----- | -------- | ------------------ | ----------------- | ------------------ | ---------------------------------- |
| 1   | llrd075             | 36843 | 3iy42zvh | 0.817 (ep 3-5)     | **0.8153**        | 0.7143             | **Best tied — +0.04pp vs no-LLRD** |
| 2   | llrd065             | 36844 | wexroxt9 | 0.817 (ep 3)       | 0.8145            | 0.7147             | Slightly worse — too aggressive    |
| 3   | llrd075_reinit      | 36845 | h55poqwe | 0.803 (ep 9)       | 0.8036            | 1.2759             | EMA dragged by fresh head          |
| 4   | reinit_head         | 36846 | 6w9tvrnz | 0.801 (ep 8)       | 0.8037            | 1.2730             | EMA dragged, no LLRD               |
| 5   | llrd075_3aug        | 36847 | 2ascbt4h | 0.816 (ep 4)       | 0.8139            | 0.7179             | Three-aug slightly hurt with LLRD  |
| 6   | llrd075_lr3e4       | 36848 | 676bwxus | 0.816 (ep 1)       | 0.8067            | 0.7501             | LR=3e-4 still overfits with LLRD   |
| 7   | llrd085             | 36849 | w2p695on | 0.817 (ep 3-5)     | **0.8153**        | 0.7147             | **Best tied — mild LLRD optimal**  |
| 8   | llrd075_reinit_3aug | 36850 | mjjuasdr | 0.799 (ep 9)       | 0.8008            | 1.2988             | EMA dragged, 3aug + reinit         |

### Wave 3 Conclusions

1. **LLRD improves final accuracy by +0.04pp** (0.8153 vs 0.8149 no-LLRD baseline), by reducing post-peak degradation
1. **LLRD 0.75 and 0.85 perform identically** — both optimal. 0.65 too aggressive, 0.85 slightly better loss
1. **Peak ceiling unchanged at 0.817** — LLRD doesn't break the ceiling, only sustains it longer
1. **Head re-initialization is counterproductive** — EMA of fresh head can't recover in 10 epochs
1. **LR=3e-4 still overfits** even with LLRD protection
1. **Three-augment + LLRD slightly worse** than LLRD alone in 10 epochs — augmentation slows convergence
1. **LLRD's main value**: at epoch 8, LLRD runs at 0.816 vs Wave 2 at 0.813 (+0.3pp retention)

______________________________________________________________________

## Wave 4 — LLRD + data augmentation exploration

### Motivation

Wave 3 confirmed LLRD 0.75 as beneficial but the 0.817 ceiling persists. Wave 4
explores whether combining LLRD with specific data augmentation strategies can
break through, and whether longer training (20 epochs) with LLRD sustains
improvement past epoch 10.

### Fixed Recipe

- **LLRD**: 0.75, **WD**: 0.3, **DP**: 0.2, **FiLM WD**: 0 (free)
- **LR**: 1e-4, **Scheduler**: Cosine, 5-epoch warmup
- **Epochs**: 10 (unless noted), **EMA**: 0.99996

### Ablation Plan

| #   | Config                | Epochs | Augmentation              | SLURM | Node | Rationale                                    |
| --- | --------------------- | ------ | ------------------------- | ----- | ---- | -------------------------------------------- |
| 1   | `llrd075_20ep`        | **20** | baseline RA               | 36853 | 06   | Extended training with LLRD protection       |
| 2   | `llrd075_lr15e4`      | 10     | baseline RA               | 36854 | 06   | Mid-range LR sweet spot                      |
| 3   | `llrd075_cutmix`      | 10     | **CutMix=1.0**            | 36855 | 06   | Spatial regularization (preserves structure) |
| 4   | `llrd075_lightmix`    | 10     | **mixup=0.3, cutmix=0.5** | 36856 | 06   | Light mix (pending, waiting for slot)        |
| 5   | `llrd075_strongra`    | 10     | **rand-m14**              | 36857 | 08   | Stronger RandAugment                         |
| 6   | `llrd075_erasing`     | 10     | **RE=0.25**               | 36858 | 08   | Random erasing regularization                |
| 7   | `llrd065_20ep`        | **20** | baseline RA               | —     | —    | Aggressive LLRD + long training              |
| 8   | `llrd075_3aug_cutmix` | 10     | **3aug + CutMix**         | —     | —    | Combined spatial+pretrain aug                |

### Results (10-epoch runs)

| #   | Config           | SLURM | W&B ID   | Final val/acc_ema | Final val/loss_ema | Notes                    |
| --- | ---------------- | ----- | -------- | ----------------- | ------------------ | ------------------------ |
| 2   | llrd075_lr15e4   | 36854 | zfdx4lev | 0.8146            | 0.7181             | Higher LR slightly worse |
| 3   | llrd075_cutmix   | 36855 | rlihc3g6 | 0.8142            | 0.7393             | CutMix hurts, high loss  |
| 5   | llrd075_strongra | 36857 | hvndo674 | **0.8152**        | 0.7138             | Strong RA ties baseline  |
| 6   | llrd075_erasing  | 36858 | 1qjalkq7 | 0.8148            | **0.7135**         | Best val_loss            |

### 20-epoch and remaining runs (in progress)

| #   | Config              | SLURM | Epochs | Notes            |
| --- | ------------------- | ----- | ------ | ---------------- |
| 1   | llrd075_20ep        | 36853 | 20     | Running (~ep 10) |
| 4   | llrd075_lightmix    | 36856 | 10     | Running (~ep 6)  |
| 7   | llrd065_20ep        | 36869 | 20     | Running (~ep 6)  |
| 8   | llrd075_3aug_cutmix | 36870 | 10     | Running (~ep 5)  |

### Wave 4b — Combo experiments (launched from freed slots)

| #   | Config                   | SLURM | Epochs | Aug              | Rationale                     |
| --- | ------------------------ | ----- | ------ | ---------------- | ----------------------------- |
| 9   | llrd075_strongra_20ep    | 36873 | **20** | RA m14           | Best aug + extended training  |
| 10  | llrd085_strongra         | 36874 | 10     | RA m14           | Best LLRD + best aug          |
| 11  | llrd075_strongra_erasing | 36875 | 10     | RA m14 + RE 0.25 | Double augmentation (pending) |
| 12  | llrd075_erasing_20ep     | 36876 | **20** | RE 0.25          | Best val_loss + extended      |
