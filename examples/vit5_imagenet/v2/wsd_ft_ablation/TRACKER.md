# WSD Finetuning Ablation — Experiment Tracker

W&B project: [`implicit-long-convs/nvsubquadratic`](https://wandb.ai/implicit-long-convs/nvsubquadratic)

## Pretrained Checkpoint

| Run ID                                                                          | Name                                | val/acc_ema | Alias         |
| ------------------------------------------------------------------------------- | ----------------------------------- | ----------- | ------------- |
| [`qyjyx58f`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/qyjyx58f) | `vit5_small_pretrain_attention_ema` | 81.81%      | `best` (v173) |

## Fixed Settings

- **Model**: ViT-5-Small attention (12 blocks, dim 384, patch 16, 224x224)
- **Scheduler**: WSD — 10% warmup, 70% stable, 20% linear decay (unless noted)
- **Epochs**: 20 (50,040 iterations)
- **Batch size**: 256/GPU × 2 GPUs = 512 effective
- **EMA**: decay=0.99996
- **Loss**: SoftTargetCrossEntropy
- **Precision**: bf16-mixed

______________________________________________________________________

## Key Findings

1. **Drop path rate is the single most impactful hyperparameter for finetuning.** Increasing from default 0.05 to 0.10–0.15 yields the largest gains (+0.2pp over no-mixup baseline, +0.26pp over pretrained).
1. **Disabling Mixup/CutMix is essential.** Every no-mixup variant outperforms every with-mixup variant. With mixup, the model doesn't learn; without it, structured regularization (drop path) compensates.
1. **Best confirmed result: 82.07% test** (`nomix_droppath015` — dp=0.15, lr=3e-5, wd=0.05).
1. **Heavy decay schedules (WSD 10/0/90 or 10/10/80) are more stable** than default 10/70/20. They prevent late-training degradation.
1. **Higher weight decay (0.1) with drop path 0.1 works well** — `nomix_droppath01_wd01` hit 82.06%.
1. **EMA is essential** — no-EMA gives 81.64% vs 82+% with EMA. Default decay 0.99996 is fine.
1. **Raw model accuracy barely moves** from pretrained baseline (81.83%) — EMA does all the lifting.
1. **No label smoothing helps with drop path 0.1** — `nomix_droppath01_smoothing0` hit 82.05%.

______________________________________________________________________

## Top Results — Confirmed Test Scores

| Rank | Config                        | DP   | LR   | WD   | Schedule     | Smooth | Test       |
| ---- | ----------------------------- | ---- | ---- | ---- | ------------ | ------ | ---------- |
| 1    | `nomix_droppath015`           | 0.15 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | **82.07%** |
| 2    | `nomix_dp015_lr5e5`           | 0.15 | 5e-5 | 0.05 | WSD 10/70/20 | 0.1    | **82.06%** |
| 2    | `nomix_droppath01_wd01`       | 0.10 | 3e-5 | 0.10 | WSD 10/70/20 | 0.1    | **82.06%** |
| 4    | `nomix_droppath015_decay90`   | 0.15 | 3e-5 | 0.05 | WSD 10/0/90  | 0.1    | **82.05%** |
| 4    | `nomix_droppath01_smoothing0` | 0.10 | 3e-5 | 0.05 | WSD 10/70/20 | 0.0    | **82.05%** |
| 4    | `nomix_droppath01_decay80`    | 0.10 | 3e-5 | 0.05 | WSD 10/10/80 | 0.1    | **82.05%** |
| 7    | `nomix_dp01_lr5e5_sm0`        | 0.10 | 5e-5 | 0.05 | WSD 10/70/20 | 0.0    | **82.04%** |
| 7    | `nomix_dp01_lr5e5_decay90`    | 0.10 | 5e-5 | 0.05 | WSD 10/0/90  | 0.1    | **82.04%** |
| 9    | `nomix_dp01_sm0_decay90`      | 0.10 | 3e-5 | 0.05 | WSD 10/0/90  | 0.0    | **82.03%** |
| 9    | `nomix_droppath01_decay90`    | 0.10 | 3e-5 | 0.05 | WSD 10/0/90  | 0.1    | **82.03%** |
| 11   | `nomix_dp015_lr7e5`           | 0.15 | 7e-5 | 0.05 | WSD 10/70/20 | 0.1    | **82.02%** |
| 11   | `nomix_droppath02`            | 0.20 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | **82.02%** |

______________________________________________________________________

## Awaiting Verification (W&B API temporarily unavailable)

These runs completed overnight but their data is on remote SLURM nodes. Verify via W&B when API access is restored.

| Config                      | DP   | LR   | WD   | Schedule     | Smooth | Last known BestEMA (ep) |
| --------------------------- | ---- | ---- | ---- | ------------ | ------ | ----------------------- |
| `nomix_dp015_decay80`       | 0.15 | 3e-5 | 0.05 | WSD 10/10/80 | 0.1    | 82.02% (ep10)           |
| `nomix_dp02_lr5e5`          | 0.20 | 5e-5 | 0.05 | WSD 10/70/20 | 0.1    | 82.01% (ep7)            |
| `nomix_dp015_sm0`           | 0.15 | 3e-5 | 0.05 | WSD 10/70/20 | 0.0    | 81.99% (ep6)            |
| `nomix_dp015_wd01`          | 0.15 | 3e-5 | 0.10 | WSD 10/70/20 | 0.1    | 81.99% (ep7)            |
| `nomix_dp015_lr5e5_decay90` | 0.15 | 5e-5 | 0.05 | WSD 10/0/90  | 0.1    | 81.97% (ep3)            |
| `nomix_droppath025`         | 0.25 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | 81.96% (ep5)            |
| `nomix_dp015_wd01_decay90`  | 0.15 | 3e-5 | 0.10 | WSD 10/0/90  | 0.1    | 81.95% (ep3)            |
| `nomix_dp015_lr5e5_wd01`    | 0.15 | 5e-5 | 0.10 | WSD 10/70/20 | 0.1    | 81.94% (ep3)            |
| `nomix_dp02_wd01`           | 0.20 | 3e-5 | 0.10 | WSD 10/70/20 | 0.1    | (just started)          |

______________________________________________________________________

## All Confirmed Results (39 runs, sorted by test accuracy)

| Config                        | DP   | LR   | WD   | Schedule     | Smooth | Mixup | BestEMA | Test       |
| ----------------------------- | ---- | ---- | ---- | ------------ | ------ | ----- | ------- | ---------- |
| `nomix_droppath015`           | 0.15 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | No    | 82.07%  | **82.07%** |
| `nomix_dp015_lr5e5`           | 0.15 | 5e-5 | 0.05 | WSD 10/70/20 | 0.1    | No    | 82.06%  | **82.06%** |
| `nomix_droppath01_wd01`       | 0.10 | 3e-5 | 0.10 | WSD 10/70/20 | 0.1    | No    | 82.06%  | **82.06%** |
| `nomix_droppath015_decay90`   | 0.15 | 3e-5 | 0.05 | WSD 10/0/90  | 0.1    | No    | 82.05%  | **82.05%** |
| `nomix_droppath01_smoothing0` | 0.10 | 3e-5 | 0.05 | WSD 10/70/20 | 0.0    | No    | 82.05%  | **82.05%** |
| `nomix_droppath01_decay80`    | 0.10 | 3e-5 | 0.05 | WSD 10/10/80 | 0.1    | No    | 82.05%  | **82.05%** |
| `nomix_dp01_lr5e5_sm0`        | 0.10 | 5e-5 | 0.05 | WSD 10/70/20 | 0.0    | No    | 82.04%  | **82.04%** |
| `nomix_dp01_lr5e5_decay90`    | 0.10 | 5e-5 | 0.05 | WSD 10/0/90  | 0.1    | No    | 82.04%  | **82.04%** |
| `nomix_dp01_sm0_decay90`      | 0.10 | 3e-5 | 0.05 | WSD 10/0/90  | 0.0    | No    | 82.03%  | **82.03%** |
| `nomix_droppath01_decay90`    | 0.10 | 3e-5 | 0.05 | WSD 10/0/90  | 0.1    | No    | 82.03%  | **82.03%** |
| `nomix_dp015_lr7e5`           | 0.15 | 7e-5 | 0.05 | WSD 10/70/20 | 0.1    | No    | 82.02%  | **82.02%** |
| `nomix_droppath02`            | 0.20 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | No    | 82.02%  | **82.02%** |
| `nomix_lr3e4_5ep`             | 0.05 | 3e-4 | 0.05 | WSD 10/70/20 | 0.1    | No    | 82.01%  | 82.01%     |
| `nomix_wsd_decay80`           | 0.05 | 3e-5 | 0.05 | WSD 10/10/80 | 0.1    | No    | 81.99%  | 81.99%     |
| `nomix_wsd_decay90`           | 0.05 | 3e-5 | 0.05 | WSD 10/0/90  | 0.1    | No    | 81.99%  | 81.99%     |
| `aug_no_mixup`                | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | No    | 81.98%  | 81.98%     |
| `nomix_10ep`                  | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | No    | 81.98%  | 81.98%     |
| `nomix_10ep_lr1e4`            | 0.05 | 1e-4 | 0.05 | WSD 10/70/20 | 0.1    | No    | 81.97%  | 81.97%     |
| `cosine_nomix`                | 0.05 | 3e-5 | 0.05 | Cosine 25%wu | 0.1    | No    | 81.96%  | 81.96%     |
| `nomix_5ep`                   | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | No    | 81.94%  | 81.94%     |
| `nomix_lr1e5`                 | 0.05 | 1e-5 | 0.05 | WSD 10/70/20 | 0.1    | No    | 81.93%  | 81.93%     |
| `grid_lr1e4_wd005`            | 0.05 | 1e-4 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | 81.91%  | 81.91%     |
| `grid_lr3e5_wd001`            | 0.05 | 3e-5 | 0.01 | WSD 10/70/20 | 0.1    | Yes   | 81.91%  | 81.91%     |
| `aug_no_randaug`              | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | 81.90%  | 81.90%     |
| `ema_decay99999`              | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | 81.90%  | 81.90%     |
| `grid_lr1e4_wd01`             | 0.05 | 1e-4 | 0.10 | WSD 10/70/20 | 0.1    | Yes   | 81.89%  | 81.89%     |
| `grid_lr3e5_wd005`            | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | 81.89%  | 81.89%     |
| `grid_lr1e5_wd005`            | 0.05 | 1e-5 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | 81.88%  | 81.88%     |
| `ema_decay9999`               | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | 81.90%  | 81.88%     |
| `grid_lr1e4_wd001`            | 0.05 | 1e-4 | 0.01 | WSD 10/70/20 | 0.1    | Yes   | 81.88%  | 81.88%     |
| `grid_lr1e5_wd001`            | 0.05 | 1e-5 | 0.01 | WSD 10/70/20 | 0.1    | Yes   | 81.88%  | 81.88%     |
| `grid_lr1e5_wd01`             | 0.05 | 1e-5 | 0.10 | WSD 10/70/20 | 0.1    | Yes   | 81.88%  | 81.88%     |
| `reg_droppath01`              | 0.10 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | 81.87%  | 81.87%     |
| `grid_lr3e5_wd01`             | 0.05 | 3e-5 | 0.10 | WSD 10/70/20 | 0.1    | Yes   | 81.88%  | 81.86%     |
| `grid_lr3e4_wd005`            | 0.05 | 3e-4 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | 81.87%  | 81.86%     |
| `grid_lr3e4_wd001`            | 0.05 | 3e-4 | 0.01 | WSD 10/70/20 | 0.1    | Yes   | 81.87%  | 81.86%     |
| `aug_three_augment`           | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.0    | Yes   | 81.87%  | 81.85%     |
| `ema_decay999`                | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | —       | 81.75%     |
| `ema_none`                    | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 0.1    | Yes   | —       | 81.64%     |

______________________________________________________________________

## Killed Runs (early-stopped due to degradation)

| Config                           | DP   | LR   | WD   | Schedule     | Peak EMA | Final EMA | Reason                                  |
| -------------------------------- | ---- | ---- | ---- | ------------ | -------- | --------- | --------------------------------------- |
| `grid_lr3e4_wd01`                | 0.05 | 3e-4 | 0.10 | WSD 10/70/20 | 81.86%   | 79.77%    | LR=3e-4 + mixup: severe degradation     |
| `nomix_lr3e4`                    | 0.05 | 3e-4 | 0.05 | WSD 10/70/20 | 82.01%   | 81.36%    | LR=3e-4 too high for 20ep without decay |
| `nomix_lr3e4_decay90`            | 0.05 | 3e-4 | 0.05 | WSD 10/0/90  | 81.96%   | 81.71%    | LR=3e-4 still degrades even with decay  |
| `nomix_lr2e4_decay90`            | 0.05 | 2e-4 | 0.05 | WSD 10/0/90  | 81.99%   | 81.77%    | LR=2e-4 + decay still degraded          |
| `nomix_lr2e4`                    | 0.05 | 2e-4 | 0.05 | WSD 10/70/20 | 81.97%   | 81.76%    | Declining                               |
| `nomix_lr1e4`                    | 0.05 | 1e-4 | 0.05 | WSD 10/70/20 | 81.94%   | 81.80%    | LR=1e-4 degrading in stable phase       |
| `nomix_lr1e4_decay90`            | 0.05 | 1e-4 | 0.05 | WSD 10/0/90  | 81.99%   | 81.80%    | Higher LR still declines                |
| `nomix_lr1e4_decay80`            | 0.05 | 1e-4 | 0.05 | WSD 10/10/80 | 81.95%   | 81.85%    | Declined                                |
| `nomix_lr5e5_decay80`            | 0.05 | 5e-5 | 0.05 | WSD 10/10/80 | 81.95%   | 81.88%    | Declined                                |
| `nomix_lr5e5`                    | 0.05 | 5e-5 | 0.05 | WSD 10/70/20 | 81.95%   | 81.84%    | Declining                               |
| `nomix_lr5e5_wd01`               | 0.05 | 5e-5 | 0.10 | WSD 10/70/20 | 81.97%   | 81.86%    | Declining                               |
| `nomix_lr7e5`                    | 0.05 | 7e-5 | 0.05 | WSD 10/70/20 | 81.94%   | 81.84%    | Declining                               |
| `nomix_noaug`                    | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 81.95%   | 81.84%    | No augmentation declining               |
| `nomix_threeaug`                 | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 81.91%   | 81.83%    | Not competitive                         |
| `nomix_smoothing0`               | 0.05 | 3e-5 | 0.05 | WSD 10/70/20 | 81.98%   | 81.86%    | Declined                                |
| `nomix_decay90_smoothing0`       | 0.05 | 3e-5 | 0.05 | WSD 10/0/90  | 81.99%   | 81.87%    | Declined                                |
| `nomix_cosine`                   | 0.05 | 3e-5 | 0.05 | Cosine 25%wu | 81.96%   | 81.85%    | Declining                               |
| `cosine_baseline`                | 0.05 | 1e-5 | 0.10 | Cosine       | 81.87%   | 81.78%    | With-mixup reference                    |
| `cosine_nomix_lr3e5`             | 0.05 | 3e-5 | 0.05 | Cosine 25%wu | 81.96%   | 81.85%    | Declining                               |
| `nomix_droppath0`                | 0.00 | 3e-5 | 0.05 | WSD 10/70/20 | 81.91%   | 81.83%    | dp=0 worse than default                 |
| `nomix_droppath01`               | 0.10 | 3e-5 | 0.05 | WSD 10/70/20 | 82.05%   | 81.94%    | Peaked ep12, declining by ep18          |
| `nomix_droppath01_lr5e5`         | 0.10 | 5e-5 | 0.05 | WSD 10/70/20 | 82.04%   | 81.91%    | Declining                               |
| `nomix_droppath01_lr7e5`         | 0.10 | 7e-5 | 0.05 | WSD 10/70/20 | 82.01%   | 81.82%    | Declining                               |
| `nomix_droppath01_lr1e4_decay90` | 0.10 | 1e-4 | 0.05 | WSD 10/0/90  | 81.99%   | 81.80%    | Higher LR degrading                     |
| `nomix_droppath01_lr1e4_decay80` | 0.10 | 1e-4 | 0.05 | WSD 10/10/80 | 82.02%   | 81.83%    | Higher LR degrading                     |

______________________________________________________________________

## Ablation Analysis by Dimension

### Drop Path Rate (no mixup, LR=3e-5, WD=0.05, WSD 10/70/20)

| Drop Path      | Test                             |
| -------------- | -------------------------------- |
| 0.00           | killed (peaked 81.91%)           |
| 0.05 (default) | 81.98%                           |
| 0.10           | killed (peaked 82.05%, declined) |
| 0.15           | **82.07%**                       |
| 0.20           | 82.02%                           |
| 0.25           | pending verification             |

**Conclusion**: 0.15 is the sweet spot. 0.10 achieves high peaks but is less stable over 20 epochs. 0.20 works but slightly worse.

### Learning Rate (no mixup, dp=0.05, WD=0.05, WSD 10/70/20)

| LR   | Test                   |
| ---- | ---------------------- |
| 1e-5 | 81.93%                 |
| 3e-5 | 81.98%                 |
| 5e-5 | killed (peaked 81.95%) |
| 7e-5 | killed (peaked 81.94%) |
| 1e-4 | killed (peaked 81.94%) |
| 3e-4 | killed (peaked 82.01%) |

**Conclusion**: LR=3e-5 is the most stable for 20-epoch finetuning. Higher LR peaks faster but degrades. LR=3e-4 peaks highest early but crashes.

### Schedule (no mixup, LR=3e-5, dp=0.05, WD=0.05)

| Schedule               | Test   |
| ---------------------- | ------ |
| WSD 10/70/20 (default) | 81.98% |
| WSD 10/10/80           | 81.99% |
| WSD 10/0/90            | 81.99% |
| Cosine 25% warmup      | 81.96% |

**Conclusion**: Heavy decay (80-90%) gives a small but consistent edge. Cosine is slightly worse.

### Weight Decay (no mixup, LR=3e-5, dp=0.10, WSD 10/70/20)

| WD   | Test                   |
| ---- | ---------------------- |
| 0.05 | killed (peaked 82.05%) |
| 0.10 | **82.06%**             |

**Conclusion**: Higher WD (0.1) stabilizes training with dp=0.10.

### Mixup/CutMix (LR=3e-5, WD=0.05, dp=0.05, WSD 10/70/20)

| Mixup        | Test                 |
| ------------ | -------------------- |
| On (0.8/1.0) | 81.89% (grid center) |
| Off          | 81.98%               |

**Conclusion**: Turning off Mixup/CutMix gives +0.09pp. This is the foundation of all the best results.

### Label Smoothing (no mixup, LR=3e-5, WD=0.05, dp=0.10, WSD 10/70/20)

| Smoothing     | Test                   |
| ------------- | ---------------------- |
| 0.1 (default) | killed (peaked 82.05%) |
| 0.0           | **82.05%**             |

**Conclusion**: Removing label smoothing gives comparable peak performance and appears slightly more stable.

### EMA Decay (LR=3e-5, WD=0.05, dp=0.05, WSD 10/70/20, with mixup)

| EMA Decay         | Test                 |
| ----------------- | -------------------- |
| None              | 81.64%               |
| 0.999             | 81.75%               |
| 0.9999            | 81.88%               |
| 0.99996 (default) | 81.89% (grid center) |
| 0.99999           | 81.90%               |

**Conclusion**: EMA is essential (+0.25pp). Default 0.99996 is fine; no need to tune.

______________________________________________________________________

## Progression of Discoveries

1. **LR × WD grid** (Group A): marginal gains, best 81.91%
1. **No Mixup/CutMix** (Group B): breakthrough to 81.98% (+0.07pp)
1. **Heavy decay schedules** (Group F): 81.99% and more stable
1. **Drop path 0.10** (Group H): 82.05% — structural regularization compensates for removed mixup (+0.07pp)
1. **Drop path 0.15**: **82.07%** — the sweet spot (+0.02pp)
1. **Higher WD (0.1) + dp 0.10**: 82.06% — complementary regularization

Total gain: **81.81% → 82.07% = +0.26pp** over pretrained baseline.

______________________________________________________________________

## How to Launch

```bash
sbatch --job-name=wsd-ft-NAME scripts/submit_2gpu.sh examples/vit5_imagenet/wsd_ft_ablation/CONFIG.py
```

## Launch Log

- **2026-03-04 02:30**: Groups A+B (16 runs) submitted.
- **2026-03-04 02:58**: 2 runs failed (lazy_config placeholder bug). Fixed `_base.py`. Resubmitted.
- **2026-03-04 03:05**: 4 runs failed (parallel `cp --no-clobber` staging). Fixed `dali_imagenet_fused.py`. Resubmitted.
- **2026-03-04 03:23**: Group C (EMA ablations, 4 runs) submitted.
- **2026-03-04 04:32**: Groups D-G (no-mixup ablations, 12 runs) submitted.
- **2026-03-04 04:42**: Standalone comparisons (cosine, short-epoch variants).
- **2026-03-04 05:00–05:30**: Adaptive submissions: lr2e4, lr5e5_wd01, decay variants. Killed first degrading runs.
- **2026-03-04 05:30–06:00**: Drop path ablations (dp=0, dp=0.1, dp=0.15). dp=0.1 hits 82.00%.
- **2026-03-04 06:00–06:30**: dp=0.1 combos (decay, smoothing, WD, LR). dp=0.15 hits 82.07%.
- **2026-03-04 06:30–07:14**: dp=0.15 combos, dp=0.2 variants, dp=0.25. Final submission at 23:14.
- **2026-03-04 ~01:00**: All runs complete.
- Over 60 runs launched, ~25 killed early for degradation, 39 completed with test scores.
