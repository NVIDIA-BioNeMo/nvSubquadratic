# Hyena Ablation Tracker

Goal: Close the 0.72pp gap between Hyena (81.45%) and Attention (82.17%).

## Baselines

| Model                 | Peak val/acc_ema | Peak Epoch | Final (ep800) | wandb ID | Status |
| --------------------- | ---------------- | ---------- | ------------- | -------- | ------ |
| Attention v3          | 82.22%           | 743        | 82.09%        | 44or24g1 | Done   |
| Hyena v3 (omega_0=10) | 81.45%           | 639        | 81.02%        | 5y0kxe8q | Done   |
| FiLM-Hyena            | 81.83%           | 679        | 81.61%        | peeaqdkq | Done   |

## Tier 1a: Single-variable tests (launched in parallel)

| #   | Name           | Hypothesis                | Override                            | SLURM Job | Node        | wandb ID | Peak val/acc_ema   | Final (ep800) | Status |
| --- | -------------- | ------------------------- | ----------------------------------- | --------- | ----------- | -------- | ------------------ | ------------- | ------ |
| 2   | drop_path=0.15 | H2: regularization        | `net.block_cfg.drop_path_rate=0.15` | 33923     | b65c909e-19 | qva7je5g | 0.8128 (ep739)     | 0.8120        | DONE   |
| 5   | no pos_embed   | H4: absolute pos overfits | `net.use_pos_embed=False`           | 33910     | b65c909e-08 | bueo59fr | **0.8141** (ep643) | 0.8099        | DONE   |

## Tier 1b: Follow-up (launch after Tier 1a or when nodes free up)

| #   | Name                | Hypothesis        | Override                                                     | SLURM Job | Node        | wandb ID | Best val/acc_ema | Status                                               |
| --- | ------------------- | ----------------- | ------------------------------------------------------------ | --------- | ----------- | -------- | ---------------- | ---------------------------------------------------- |
| 1   | eta_min=4e-5        | H1: LR floor      | `scheduler.eta_min=4e-5`                                     | --        | --          | --       | --               | CANCELLED (low priority given overfitting diagnosis) |
| 3   | eta_min + drop_path | H1+H2 combined    | `scheduler.eta_min=4e-5 net.block_cfg.drop_path_rate=0.1`    | --        | --          | --       | --               | CANCELLED (low priority)                             |
| 4   | AdamW               | H3: LAMB mismatch | AdamW config (lr=1e-3, betas=0.9/0.95, wd=0.05, warmup=20ep) | 33980     | b65c909e-04 | np4b8amm | 0.8005 (ep579)   | DONE (final 0.7893)                                  |

## Additional ablations

| #   | Name            | Hypothesis                    | Override                                        | SLURM Job | Node        | wandb ID | Best val/acc_ema   | Status              |
| --- | --------------- | ----------------------------- | ----------------------------------------------- | --------- | ----------- | -------- | ------------------ | ------------------- |
| 2b  | drop_path=0.1   | H2: lighter reg               | `net.block_cfg.drop_path_rate=0.1`              | 33979     | b65c909e-20 | ine6e1lx | **0.8159** (ep699) | DONE (final 0.8143) |
| 6   | SIREN hidden WD | SIREN expressiveness overfits | `kernel_cfg.weight_decay_on_hidden_layers=True` | 33981     | b65c909e-06 | hvnnys29 | 0.8077 (ep635)     | DONE (final 0.8011) |

## Tier 1c: Schedule & LR experiments

| #   | Name         | Hypothesis                         | Override                 | SLURM Job | Node        | wandb ID | Peak val/acc_ema | Status                                            |
| --- | ------------ | ---------------------------------- | ------------------------ | --------- | ----------- | -------- | ---------------- | ------------------------------------------------- |
| 11  | eta_min=1e-5 | H5: LR floor prevents late overfit | `scheduler.eta_min=1e-5` | 34069     | b65c909e-13 | 70i2uuvx | 0.8146 (ep647)   | DONE (final 0.8097, no improvement over baseline) |
| 12  | LR=6e-3      | H6: higher LR → flatter minima     | `optimizer.lr=6e-3`      | 34070     | b65c909e-14 | o77wzn4w | 0.7835 (ep362)   | CANCELLED (underperforming)                       |
| 13  | LR=5e-3      | H6b: moderate LR increase          | `optimizer.lr=5e-3`      | 34118     | b65c909e-04 | d7tsbmoj | 0.8135 (ep731)   | DONE (final 0.8116)                               |

## Tier 1d: Combinations (based on drop_path=0.1 being best single-variable result)

| #   | Name                    | Hypothesis                 | Override                                                   | SLURM Job | Node        | wandb ID | Peak val/acc_ema   | Status                            |
| --- | ----------------------- | -------------------------- | ---------------------------------------------------------- | --------- | ----------- | -------- | ------------------ | --------------------------------- |
| 14  | drop_path=0.1 + eta_min | Best reg + schedule fix    | `net.block_cfg.drop_path_rate=0.1 scheduler.eta_min=1e-5`  | 34119     | b65c909e-06 | 7py11egn | **0.8156** (ep791) | DONE (final 0.8151, best result!) |
| 15  | drop_path=0.1 + no_pos  | Best reg + no absolute pos | `net.block_cfg.drop_path_rate=0.1 net.use_pos_embed=False` | 34120     | b65c909e-12 | wr6nl6no | 0.8140 (ep691)     | DONE (final 0.8129)               |

## Tier 1e: Regularization fine-tuning & augmentation

| #   | Name            | Hypothesis                             | Override                             | SLURM Job | Node        | wandb ID | Peak val/acc_ema | Status                                   |
| --- | --------------- | -------------------------------------- | ------------------------------------ | --------- | ----------- | -------- | ---------------- | ---------------------------------------- |
| 16  | drop_path=0.075 | Sweet spot between 0.05 and 0.1        | `net.block_cfg.drop_path_rate=0.075` | 34235     | b65c909e-42 | horchgti | 0.8149 (ep723)   | DONE (final 0.8136)                      |
| 17  | smoothing=0.1   | Label smoothing (was 0.0, default 0.1) | `dataset.mixup_cfg.smoothing=0.1`    | 34286     | b65c909e-06 | 38kbg225 | 0.8141 (ep663)   | DONE (final 0.8089, worse than baseline) |

## Tier 2: Conditional on Tier 1 results

| #   | Name             | Hypothesis        | Override                                | SLURM Job | Node | wandb ID | Best val/acc_ema | Status  |
| --- | ---------------- | ----------------- | --------------------------------------- | --------- | ---- | -------- | ---------------- | ------- |
| 7   | AdamW + eta_min  | H3+H1             | AdamW config + `scheduler.eta_min=1e-5` | --        | --   | --       | --               | PENDING |
| 8   | weight_decay=0.1 | H2 alt (global)   | `optimizer.weight_decay=0.1`            | --        | --   | --       | --               | PENDING |
| 9   | RandAugment      | Augmentation      | DALI config change                      | --        | --   | --       | --               | PENDING |
| 10  | RoPE             | Relative position | `use_rope=True`                         | --        | --   | --       | --               | PENDING |

## Analysis

### Peak vs final decay (finished runs, sorted by peak)

| Run               | Peak       | Peak Ep | Final      | Decay       | Gap to Attn |
| ----------------- | ---------- | ------- | ---------- | ----------- | ----------- |
| Attention         | 82.22%     | 743     | 82.09%     | -0.13pp     | --          |
| FiLM-Hyena        | 81.83%     | 679     | 81.61%     | -0.22pp     | -0.39pp     |
| **dp0.1+eta_min** | **81.56%** | **791** | **81.51%** | **-0.04pp** | **-0.66pp** |
| drop_path=0.1     | 81.59%     | 699     | 81.43%     | -0.16pp     | -0.63pp     |
| drop_path=0.075   | 81.49%     | 723     | 81.36%     | -0.13pp     | -0.73pp     |
| eta_min=1e-5      | 81.46%     | 647     | 80.97%     | -0.49pp     | -0.76pp     |
| Hyena baseline    | 81.45%     | 639     | 81.02%     | -0.43pp     | -0.77pp     |
| smoothing=0.1     | 81.41%     | 663     | 80.89%     | -0.53pp     | -0.81pp     |
| dp0.1+no_pos      | 81.40%     | 691     | 81.29%     | -0.11pp     | -0.82pp     |
| no pos_embed      | 81.41%     | 643     | 80.99%     | -0.42pp     | -0.81pp     |
| LR=5e-3           | 81.35%     | 731     | 81.16%     | -0.19pp     | -0.87pp     |
| drop_path=0.15    | 81.28%     | 739     | 81.20%     | -0.08pp     | -0.94pp     |
| SIREN hidden WD   | 80.77%     | 635     | 80.11%     | -0.66pp     | -1.45pp     |
| AdamW             | 80.05%     | 579     | 78.93%     | -1.12pp     | -2.17pp     |

### Key findings

- **Best combination**: dp0.1+eta_min — peak 81.56% (ep791), final 81.51%, only 0.04pp decay
  - Peaks 152 epochs later than baseline; eta_min alone did nothing but combined with drop_path it works
  - Best final accuracy of any Hyena ablation (81.51% vs baseline 81.02%)
- **Best single-variable**: drop_path=0.1 — peak 81.59%, but more decay (0.16pp) and lower final (81.43%)
- **eta_min=1e-5 alone ineffective**: needs regularization to have impact
- **no_pos_embed adds nothing on top of drop_path=0.1**: dp0.1+no_pos ≈ dp0.1 alone
- **Remaining gap to attention at peak**: 0.66pp (down from 0.77pp baseline)
- **Remaining gap at final**: 0.58pp (down from 1.07pp baseline)
- **drop_path=0.075**: 81.49%/81.36%, monotonically between dp=0.05 and dp=0.1 — confirms dp=0.1 is the sweet spot
- **smoothing=0.1 hurts**: 81.41%/80.89%, *worse* decay than baseline (0.53pp vs 0.43pp). Label smoothing is counterproductive
- **FiLM parameterization**: residual+identity+no_wd (run 20) stable at ep279, val/acc_ema=76.6%
- **FiLM+pos_embed**: all 4 variants collapsed or had dead FiLM (see Tier 3 analysis)

### Checkpoint analysis (run 5y0kxe8q)

EMA weight drift from best (epoch 639) to latest (epoch 791) reveals:

- **SIREN layers barely move** during overfitting (|Δ|/|w| = 0.01-0.05, cosine ~0.999)
- **Hyena shortcut** drifts most (0.29x, shrinking) -- WD pulling down gating weights
- **QKV/MLP** drift 0.22-0.23x (all shrinking under WD)
- Overfitting happens in standard ViT components, not SIREN

## Tier 3: FiLM parameterization & pos-embed conditioning

Refactored `KernelFiLMGenerator` to support configurable parameterization (`residual` vs `direct`),
`no_weight_decay` flag, and initialization type (`identity` vs `small_random`).
For `residual` mode, modulation is `(1 + gamma) * h + beta` (identity when gamma=0, beta=0).
For `direct` mode, modulation is `gamma * h + beta` (identity when gamma=1, beta=0).

### Standard FiLM (no pos-embed conditioning)

| #   | Name              | Config                             | SLURM Job | wandb (v_num) | Epochs | Best loss | Status                                  |
| --- | ----------------- | ---------------------------------- | --------- | ------------- | ------ | --------- | --------------------------------------- |
| 18  | resid+small_rand  | residual, small_random init, no_wd | 34755     | w5op          | 35     | 3.99      | CANCELLED (loss spikes at ep15-17)      |
| 19  | direct+nowd       | direct, identity init, no_wd       | 34756     | 38x3          | 31     | 6.28      | CANCELLED (re-launched with posemb)     |
| 20  | **resid+id+nowd** | residual, identity init, no_wd     | 34813     | ziq3          | 279+   | 2.91      | **RUNNING** (val/acc_ema=76.6%, stable) |

### Pos-embed FiLM conditioning (`film_on_pos_embed=True`)

FiLM applied *before* sin() on the positional embedding: `sin(gamma * (Wx + b) + beta)`.
This enables input-dependent spatial warping of the kernel coordinates.

| #   | Name               | Config                                    | SLURM Job | wandb (v_num) | Epochs | Collapse ep | Status                                      |
| --- | ------------------ | ----------------------------------------- | --------- | ------------- | ------ | ----------- | ------------------------------------------- |
| 21  | resid+nowd+posemb  | residual, identity, no_wd, omega_0=10     | 34819     | rfv7          | 66     | ~17         | COLLAPSED (loss → 6.9)                      |
| 22  | direct+nowd+posemb | direct, identity, no_wd, omega_0=10       | 34852     | bktk          | 3      | ~2          | COLLAPSED (loss → 6.8)                      |
| 23  | resid+wd+posemb    | residual, identity, **wd ON**, omega_0=10 | 34855     | kufi          | 16     | --          | CANCELLED (stable but FiLM dead: weights=0) |
| 24  | resid+nowd+omega1  | residual, identity, no_wd, **omega_0=1**  | 34860     | 06gv          | 16     | ~15         | COLLAPSED (delayed but same failure)        |

### Pos-embed instability analysis

**Root cause**: omega_0 amplification of gamma in the pre-sin FiLM.

The pos-embed linear weights have omega_0 baked in (weights ~\[-30, +30\] for omega_0=10),
producing pre-activation values in \[-52, +52\]. When FiLM applies `(1+gamma) * pre_act + beta`
before `sin()`, gamma multiplies these large values:

| gamma | pos-embed disruption (pre-sin) | hidden disruption (post-sin) | Amplification |
| ----- | ------------------------------ | ---------------------------- | ------------- |
| 0.01  | mean=0.073, max=0.49           | mean=0.006, max=0.01         | 12x / 49x     |
| 0.05  | mean=0.350, max=1.89           | mean=0.032, max=0.05         | 11x / 38x     |
| 0.10  | mean=0.611, max=1.99           | mean=0.063, max=0.10         | 10x / 20x     |

A gamma of just 0.05 makes `sin()` output essentially random (max disruption 1.89/2.0).
Hidden layers are unaffected because FiLM is applied *after* sin() on values in \[-1, 1\].

**Collapse mechanism** (from checkpoint forensics, run 21):

1. Init: gamma=0, beta=0 (identity). Weight norms ~2-3.
1. Epoch 15 (pre-collapse): gamma std ~0.02-0.06. Already causing significant disruption.
1. Positive feedback: disrupted sine → noisy gradients → weights grow faster → larger gamma.
1. Post-collapse (epoch 63): gamma std 0.2-0.4, weight norms 12-24. Sine is random.

**Weight decay (run 23)**: Prevents collapse by keeping output weights at exactly zero.
Identity init + weight decay = dead zone: the FiLM never learns *anything*.

**Reduced omega_0 (run 24)**: Delays collapse from epoch 2-3 to epoch 14-15 by reducing
pre-activation magnitude ~10x, but the feedback loop still exists.

**Conclusion**: Pre-sin FiLM on pos-embed is fundamentally unstable without hard bounds on gamma.
Next experiments should explore: (a) tanh clamping of gamma/beta, or (b) moving FiLM to after
sine in the pos-embed layer.

## Notes

- All runs use `wandb.job_group=vit5_imagenet_hyena_ablation`
- Base config: `examples/vit5_imagenet/v3/vit5_small_pretrain_hyena_cls_row_gated_ema.py`
- FiLM base config: `examples/vit5_imagenet/v3/vit5_small_pretrain_hyena_cls_row_gated_film_ema.py`
- Pos-embed config: `examples/vit5_imagenet/v3/vit5_small_pretrain_hyena_cls_row_gated_film_posemb_ema.py`
- Code change for Run 5: `use_pos_embed` flag added to `ViT5ClassificationNet` (gates `pos_embed` allocation and forward add)
- Code change for Run 6: `weight_decay_on_hidden_layers` flag added to `SIRENKernelND`
- Code changes for Tier 3: Refactored `KernelFiLMGenerator` with `film_parameterization`, `no_weight_decay`, `init_type`, `init_std`; added `film_on_pos_embed` to `SIRENKernelND`
