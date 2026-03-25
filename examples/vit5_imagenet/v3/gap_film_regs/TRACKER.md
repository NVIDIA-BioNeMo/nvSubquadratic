# GAP + Registers + FiLM Ablation Sweep (L_cache=14 fix)

## Baseline

| Model                           | Run        | val/acc_ema | Notes               |
| ------------------------------- | ---------- | ----------- | ------------------- |
| GAP Hyena (pretrained, no FiLM) | `tcji9tfx` | **81.50%**  | Starting checkpoint |
| Previous FiLM ceiling (CLS)     | —          | **81.70%**  | Target to beat      |

## Fixed Settings (all runs)

- 25 epochs, cosine schedule, 5-epoch warmup (20%)
- Batch 256/GPU x 2 GPUs = effective 512
- EMA decay 0.99996
- SoftTargetCE loss, label smoothing 0.1
- No Mixup/CutMix, standard RandAugment (rand-m9-mstd0.5-inc1)
- bf16-mixed, torch.compile (max-autotune-no-cudagraphs)
- fft_backend: torch_fft
- L_cache=14 (auto-extends to 15 for register models)
- reg_init: zeros (default), except \_trunc variants
- Submit: scripts/submit_2gpu.sh (2 GPUs, 32 CPUs)
- Nodes: b65c909e-06, b65c909e-08 (always both occupied)

## Sweep Axes

### Primary (core grid: 2 reg x 3 FiLM x 2 recipes = 12 runs)

- **num_registers**: 4, 14
- **FiLM mode**:
  - film3_after: 3 layers, after pos_embed sine (modulates pos_embed + hidden\[0\] + hidden\[1\])
  - film3_before: 3 layers, before pos_embed sine (hidden\[0\] + hidden\[1\], 3rd pair unused but larger MLP)
  - film2: 2 layers, before (hidden\[0\] + hidden\[1\] only, minimal MLP)
- **LR recipe**:
  - A (aggressive): lr=3e-5, wd=0.05, dp=0.15
  - B (conservative): lr=1e-5, wd=0.1, dp=0.05

### Targeted ablations (on film3_after, Recipe A: 12 runs)

- **film_wd**: global (default) vs 0.001 (2 runs)
- **reg_init**: zeros (default) vs trunc_normal (2 runs)
- **LLRD**: 0.75 (2 runs)
- **trunc_normal + low film_wd** combined (2 runs)
- **film_init small_random** (2 runs)
- **film_hidden_dim=128** (2 runs)

______________________________________________________________________

## Full Experiment List

### Wave 1 (AdamW) — CANCELLED (overfitting)

All 8 AdamW runs (jobs 36928-36935) cancelled due to early overfitting.

### Wave 1 (LAMB) — LR sweep with film3_after (8 runs)

Optimizer switched to `apex.optimizers.FusedLAMB` (same as pretraining).
All runs use film3_after, wd=0.05, dp=0.15, reg_init=zeros.

| #   | Config               | Regs | lr   | Optimizer | Node | Job ID | val/acc_ema | Status               |
| --- | -------------------- | ---- | ---- | --------- | ---- | ------ | ----------- | -------------------- |
| 1   | r14_f3after_lamb_3e5 | 14   | 3e-5 | LAMB      | 06   | 36939  | **81.56%**  | finished             |
| 2   | r14_f3after_lamb_1e4 | 14   | 1e-4 | LAMB      | 06   | 36940  | **81.57%**  | finished             |
| 3   | r14_f3after_lamb_3e4 | 14   | 3e-4 | LAMB      | 06   | 36941  |             | CANCELLED (user)     |
| 4   | r14_f3after_lamb_1e3 | 14   | 1e-3 | LAMB      | 06   | 36942  |             | CANCELLED (diverged) |
| 5   | r4_f3after_lamb_3e5  | 4    | 3e-5 | LAMB      | 08   | 36943  | **81.56%**  | finished             |
| 6   | r4_f3after_lamb_1e4  | 4    | 1e-4 | LAMB      | 08   | 36944  | **81.58%**  | finished             |
| 7   | r4_f3after_lamb_3e4  | 4    | 3e-4 | LAMB      | 08   | 36945  |             | CANCELLED (user)     |
| 8   | r4_f3after_lamb_1e3  | 4    | 1e-3 | LAMB      | 08   | 36946  |             | CANCELLED (diverged) |

**Final observations**: All 4 completed runs peaked at 81.56-81.58% (best: r4 1e-4 at 81.58%). Overfitting after ~epoch 10 (val_loss rising, val_acc declining from peak). 4 vs 14 regs nearly identical. lr=1e-3 diverged. lr=3e-4 cancelled early by user.

### Augmentation experiments (LAMB lr=3e-4, 14 regs, film3_after)

Problem: current finetuning uses no Mixup/CutMix, but pretraining used Mixup=0.8 + CutMix=1.0. Adding these back may fix overfitting.

**NOTE**: All A1-A4 used identity FiLM init (deadlocked — FiLM not contributing). Results reflect aug recipe only.

| #   | Config                             | Augmentation        | Mixup | CutMix | Smooth | dp   | Job ID | WandB      | val/acc_ema      | Status                     |
| --- | ---------------------------------- | ------------------- | ----- | ------ | ------ | ---- | ------ | ---------- | ---------------- | -------------------------- |
| A1  | r14_f3after_lamb_3e4_aug_pretrain  | three-aug           | 0.8   | 1.0    | 0.0    | 0.05 | 36956  | `0gy1fisr` | **81.52%** (ep7) | CANCELLED (overfitting)    |
| A2  | r14_f3after_lamb_3e4_aug_pt_reg    | three-aug           | 0.8   | 1.0    | 0.1    | 0.15 | 36957  | `qjs9fddo` | **81.61%** (ep6) | CANCELLED (overfitting)    |
| A3  | r14_f3after_lamb_3e4_aug_ramix     | RandAug m9          | 0.8   | 1.0    | 0.1    | 0.15 | 36972  | `0zr3d661` | **81.67%** (ep6) | CANCELLED (relaunch as B1) |
| A4  | r14_f3after_lamb_3e4_aug_ramix_ra3 | RandAug m9 + RA(x3) | 0.8   | 1.0    | 0.1    | 0.15 | 36973  | `0uek6c7e` | **81.62%** (ep5) | CANCELLED (overfitting)    |

**Winner**: A3 (RandAug m9 + Mixup 0.8 + CutMix 1.0 + smoothing 0.1 + dp 0.15) — used as baseline for all subsequent sweeps.

### Recipe ablations (data / LR / WD composition, identity FiLM init)

| #   | Config                          | lr   | wd   | dp   | Aug        | Mixup | CutMix | Job ID | val/acc_ema       | Status                  |
| --- | ------------------------------- | ---- | ---- | ---- | ---------- | ----- | ------ | ------ | ----------------- | ----------------------- |
| R1  | r14_f3after_lamb_3e4_wd01_mixup | 3e-4 | 0.1  | 0.15 | RandAug m9 | 0.8   | 1.0    | 36980  | **81.65%** (ep5)  | CANCELLED (overfitting) |
| R2  | r14_f3after_lamb_3e4_wd01       | 3e-4 | 0.1  | 0.15 | RandAug m9 | 0.0   | 0.0    | 36981  | **81.64%** (ep10) | CANCELLED (overfitting) |
| R3  | r14_f3after_lamb_1e4_mixup      | 1e-4 | 0.05 | 0.15 | three-aug  | 0.8   | 1.0    | 36982  | **81.64%** (ep11) | CANCELLED (overfitting) |
| R4  | r14_f3after_lamb_ref_recipe     | 1e-5 | 0.1  | 0.05 | RandAug m9 | 0.0   | 0.0    | —      |                   | deprioritised           |
| R5  | r14_f3after_lamb_ref_mixup      | 1e-5 | 0.1  | 0.05 | RandAug m9 | 0.8   | 1.0    | —      |                   | deprioritised           |

______________________________________________________________________

## A3 Baseline Sweep (current)

**Baseline**: 0zr3d661 / A3 recipe — 14 regs, 3 FiLM layers, film_after_pos_embed=True,
LAMB lr=3e-4, wd=0.05, dp=0.15, RandAug m9, Mixup 0.8, CutMix 1.0, smoothing 0.1.

**Key discovery**: identity FiLM init creates a LAMB trust-ratio deadlock (FiLM output weights stuck at ~1e-17).
`small_random` init (`film_init_type="small_random"`) fixes this — w_norm_out_avg=0.011 after 1 epoch.

### B: Baseline + FiLM fix (already running)

| Tag | Config                            | Change from A3        | Init | Job   | val/acc_ema | Status           |
| --- | --------------------------------- | --------------------- | ---- | ----- | ----------- | ---------------- |
| B1  | r14_f3after_lamb_3e4_aug_ramix_sr | film_init=sr          | sr   | 36984 | `2lldljq6`  | **81.63%** (ep5) |
| B2  | r14_f3after_lamb_1e4_aug_ramix_sr | lr=1e-4, film_init=sr | sr   | 36986 | `gqcvou4j`  | **81.66%** (ep7) |

### Sweep 1: Weight Decay (A3 baseline, vary wd)

| Tag | Config             | wd   | Init     | Job   | val/acc_ema | Status           |
| --- | ------------------ | ---- | -------- | ----- | ----------- | ---------------- |
| —   | (B1)               | 0.05 | sr       | 36984 | `2lldljq6`  | **81.63%** (ep5) |
| W1  | r14_ramix_sr_wd002 | 0.02 | sr       | 36990 | `oeesikj4`  | **81.63%** (ep7) |
| W2  | r14_ramix_sr_wd01  | 0.1  | sr       | 36991 | `ovkwvbc1`  | **81.66%** (ep5) |
| W3  | r14_ramix_sr_wd02  | 0.2  | sr       | 36992 | `4g2q3hxs`  | **81.66%** (ep6) |
| W1i | r14_ramix_id_wd002 | 0.02 | identity | 37021 |             | running          |
| W2i | r14_ramix_id_wd01  | 0.1  | identity | 37010 | `qzs6c1hr`  | **81.68%** (ep5) |
| W3i | r14_ramix_id_wd02  | 0.2  | identity | 37011 | `xqqlh4hk`  | **81.68%** (ep6) |

### Sweep 2: Drop Path Rate (A3 baseline, vary dp)

| Tag | Config             | dp   | Init     | Job   | val/acc_ema | Status           |
| --- | ------------------ | ---- | -------- | ----- | ----------- | ---------------- |
| D1  | r14_ramix_sr_dp005 | 0.05 | sr       | 37002 | `qrnv7grz`  | **81.64%** (ep8) |
| D2  | r14_ramix_sr_dp010 | 0.10 | sr       | 37003 | `891e3827`  | **81.63%** (ep8) |
| —   | (B1)               | 0.15 | sr       | 36984 | `2lldljq6`  | **81.63%** (ep5) |
| D3  | r14_ramix_sr_dp020 | 0.20 | sr       | 37004 | `s11cw6sm`  | **81.62%** (ep4) |
| D4  | r14_ramix_sr_dp025 | 0.25 | sr       | 37006 | `mnhn46rs`  | **81.57%** (ep5) |
| D5  | r14_ramix_sr_dp030 | 0.30 | sr       | 37007 | `72fu3uen`  | **81.52%** (ep2) |
| D2i | r14_ramix_id_dp010 | 0.10 | identity | 37023 |             | pending          |
| D4i | r14_ramix_id_dp025 | 0.25 | identity | 37024 |             | pending          |

### Sweep 3: FiLM Weight Decay (A3 baseline, vary film_wd)

| Tag | Config               | film_wd             | Init | Job   | val/acc_ema | Status                        |
| --- | -------------------- | ------------------- | ---- | ----- | ----------- | ----------------------------- |
| —   | (B1)                 | False (global=0.05) | sr   | 36984 | `2lldljq6`  | **81.63%** (ep5)              |
| F1  | r14_ramix_sr_fwdnone | True (no WD)        | sr   | 37008 | `nxmyscsk`  | **81.62%** (ep5), w_norm=2.82 |
| F2  | r14_ramix_sr_fwd1e3  | 0.001               | sr   | 37009 | `ub2dwisx`  | **81.66%** (ep6)              |

### Sweep 4: Learning Rate (A3 baseline, vary lr)

| Tag | Config             | lr   | Init | Job   | val/acc_ema | Status           |
| --- | ------------------ | ---- | ---- | ----- | ----------- | ---------------- |
| —   | (B2)               | 1e-4 | sr   | 36986 | `gqcvou4j`  | **81.66%** (ep7) |
| —   | (B1)               | 3e-4 | sr   | 36984 | `2lldljq6`  | **81.63%** (ep5) |
| L1  | r14_ramix_sr_lr5e4 | 5e-4 | sr   | 37012 | `pbsxxvmg`  | **81.64%** (ep6) |
| L2  | r14_ramix_sr_lr1e3 | 1e-3 | sr   | 37022 |             | pending          |

### Sweep 5: Architecture (A3 baseline + sr, vary arch)

| Tag | Config               | Change          | Init | Job   | val/acc_ema | Status           |
| --- | -------------------- | --------------- | ---- | ----- | ----------- | ---------------- |
| A5  | r14_ramix_sr_r4      | 4 registers     | sr   | 37013 | `wa8nk8js`  | **81.66%** (ep5) |
| A6  | r14_ramix_sr_f2      | 2 FiLM layers   | sr   | 37015 | `p3fn6tsz`  | **81.63%** (ep5) |
| A7  | r14_ramix_sr_fh128   | film_hidden=128 | sr   | 37016 | `uc2wnq55`  | **81.65%** (ep5) |
| A8  | r14_ramix_sr_fbefore | film_before_pos | sr   | 37017 |             | running          |

### Sweep 6: Training Recipe (A3 baseline + sr, misc)

| Tag | Config             | Change              | Init | Job   | val/acc_ema | Status  |
| --- | ------------------ | ------------------- | ---- | ----- | ----------- | ------- |
| T1  | r14_ramix_sr_llrd  | LLRD=0.75           | sr   | 37018 |             | running |
| T2  | r14_ramix_sr_ra3   | repeated_aug x3     | sr   | 37019 |             | running |
| T3  | r14_ramix_sr_re025 | random_erasing=0.25 | sr   | 37020 |             | running |

### Waves 2-3 (old plan, deprioritised)

Superseded by A3-baseline sweep above. May revisit if sweep results warrant it.
