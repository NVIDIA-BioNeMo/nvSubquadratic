# GAP + Registers + FiLM Ablation Sweep

## Baseline

| Model                           | Run        | val/acc_ema | Notes               |
| ------------------------------- | ---------- | ----------- | ------------------- |
| GAP Hyena (pretrained, no FiLM) | `tcji9tfx` | **81.50%**  | Starting checkpoint |
| Previous FiLM ceiling (CLS)     | —          | **81.70%**  | Target to beat      |

## Fixed Settings (all runs)

- 25 epochs, cosine schedule, 5-epoch warmup
- Batch 256/GPU x 2 GPUs = effective 512
- EMA decay 0.99996
- SoftTargetCE loss, label smoothing 0.1
- No Mixup/CutMix, standard RandAugment (rand-m9-mstd0.5-inc1)
- bf16-mixed, torch.compile (max-autotune-no-cudagraphs)
- fft_backend: torch_fft

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

- **reg_init**: trunc_normal (default) vs zeros (2 runs)
- **film_wd**: global (default) vs 0.001 (2 runs)
- **LLRD**: 0.75 (2 runs)
- **zero-init + low film_wd** combined (2 runs)
- **film_init small_random** (2 runs)
- **film_hidden_dim=128** (2 runs)

## Wave 1 (8 runs) — Recipe A core grid + zero-init

| Config          | Regs | FiLM    | Recipe | Ablation  | Job ID | val/acc_ema      | Status |
| --------------- | ---- | ------- | ------ | --------- | ------ | ---------------- | ------ |
| r14_f3after_A   | 14   | 3after  | A      | —         | 36890  | 81.35% (e16)     | done   |
| r14_f3before_A  | 14   | 3before | A      | —         | 36891  | **81.40%** (e20) | done   |
| r14_f2_A        | 14   | f2      | A      | —         | 36892  | **81.42%** (e18) | done   |
| r4_f3after_A    | 4    | 3after  | A      | —         | 36893  | 81.36% (e?)      | done   |
| r4_f3before_A   | 4    | 3before | A      | —         | 36894  | 81.40% (e18)     | done   |
| r4_f2_A         | 4    | f2      | A      | —         | 36895  | 81.39% (e18)     | done   |
| r14_f3after_A_z | 14   | 3after  | A      | zero-init | 36896  | 81.35% (e17)     | done   |
| r4_f3after_A_z  | 4    | 3after  | A      | zero-init | 36897  | 81.40% (e?)      | done   |

## Wave 2 (8 runs) — Recipe B core grid + film_wd ablation

| Config            | Regs | FiLM    | Recipe | Ablation      | Job ID | val/acc_ema | Status  |
| ----------------- | ---- | ------- | ------ | ------------- | ------ | ----------- | ------- |
| r14_f3after_B     | 14   | 3after  | B      | —             |        |             | pending |
| r14_f3before_B    | 14   | 3before | B      | —             |        |             | pending |
| r14_f2_B          | 14   | f2      | B      | —             |        |             | pending |
| r4_f3after_B      | 4    | 3after  | B      | —             |        |             | pending |
| r4_f3before_B     | 4    | 3before | B      | —             |        |             | pending |
| r4_f2_B           | 4    | f2      | B      | —             |        |             | pending |
| r14_f3after_A_lwd | 14   | 3after  | A      | film_wd=0.001 |        |             | pending |
| r4_f3after_A_lwd  | 4    | 3after  | A      | film_wd=0.001 |        |             | pending |

## Wave 3 (8 runs) — Targeted ablations (film3_after, Recipe A)

| Config              | Regs | FiLM   | Recipe | Ablation     | Job ID | val/acc_ema | Status  |
| ------------------- | ---- | ------ | ------ | ------------ | ------ | ----------- | ------- |
| r14_f3after_A_llrd  | 14   | 3after | A      | LLRD 0.75    |        |             | pending |
| r4_f3after_A_llrd   | 4    | 3after | A      | LLRD 0.75    |        |             | pending |
| r14_f3after_A_z_lwd | 14   | 3after | A      | zero+lwd     |        |             | pending |
| r4_f3after_A_z_lwd  | 4    | 3after | A      | zero+lwd     |        |             | pending |
| r14_f3after_A_sr    | 14   | 3after | A      | small_random |        |             | pending |
| r4_f3after_A_sr     | 4    | 3after | A      | small_random |        |             | pending |
| r14_f3after_A_fh128 | 14   | 3after | A      | fh=128       |        |             | pending |
| r4_f3after_A_fh128  | 4    | 3after | A      | fh=128       |        |             | pending |
