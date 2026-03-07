# Hyena Ablation Tracker

Goal: Close the 0.72pp gap between Hyena (81.45%) and Attention (82.17%).

## Baselines

| Model                 | Best val/acc_ema | Peak Epoch | wandb ID | Status |
| --------------------- | ---------------- | ---------- | -------- | ------ |
| Attention v3          | 82.17%           | 751        | 44or24g1 | Done   |
| Hyena v3 (omega_0=10) | 81.45%           | 639        | 5y0kxe8q | Done   |

## Tier 1a: Single-variable tests (launched in parallel)

| #   | Name           | Hypothesis                | Override                            | SLURM Job | Node        | wandb ID | Best val/acc_ema  | Status  |
| --- | -------------- | ------------------------- | ----------------------------------- | --------- | ----------- | -------- | ----------------- | ------- |
| 2   | drop_path=0.15 | H2: regularization        | `net.block_cfg.drop_path_rate=0.15` | 33923     | b65c909e-19 | je5g     | 0.779 (ep435)     | RUNNING |
| 5   | no pos_embed   | H4: absolute pos overfits | `net.use_pos_embed=False`           | 33910     | b65c909e-08 | 59fr     | **0.803** (ep487) | RUNNING |

## Tier 1b: Follow-up (launch after Tier 1a or when nodes free up)

| #   | Name                | Hypothesis        | Override                                                     | SLURM Job | Node        | wandb ID | Best val/acc_ema | Status                                               |
| --- | ------------------- | ----------------- | ------------------------------------------------------------ | --------- | ----------- | -------- | ---------------- | ---------------------------------------------------- |
| 1   | eta_min=4e-5        | H1: LR floor      | `scheduler.eta_min=4e-5`                                     | --        | --          | --       | --               | CANCELLED (low priority given overfitting diagnosis) |
| 3   | eta_min + drop_path | H1+H2 combined    | `scheduler.eta_min=4e-5 net.block_cfg.drop_path_rate=0.1`    | --        | --          | --       | --               | CANCELLED (low priority)                             |
| 4   | AdamW               | H3: LAMB mismatch | AdamW config (lr=1e-3, betas=0.9/0.95, wd=0.05, warmup=20ep) | 33980     | b65c909e-04 | 8amm     | 0.702 (ep128)    | RUNNING                                              |

## Additional ablations

| #   | Name            | Hypothesis                    | Override                                        | SLURM Job | Node        | wandb ID | Best val/acc_ema | Status  |
| --- | --------------- | ----------------------------- | ----------------------------------------------- | --------- | ----------- | -------- | ---------------- | ------- |
| 2b  | drop_path=0.1   | H2: lighter reg               | `net.block_cfg.drop_path_rate=0.1`              | 33979     | b65c909e-20 | e1lx     | 0.714 (ep132)    | RUNNING |
| 6   | SIREN hidden WD | SIREN expressiveness overfits | `kernel_cfg.weight_decay_on_hidden_layers=True` | 33981     | b65c909e-06 | ys29     | 0.709 (ep125)    | RUNNING |

## Tier 2: Conditional on Tier 1 results

| #   | Name             | Hypothesis        | Override                                | SLURM Job | Node | wandb ID | Best val/acc_ema | Status  |
| --- | ---------------- | ----------------- | --------------------------------------- | --------- | ---- | -------- | ---------------- | ------- |
| 7   | AdamW + eta_min  | H3+H1             | AdamW config + `scheduler.eta_min=1e-5` | --        | --   | --       | --               | PENDING |
| 8   | weight_decay=0.1 | H2 alt (global)   | `optimizer.weight_decay=0.1`            | --        | --   | --       | --               | PENDING |
| 9   | RandAugment      | Augmentation      | DALI config change                      | --        | --   | --       | --               | PENDING |
| 10  | RoPE             | Relative position | `use_rope=True`                         | --        | --   | --       | --               | PENDING |

## Checkpoint analysis (run 5y0kxe8q)

EMA weight drift from best (epoch 639) to latest (epoch 791) reveals:

- **SIREN layers barely move** during overfitting (|Δ|/|w| = 0.01-0.05, cosine ~0.999)
- **Hyena shortcut** drifts most (0.29x, shrinking) -- WD pulling down gating weights
- **QKV/MLP** drift 0.22-0.23x (all shrinking under WD)
- Overfitting happens in standard ViT components, not SIREN

## Notes

- All runs use `wandb.job_group=vit5_imagenet_hyena_ablation`
- Base config: `examples/vit5_imagenet/v3/vit5_small_pretrain_hyena_cls_row_gated_ema.py`
- Code change for Run 5: `use_pos_embed` flag added to `ViT5ClassificationNet` (gates `pos_embed` allocation and forward add)
- Code change for Run 6: `weight_decay_on_hidden_layers` flag added to `SIRENKernelND`
