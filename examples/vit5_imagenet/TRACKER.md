# ViT-5 Small ImageNet-1k — Experiment Tracker

W&B project: [`implicit-long-convs/nvsubquadratic`](https://wandb.ai/implicit-long-convs/nvsubquadratic)

## Config files

| Config | Mixer | Optimizer | CLS token | Registers | Notes |
|--------|-------|-----------|-----------|-----------|-------|
| `vit5_small_pretrain.py` | ViT5Attention | `torch_optimizer.Lamb` | Yes | 4 | Baseline attention, non-Apex LAMB |
| `vit5_small_pretrain_apex.py` | ViT5Attention | Apex `FusedLAMB` | Yes | 4 | Attention + fused LAMB |
| `vit5_small_pretrain_hyena_apex.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | Yes (mean-pool update) | 0 | Hyena replaces attention; CLS updated via mean-pool of mixed patches each layer |
| `vit5_small_pretrain_hyena_gap_apex.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | **No** (GAP) | 0 | Same as above but CLS token removed; classification via global average pooling |
| `vit5_small_pretrain_hyena_cls_row_apex.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | Yes (in-grid) | 13 (prepended) | CLS + 13 registers form extra row at top of 2D grid → 15×14. CLS participates directly in 2D convolution. Registers are global. |
| `vit5_small_pretrain_hyena_gap_apex_qk_rmsnorm.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | **No** (GAP) | 0 | Same as `hyena_gap_apex` but adds QK RMSNorm (dim=384) to the Hyena mixer |
| `vit5_small_pretrain_multihead_hyena_gap_apex_qk_rmsnorm.py` | Multihead Hyena (CKConvMultiheadND + SIREN) | Apex `FusedLAMB` | **No** (GAP) | 0 | Dense multi-head mixing (6 heads, 64 head_dim) replaces depthwise CKConvND. PerHeadRMSNorm for QK normalization. |
| `vit5_small_pretrain_hyena_gap_apex_optimized.py` | Hyena (CKConvND + SIREN, optimized) | Apex `FusedLAMB` | **No** (GAP) | 0 | SiLU nonlinear gates on k and v, L2Norm on q (conditional on k), output RMSNorm |

All configs share: ViT-5-Small (12 blocks, dim 384, patch 16, 224x224), LAMB lr=4e-3, wd=0.05, batch 2048, 800 epochs, cosine schedule, 5 warmup epochs, 3-Augment, Mixup 0.8 + CutMix 1.0, BCE loss, DropPath 0.05, LayerScale 1e-4, bf16-mixed.

## Active runs

| Job ID | Job name | Config | W&B run | Node | Status | Last epoch | val/loss | val/acc | it/s |
|--------|----------|--------|---------|------|--------|------------|----------|---------|------|
| 30923 | `vit5-apex` | `vit5_small_pretrain_apex.py` | [`ia7b26u7`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ia7b26u7) | b65c909e-01 | Running | ~744 | 0.899 | 79.9% | ~2.5 |
| 30924 | `vit5-simd` | `vit5_small_pretrain.py` | [`ea0z7ttf`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ea0z7ttf) | b65c909e-20 | Running | ~720 | 0.915 | 78.0% | ~2.5 |
| 30969 | `vit5-hyena` | `vit5_small_pretrain_hyena_apex.py` | [`r9pmc5ps`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/r9pmc5ps) | b65c909e-41 | Running | ~489 | 0.969 | 76.4% | ~2.6 |
| 30995 | `vit5-hyena-gap` | `vit5_small_pretrain_hyena_gap_apex.py` | [`k7lme5pm`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/k7lme5pm) | b65c909e-18 | Running | ~371 | 1.020 | 75.0% | ~2.3 |
| 31011 | `hyena-cls-row` | `vit5_small_pretrain_hyena_cls_row_apex.py` | [`yihu8hx9`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/yihu8hx9) | b65c909e-04 | Running (compile=False) | ~308 | 1.070 | 73.8% | ~2.2 |
| 31087 | `hyena-gap-rms` | `vit5_small_pretrain_hyena_gap_apex_qk_rmsnorm.py` | [`pk1c1ahi`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/pk1c1ahi) | b65c909e-03 | Running | ~92 | 1.320 | 68.5% | ~2.6 |
| 31119 | `hyena-gap-rms-mh` | `vit5_small_pretrain_multihead_hyena_gap_apex_qk_rmsnorm.py` | [`yks03rks`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/yks03rks) | b65c909e-21 | Running | ~5 | 3.880 | 24.3% | ~2.5 |
| 31126 | `hyena-gap-optim` | `vit5_small_pretrain_multihead_hyena_gap_apex_optimized.py` | — | — | Pending | — | — | — | — |

## Completed runs

| Job ID | Job name | Config | W&B run | Node | Final epoch | test/loss | test/acc | it/s |
|--------|----------|--------|---------|------|-------------|-----------|----------|------|
| 30931 | `vit5-apex-refaug` | `vit5_small_pretrain_apex.py` | [`2y06y121`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2y06y121) | b65c909e-21 | 800 | 0.730 | 81.64% | ~3.1 |

## Run descriptions

### `vit5-apex` (30923) — Attention + Apex FusedLAMB
Primary attention baseline with autoresume enabled. Uses ViT5Attention (6 heads, RoPE, QK-norm) + 4 register tokens. CLS token readout.

### `vit5-simd` (30924) — Attention + non-Apex LAMB
Same architecture as `vit5-apex` but uses `torch_optimizer.Lamb` instead of Apex FusedLAMB. Serves as an optimizer ablation — slightly slower due to non-fused optimizer step.

### `vit5-apex-refaug` (30931) — Attention + Apex FusedLAMB (duplicate)
Identical config to `vit5-apex` but with autoresume disabled (fresh run started later). Acts as a second seed / reproducibility check.

### `vit5-hyena` (30969) — Hyena + Apex FusedLAMB
Replaces all ViT5Attention layers with 2D Hyena (CKConvND + SIREN kernel). No register tokens. CLS token is updated via mean-pool of mixed patches each layer. Positional info from absolute PE + SIREN kernel (no RoPE).

### `vit5-hyena-gap` (30995) — Hyena + GAP + Apex FusedLAMB
Same as `vit5-hyena` but removes the CLS token entirely. Classification via global average pooling over final patch representations. Motivated by the observation that the CLS-patch interaction in the Hyena variant is constrained to a mean-pool update, which limits bidirectional information flow.

### `hyena-cls-row` (31003 → 31011) — Hyena + CLS-row + Apex FusedLAMB
CLS token and 13 global register tokens form an extra row prepended to the top of the 2D patch grid, producing a 15×14 spatial grid instead of 14×14. CLS sits at position [0,0] and participates directly in the 2D Hyena convolution — no more mean-pool workaround. Registers persist across layers (truly global, updated by every mixer + MLP block). Launched with `compile=False` due to container `/sbin/ldconfig` issue blocking torch.compile on fresh graph shapes. Job 31003 was replaced by 31011 (resubmitted to a different node).

### `hyena-gap-rms` (31087) — Hyena + GAP + QK RMSNorm
Same architecture as `vit5-hyena-gap` but adds QK RMSNorm (dim=384, full-channel) to the Hyena mixer. Tests whether normalizing the query and key projections improves training stability and final accuracy for the depthwise Hyena variant.

### `hyena-gap-rms-mh` (31119) — Multihead Hyena + GAP + PerHeadRMSNorm
Replaces depthwise CKConvND with CKConvMultiheadND (6 heads, 64 head_dim) for dense cross-channel mixing within each head. Uses PerHeadRMSNorm for QK normalization — each head is normalized independently, matching the attention QK-norm convention. Significantly more expressive mixing than depthwise, but also more parameters in the SIREN kernel (out_dim = num_heads × head_dim × head_dim).

### `hyena-gap-optim` (31126) — Optimized Hyena mixer
Builds on the depthwise Hyena + GAP baseline with three mixer-level changes: (1) SiLU nonlinear gating on the key and value branches (replaces Identity), (2) L2Norm on query (conditionally on key when gate is Identity), and (3) output RMSNorm after the mixer (Mamba2-style). Branch: `dwromero/vit5-imagenet-optimized`.

## Reference baseline

The file `_reference_logs/wandb_export_2026-02-20T18_29_03.071-08_00.csv` contains val/loss curves for a prior attention baseline (`vit_max_small_r4_0.004lr_0.05dp`). Key milestones: epoch 50 → val/loss 1.99, epoch 250 → 1.38, epoch 800 → 0.87.

## How to launch a new run

```bash
sbatch --job-name=<NAME> --exclude=b65c909e-02 scripts/submit.sh examples/vit5_imagenet/<CONFIG>.py
```

## How to monitor

```bash
# Job status
squeue -u dwromero

# Tail stdout log
tail -f logs/<NAME>_<JOBID>.out

# Tail stderr (Triton autotuning, warnings)
tail -f logs/<NAME>_<JOBID>.err
```

## TODOs

- [ ] **Remove `ViT5HyenaAdapter`**: The adapter only reshapes `[B, T, C]` ↔ `[B, H, W, C]` around the `QKVSequenceMixer`, but `QKVSequenceMixer` (and all norms, MLP, LayerScale, DropPath in the residual block) already accept `[B, *spatial_dims, C]`. If the classification net reshapes to `[B, H, W, C]` before the blocks and back after, the adapter is unnecessary and can be deleted entirely.
