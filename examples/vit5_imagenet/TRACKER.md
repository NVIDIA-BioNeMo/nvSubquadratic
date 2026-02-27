# ViT-5 Small ImageNet-1k — Experiment Tracker

W&B project: [`implicit-long-convs/nvsubquadratic`](https://wandb.ai/implicit-long-convs/nvsubquadratic)

## Config files

| Config | Mixer | Optimizer | Dataloader | CLS token | Registers | Notes |
|--------|-------|-----------|------------|-----------|-----------|-------|
| `vit5_small_pretrain.py` | ViT5Attention | `torch_optimizer.Lamb` | PyTorch CPU | Yes | 4 | Baseline attention, non-Apex LAMB |
| `vit5_small_pretrain_apex.py` | ViT5Attention | Apex `FusedLAMB` | PyTorch CPU | Yes | 4 | Attention + fused LAMB |
| `vit5_small_pretrain_apex_dali_fused.py` | ViT5Attention | Apex `FusedLAMB` | **DALI fused** (local NVMe) | Yes | 4 | All augments in DALI pipeline + local NVMe staging |
| `vit5_small_pretrain_hyena_apex.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | PyTorch CPU | Yes (mean-pool update) | 0 | Hyena replaces attention; CLS updated via mean-pool |
| `vit5_small_pretrain_hyena_gap_apex.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | PyTorch CPU | **No** (GAP) | 0 | CLS removed; classification via global average pooling |
| `vit5_small_pretrain_hyena_gap_apex_optimized.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | PyTorch CPU | **No** (GAP) | 0 | Hyena-GAP with all pipeline optimizations |
| `vit5_small_pretrain_hyena_gap_apex_qk_rmsnorm.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | PyTorch CPU | **No** (GAP) | 0 | Hyena-GAP + RMSNorm QK normalization |
| `vit5_small_pretrain_hyena_cls_row_apex.py` | Hyena (CKConvND + SIREN) | Apex `FusedLAMB` | PyTorch CPU | Yes (in-grid) | 13 (prepended) | CLS + 13 registers as extra row → 15×14 grid |
| `VALIDATION_vit5_small_dali_fused.py` | ViT5Attention | — | DALI fused | Yes | 4 | Validation-only config for checkpoint testing |
| `v2/vit5_small_pretrain_hyena_gap_apex.py` | Hyena (SiLU gate, output RMSNorm) | Apex `FusedLAMB` | **DALI fused** (local NVMe) | **No** (GAP) | 0 | v2 Hyena-GAP: SiLU gate + output RMSNorm + DALI fused |
| `v2/vit5_small_pretrain_hyena_cls_row_apex.py` | Hyena (SiLU gate, output RMSNorm) | Apex `FusedLAMB` | **DALI fused** (local NVMe) | Yes (in-grid) | 13 (prepended) | v2 Hyena-CLS-row: SiLU gate + output RMSNorm + DALI fused |

All training configs share: ViT-5-Small (12 blocks, dim 384, patch 16, 224x224), LAMB lr=4e-3, wd=0.05, batch 2048, 800 epochs, cosine schedule, 5 warmup epochs, 3-Augment, Mixup 0.8 + CutMix 1.0, BCE loss, DropPath 0.05, LayerScale 1e-4, bf16-mixed.

## Active runs (as of 2026-02-27)

| Job ID | Job name | Config | W&B run | Node | Status | Epoch | val/loss | val/acc | it/s |
|--------|----------|--------|---------|------|--------|-------|----------|---------|------|
| 32158 | `vit5-dali-fused` | `vit5_small_pretrain_apex_dali_fused.py` | `ky33` | b65c909e-38 | Running | ~316 | 0.994 | 74.5% | ~12.4 |

### v2 runs — Hyena mixer ablations (DALI fused + local NVMe)

All v2 runs use the DALI fused data pipeline with local NVMe staging, SiLU gate nonlinearity (unless overridden), output RMSNorm, L2 QK-norm, and validate every 4 epochs.

| Job ID | Job name | Config | Gate | W&B run | Node | Status | Epoch | val/loss | val/acc | it/s |
|--------|----------|--------|------|---------|------|--------|-------|----------|---------|------|
| 32336 | `v2-hyena-gap` | `v2/…hyena_gap_apex.py` | SiLU | [`c3mbeoc5`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/c3mbeoc5) | b65c909e-01 | Running | 0 | — | — | — |
| 32337 | `v2-hyena-cls-row` | `v2/…hyena_cls_row_apex.py` | SiLU | [`96wy1zzj`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/96wy1zzj) | b65c909e-20 | Running | 0 | — | — | — |
| 32339 | `v2-hyena-gap-idgate` | `v2/…hyena_gap_apex.py` + CLI override | Identity | [`eljt4gx6`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/eljt4gx6) | b65c909e-06 | Running | 0 | — | — | — |

## Cancelled runs (second generation)

| Job ID | Job name | Config | W&B run | Final epoch | val/loss | val/acc | Notes |
|--------|----------|--------|---------|-------------|----------|---------|-------|
| 31829 | `vit5-dali` | `_deprecated/vit5_small_pretrain_apex_dali.py` | `4on8` | ~644 | 0.756 | 81.1% | Attention + DALI v1 (network FS). Cancelled to free node. |
| 31870 | `vit5-dali-v2` | `_deprecated/vit5_small_pretrain_apex_dali_optimized_v2.py` | `lp1q` | ~643 | 0.740 | 81.2% | Attention + DALI optimized (network FS). Cancelled to free node. |
| 31147 | `hyena-gap-optim` | `vit5_small_pretrain_hyena_gap_apex_optimized.py` | `fg5d` | ~707 | 0.929 | 78.7% | Hyena-GAP optimized. Cancelled — superseded by v2 runs. |
| 31221 | `mh-hyena-optim` | multihead Hyena-GAP optimized | `6ecn` | ~531 | 0.936 | 78.1% | Multi-head Hyena-GAP. Cancelled — superseded by v2 runs. |

## Completed runs (first generation)

| Job ID | Job name | Config | W&B run | Final epoch | val/loss | val/acc | Notes |
|--------|----------|--------|---------|-------------|----------|---------|-------|
| 30923 | `vit5-apex` | `vit5_small_pretrain_apex.py` | [`ia7b26u7`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ia7b26u7) | 800 | — | ~81.7% | Attention baseline (Network FS, CPU dataloader) |
| 30924 | `vit5-simd` | `vit5_small_pretrain.py` | [`ea0z7ttf`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ea0z7ttf) | 800 | — | — | Non-Apex LAMB optimizer ablation |
| 30931 | `vit5-apex-refaug` | `vit5_small_pretrain_apex.py` | [`2y06y121`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2y06y121) | 800 | — | ~81.7% | Second seed / reproducibility check |
| 30969 | `vit5-hyena` | `vit5_small_pretrain_hyena_apex.py` | [`r9pmc5ps`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/r9pmc5ps) | 800 | — | — | Hyena + CLS mean-pool |
| 30995 | `vit5-hyena-gap` | `vit5_small_pretrain_hyena_gap_apex.py` | [`k7lme5pm`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/k7lme5pm) | 800 | — | — | Hyena + GAP |
| 31003 | `hyena-cls-row` | `vit5_small_pretrain_hyena_cls_row_apex.py` | [`lx5yhn7t`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lx5yhn7t) | 800 | — | — | Hyena + CLS-in-grid (compile=False) |

## Run descriptions

### Active runs

#### `vit5-dali-fused` (32158) — Attention + DALI fused (local NVMe)
Attention baseline using the fully fused DALI pipeline with all augmentations in the DALI pipeline + local NVMe staging. Fastest data pipeline: ~12.4 it/s (2.4x v1). Started 2026-02-26. Still catching up but on track.

#### `v2-hyena-gap` (32336) — Hyena-GAP v2 (SiLU gate + output RMSNorm)
Hyena-GAP with two mixer-level changes vs v1: SiLU gate nonlinearity (adds nonlinearity to the otherwise bilinear mixer) and output RMSNorm (Mamba2-style stabilization before the residual stream). Uses DALI fused pipeline with local NVMe staging. Validates every 4 epochs.

#### `v2-hyena-cls-row` (32337) — Hyena-CLS-row v2 (SiLU gate + output RMSNorm)
Same v2 mixer changes as above but with the CLS-row architecture: CLS + 13 registers as an extra row prepended to the 2D patch grid (15×14). Uses DALI fused pipeline with local NVMe staging.

#### `v2-hyena-gap-idgate` (32338) — Hyena-GAP v2 ablation (Identity gate)
Same as `v2-hyena-gap` but with the gate nonlinearity set to Identity via CLI override, isolating the effect of the output RMSNorm without the SiLU gate.

### Completed runs (first generation)

#### `vit5-apex` (30923) — Attention + Apex FusedLAMB
Primary attention baseline with autoresume enabled. Uses ViT5Attention (6 heads, RoPE, QK-norm) + 4 register tokens. CLS token readout.

#### `vit5-simd` (30924) — Attention + non-Apex LAMB
Same architecture as `vit5-apex` but uses `torch_optimizer.Lamb` instead of Apex FusedLAMB. Serves as an optimizer ablation — slightly slower due to non-fused optimizer step.

#### `vit5-apex-refaug` (30931) — Attention + Apex FusedLAMB (duplicate)
Identical config to `vit5-apex` but with autoresume disabled (fresh run started later). Acts as a second seed / reproducibility check.

#### `vit5-hyena` (30969) — Hyena + Apex FusedLAMB
Replaces all ViT5Attention layers with 2D Hyena (CKConvND + SIREN kernel). No register tokens. CLS token is updated via mean-pool of mixed patches each layer. Positional info from absolute PE + SIREN kernel (no RoPE).

#### `vit5-hyena-gap` (30995) — Hyena + GAP + Apex FusedLAMB
Same as `vit5-hyena` but removes the CLS token entirely. Classification via global average pooling over final patch representations. Motivated by the observation that the CLS-patch interaction in the Hyena variant is constrained to a mean-pool update, which limits bidirectional information flow.

#### `hyena-cls-row` (31003) — Hyena + CLS-row + Apex FusedLAMB
CLS token and 13 global register tokens form an extra row prepended to the top of the 2D patch grid, producing a 15×14 spatial grid instead of 14×14. CLS sits at position [0,0] and participates directly in the 2D Hyena convolution — no more mean-pool workaround. Registers persist across layers (truly global, updated by every mixer + MLP block). Launched with `compile=False` due to container `/sbin/ldconfig` issue blocking torch.compile on fresh graph shapes.

## Reference baseline

The file `_reference_logs/wandb_export_2026-02-20T18_29_03.071-08_00.csv` contains val/loss curves for a prior attention baseline (`vit_max_small_r4_0.004lr_0.05dp`). Key milestones: epoch 50 → val/loss 1.99, epoch 250 → 1.38, epoch 800 → 0.87.

## How to launch a new run

```bash
sbatch --job-name=<NAME> scripts/submit.sh examples/vit5_imagenet/<CONFIG>.py
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
- [ ] **Fill in completed run final metrics**: Check W&B for final val/loss and val/acc of generation-1 runs (30923–31003) once confirmed.
