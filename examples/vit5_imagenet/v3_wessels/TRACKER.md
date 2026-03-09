# v3_wessels — Hyena Expressivity Ablations

## Goal

Close the ~0.5% ImageNet-1k accuracy gap between the ViT-5-Small Hyena variant (~81.7%) and the attention baseline (~82.2%). All experiments target the sequence mixer block only — the ViT-5 backbone (12 blocks, dim 384, patch 16, CLS-row layout) is unchanged.

Three expressivity gaps were identified in the current best Hyena config:

1. **No positional encoding in gating**: The `Q * SiLU(K)` gate has zero positional information (`use_rope=False`), unlike attention which uses 2D RoPE on Q and K. This means the gate weights are position-agnostic.
2. **Narrow SIREN kernel + FiLM bottleneck**: The SIREN kernel network (32 hidden dim, 3 layers) and FiLM conditioning bottleneck (64 dim) may limit the kernel's function approximation capacity.
3. **Depthwise-only global convolution**: CKConvND convolves each of 384 channels independently, unlike attention's per-head dense mixing across `head_dim=64` channels.
4. **Feature collapse in depthwise conv** (ConvNeXt V2, Woo et al. 2023): Depthwise convolutions produce redundant/dead channel features. GRN (Global Response Normalization) addresses this via divisive normalization across channels, promoting inter-channel feature competition.

## Config files

| Config | Mixer | Conv type | RoPE | FiLM | GRN | QK norm | Key change vs baseline |
|--------|-------|-----------|------|------|-----|---------|----------------------|
| `…hyena_cls_row_apex_gated_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | No | No | No | L2Norm | **Baseline** — existing gated Hyena |
| `…multihead_hyena_cls_row_gated_ema.py` | Multi-head Hyena (SiLU/Sigmoid) | CKConvMultiheadND (6h, d=64) | No | No | No | PerHeadRMSNorm | **Baseline** — existing multi-head gated Hyena |
| `…hyena_cls_row_gated_film_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | No | **Yes** (32h SIREN, 64d FiLM) | No | L2Norm | FiLM only — isolates FiLM contribution without RoPE |
| `…hyena_cls_row_gated_film_rope_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | **Yes** | **Yes** (32h SIREN, 64d FiLM) | No | L2Norm | Adds RoPE + FiLM conditioning |
| `…hyena_cls_row_gated_film_rope_wider_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | **Yes** | **Yes** (64h SIREN, 128d FiLM, 4 layers) | No | L2Norm | Wider SIREN/FiLM for more kernel capacity |
| `…multihead_hyena_cls_row_gated_rope_ema.py` | Multi-head Hyena (SiLU/Sigmoid) | CKConvMultiheadND (6h, d=64) | **Yes** | No | No | PerHeadRMSNorm | Adds RoPE to multi-head variant |
| `…hyena_cls_row_gated_film_grn_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | No | **Yes** (32h SIREN, 64d FiLM) | **Yes** | L2Norm | FiLM + GRN — isolates GRN without RoPE |
| `…hyena_cls_row_gated_film_rope_grn_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | **Yes** | **Yes** (32h SIREN, 64d FiLM) | **Yes** | L2Norm | FiLM + RoPE + GRN (inter-channel competition) |
| `…hyena_cls_row_gated_film_rope_wider_grn_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | **Yes** | **Yes** (64h SIREN, 128d FiLM, 4 layers) | **Yes** | L2Norm | Wider FiLM + RoPE + GRN combined |
| `…hyena_cls_row_reghead_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | No | No | No | L2Norm | **Register recycling** (depthwise) — no CLS, 14 regs, reduction head |
| `…hyena_cls_row_reghead_film_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | No | **Yes** (32h SIREN, 64d FiLM) | No | L2Norm | Register recycling (depthwise) + FiLM |
| `…multihead_hyena_cls_row_gated_film_ema.py` | Multi-head Hyena (SiLU/Sigmoid) | CKConvMultiheadND (6h, d=64) | No | **Yes** (32h SIREN, 64d FiLM) | No | PerHeadRMSNorm | Multi-head + FiLM |
| `…multihead_hyena_cls_row_gated_film_rope_ema.py` | Multi-head Hyena (SiLU/Sigmoid) | CKConvMultiheadND (6h, d=64) | **Yes** | **Yes** (32h SIREN, 64d FiLM) | No | PerHeadRMSNorm | Multi-head + FiLM + RoPE |
| `…multihead_hyena_cls_row_gated_film_rope_grn_ema.py` | Multi-head Hyena (SiLU/Sigmoid) | CKConvMultiheadND (6h, d=64) | **Yes** | **Yes** (32h SIREN, 64d FiLM) | **Yes** | PerHeadRMSNorm | Multi-head + FiLM + RoPE + GRN |
| `…multihead_hyena_cls_row_reghead_ema.py` | Multi-head Hyena (SiLU/Sigmoid) | CKConvMultiheadND (6h, d=64) | No | No | No | PerHeadRMSNorm | **Register recycling** (multihead) — no CLS, 14 regs, reduction head |
| `…multihead_hyena_cls_row_reghead_film_ema.py` | Multi-head Hyena (SiLU/Sigmoid) | CKConvMultiheadND (6h, d=64) | No | **Yes** (32h SIREN, 64d FiLM) | No | PerHeadRMSNorm | Register recycling (multihead) + FiLM |

New multi-head configs import from `_base_config.py` to reduce duplication. Depthwise configs remain self-contained. All configs use PyTorch default init (Kaiming uniform), share the same training recipe, and include EMA (`decay=0.99996`) with `val/acc_ema` checkpoint monitoring.

## Shared training recipe

ViT-5-Small (12 blocks, dim 384, patch 16, 224x224), CLS-row (15x14 grid, 13 registers), Apex FusedLAMB lr=4e-3, wd=0.05, batch 2048, 800 epochs, cosine schedule, 5 warmup epochs, 3-Augment, Mixup 0.8 + CutMix 1.0, soft-target CE loss, DropPath 0.05, LayerScale 1e-4, bf16-mixed, `torch.compile(mode="max-autotune")`.

## Experiment details

### 1. `vit5_small_pretrain_hyena_cls_row_gated_film_rope.py` — FiLM + RoPE

Adds two features to the depthwise gated Hyena baseline:
- **2D RoPE** (`use_rope=True`): Applies rotary positional encoding to Q and K before gating, giving the `Q * SiLU(K)` gate position-dependent weights.
- **FiLM conditioning**: `RegisterPooling` pools 13 register tokens into a [B, 384] conditioning vector via learnable softmax-weighted average. `KernelFiLMGenerator` (384 -> 64 -> GELU -> 2 * 2 * 32) produces per-layer (gamma, beta) pairs that modulate SIREN hidden layers, making convolution kernels input-dependent.
- Uses `compile_compatible_fftconv=True` for torch.compile compatibility.

**Expected impact**: RoPE provides the largest low-hanging fruit (already in the attention baseline, zero extra params). FiLM adds ~100K params but enables input-dependent kernels — the main expressivity advantage we want to test.

### 2. `vit5_small_pretrain_hyena_cls_row_gated_film_rope_wider.py` — wider FiLM + RoPE

Same as (1) but with increased kernel network capacity:
- SIREN hidden dim: 32 -> 64
- SIREN layers: 3 -> 4 (one more FiLM-conditioned hidden layer)
- FiLM bottleneck dim: 64 -> 128

Adds ~600K parameters total (~3% of 22M model) while significantly increasing the kernel's function approximation capacity and the FiLM conditioning bandwidth.

**Expected impact**: Tests whether the baseline SIREN/FiLM network is a bottleneck. If (1) improves over the baseline but still trails attention, the wider variant tells us whether more kernel capacity helps.

### 3. `vit5_small_pretrain_multihead_hyena_cls_row_gated_rope.py` — Multi-head + RoPE

Adds 2D RoPE to the multi-head gated Hyena baseline (CKConvMultiheadND, 6 heads, head_dim=64). No FiLM — CKConvMultiheadND doesn't forward `**mixer_kwargs` to the kernel, so FiLM support would require a code change.

Uses `PerHeadRMSNorm` for QK normalization and `find_unused_parameters=True` in the trainer.

**Expected impact**: Isolates the RoPE contribution on the multi-head architecture, which already has dense within-head channel mixing (closer to attention). If multi-head + RoPE closes the gap, it suggests the expressivity bottleneck is in gating (positional info) rather than kernel capacity.

### 4. `vit5_small_pretrain_hyena_cls_row_gated_film_rope_grn_ema.py` — FiLM + RoPE + GRN

Same as (1) but adds Global Response Normalization (GRN) from ConvNeXt V2 (Woo et al., 2023). GRN is applied after the Hyena mixer output (before LayerScale) in each residual block.

GRN promotes inter-channel feature competition via divisive normalization:
1. Compute L2 norm per channel across all spatial/token positions
2. Normalize these norms across channels (divisive normalization)
3. Scale original features by normalized response + learnable gamma/beta (zero-initialized)

This directly addresses the feature collapse problem in depthwise convolutions where channels become redundant. Negligible extra parameters (just gamma and beta per block, 384 * 2 * 12 = 9.2K total).

**Expected impact**: GRN is orthogonal to FiLM/RoPE — it operates at the channel level while FiLM operates at the kernel level and RoPE at the positional level. If the depthwise conv produces redundant features, GRN should improve feature utilization. In ConvNeXt V2, GRN was the single architectural change that closed the gap with attention-based models.

### 5. `vit5_small_pretrain_hyena_cls_row_gated_film_rope_wider_grn_ema.py` — Wider FiLM + RoPE + GRN

Combines the wider SIREN/FiLM kernel from (2) with the GRN from (4):
- SIREN hidden dim: 64, layers: 4, FiLM bottleneck: 128d
- GRN after mixer output in each residual block (zero-init gamma/beta)

**Expected impact**: Tests whether the two orthogonal improvements (wider kernel capacity + channel competition) are complementary. The best outcome would be additive gains over the individual contributions.

## Runs

> Last updated: 2026-03-09 ~10:36. Epochs/accuracy are snapshots from logs; see W&B for live curves.

| Job ID | Config | W&B run | Status | Epoch | val/acc_ema (best) | Notes |
|--------|--------|---------|--------|-------|---------------------|-------|
| 20579151 | `…hyena_cls_row_gated_film_ema.py` | [69gwd0xm](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/69gwd0xm) | **Running** | ~175 | **0.735** (0.7353 @ ep175) | FiLM, no RoPE — leading run |
| 20583567 | `…hyena_cls_row_gated_film_rope_ema.py` | [o2ojne5x](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o2ojne5x) | ~~Cancelled~~ | ~95 | 0.103 (0.194 @ ep47) | FiLM + RoPE — cancelled; RoPE mismatched to elementwise gating i.o. standard dot-product. |
| 20583569 | `…hyena_cls_row_gated_film_rope_grn_ema.py` | [swr33zw4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/swr33zw4) | ~~Cancelled~~ | ~88 | 0.205 (0.276 @ ep63) | FiLM + RoPE + GRN — cancelled; same RoPE issue |
| — | `…hyena_cls_row_gated_film_grn_ema.py` | — | Not started | — | — | FiLM + GRN, no RoPE — replaces cancelled RoPE+GRN run |
| 20583570 | `…hyena_cls_row_reghead_film_ema.py` | [awstn87e](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/awstn87e) | **Running** | ~96 | 0.575 (0.541 @ ep87) | Register head + FiLM — strong, improving |
| 20583571 | `…multihead_hyena_cls_row_gated_film_ema.py` | [r4iwlaxt](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/r4iwlaxt) | **Failed** | — | — | Disk quota / SIGTERM during compile (~7min runtime) — needs resubmit |
| — | `…apex_gated_ema.py` | — | Not started | — | — | Baseline gated Hyena |
| — | `…multihead_…_gated_ema.py` | — | Not started | — | — | Baseline multi-head |
| — | `…gated_film_rope_wider_ema.py` | — | Not started | — | — | Wider FiLM + RoPE |
| — | `…multihead_…_gated_rope_ema.py` | — | Not started | — | — | Multi-head + RoPE |
| — | `…gated_film_rope_wider_grn_ema.py` | — | Not started | — | — | Wider FiLM + RoPE + GRN |
| — | `…hyena_cls_row_reghead_ema.py` | — | Not started | — | — | Register recycling (depthwise, no FiLM) |
| — | `…multihead_…_gated_film_rope_ema.py` | — | Not started | — | — | Multi-head + FiLM + RoPE |
| — | `…multihead_…_gated_film_rope_grn_ema.py` | — | Not started | — | — | Multi-head + FiLM + RoPE + GRN |
| — | `…multihead_…_reghead_ema.py` | — | Not started | — | — | Register recycling (multihead) |
| — | `…multihead_…_reghead_film_ema.py` | — | Not started | — | — | Register recycling (multihead) + FiLM |

## Initial analysis (epoch ~90–175, 2026-03-09)

### `…hyena_cls_row_gated_film_ema.py` — FiLM, no RoPE ✅ Leading

At epoch 175, val/acc_ema = **73.5%** and still climbing. Train acc epoch is ~0.498, indicating the model is not yet at peak. Trajectory looks healthy — loss decreasing smoothly, no instability. This is the only run deep enough to give a meaningful signal; all others started ~8h later.

**Key observation**: FiLM alone (without RoPE) is training stably and well. This isolates the FiLM contribution and sets a strong baseline for comparing the RoPE variants.

### `…hyena_cls_row_gated_film_rope_ema.py` — FiLM + RoPE ⚠️ Underperforming

Training accuracy at epoch 95 is ~46.7% (reasonable), but val/acc_ema has **collapsed from a peak of 19.4% at epoch 47 down to 10.3%**. The gap between train acc and val/acc_ema (47% vs 10%) is far too large to be explained by overfitting alone.

**Most likely cause — RoPE is architecturally mismatched with elementwise gating.** In attention, RoPE works because the inner product `QKᵀ` has a clean geometric property: rotating Q and K by position-dependent angles encodes relative position into the similarity score, and the rotations cancel algebraically in the dot product. In Hyena's `Q ⊙ SiLU(K)` gate, the operation is elementwise, not a dot product. Rotating Q and K by different position-dependent angles before elementwise-multiplying them has no such cancellation — the rotations don't simplify, they just add structured noise to the gate values. The model may be learning despite this, but the positional information isn't being used in a principled way.

The EMA degradation (peak at ep47, then down to ep95) is likely a symptom, not the cause: if the non-EMA model converges to a poor solution, the EMA faithfully reflects it. The train accuracy of 46.7% is measured on Mixup/CutMix soft-label batches, which is a poor proxy for actual model quality — val/acc_ema on clean ImageNet is the honest number.

**Action**: Check W&B for non-EMA `val/acc`. If it's also ~10%, RoPE is genuinely hurting the model; if it's ~35–40%, the EMA mechanics are the issue and a lower decay or delayed EMA start would help. Bet is on the former.

### `…hyena_cls_row_gated_film_rope_grn_ema.py` — FiLM + RoPE + GRN ⚠️ Same issue, slightly better

Same pattern: val/acc_ema peaked at 27.6% (epoch 63) and dropped to 20.5% at epoch 88. The GRN variant consistently beats the no-GRN RoPE variant at comparable epochs (27.6% vs 19.4% peak), suggesting GRN provides real benefit even in the presence of a problematic RoPE — but the fundamental issue of RoPE being mismatched to elementwise gating remains.

### `…hyena_cls_row_reghead_film_ema.py` — Register head + FiLM ✅ Promising

At epoch 96, val/acc_ema = **57.5%** (best so far 54.1% at epoch 87, improved since). This is a strong result at epoch 96 — at the same epoch, the FiLM-only run (20579151) was around 40–45%. The register reduction head (Mamba-R style) combined with FiLM conditioning appears effective. No EMA instability observed.

**Key observation**: Depthwise variant with register head outpacing depthwise with FiLM at the same epoch count. Worth watching whether this gap persists at epoch 175.

### `…multihead_hyena_cls_row_gated_film_ema.py` — Multi-head + FiLM ❌ Failed

Killed after ~7 minutes (before first epoch completed), likely disk quota during Triton backward-pass compilation — same issue as the previous batch. The triton cache fix (`TRITON_CACHE_DIR=/tmp/...`) may not have been in effect for this job if it was submitted before the script was updated. **Needs resubmit.**

## How to launch

```bash
sbatch --job-name=<NAME> scripts/submit.sh examples/vit5_imagenet/v3_wessels/<CONFIG>.py
```

## TODOs

- [ ] Launch all three experiments and fill in run table
- [ ] After initial results (~100-200 epochs), compare learning curves against baselines on W&B
- [x] Add FiLM support to `CKConvMultiheadND` (forward `**mixer_kwargs` to kernel) and create multi-head + FiLM configs
- [ ] If wider SIREN helps: try even wider (128h) or deeper (5-6 layers) variants
- [ ] **GRN placement revisit**: In ConvNeXt V2, GRN is placed *inside* the MLP (after expansion + activation, before projection) and *replaces* LayerScale (found redundant). Our current config adds GRN after the mixer output while keeping LayerScale. If GRN shows promise, test: (a) moving GRN inside MLP, (b) dropping LayerScale (`layer_scale_init=0`), or (c) both — to match ConvNeXt V2 more closely.
- [ ] **Register recycling — MAP variant**: If `…reghead_ema.py` shows promise, try Multi-head Attention Pooling (MAP) over registers as the classification head. MAP uses a learnable query that cross-attends to all 14 register outputs (used in CoCa, DINOv2). More expressive than concat+reduce but adds ~3C² parameters. Implement as `RegisterMAP(hidden_dim, num_heads)` in `nvsubquadratic/modules/register_map.py` with a new config `…multihead_hyena_cls_row_regmap_ema.py`.
