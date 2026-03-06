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
| `…hyena_cls_row_gated_film_rope_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | **Yes** | **Yes** (32h SIREN, 64d FiLM) | No | L2Norm | Adds RoPE + FiLM conditioning |
| `…hyena_cls_row_gated_film_rope_wider_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | **Yes** | **Yes** (64h SIREN, 128d FiLM, 4 layers) | No | L2Norm | Wider SIREN/FiLM for more kernel capacity |
| `…multihead_hyena_cls_row_gated_rope_ema.py` | Multi-head Hyena (SiLU/Sigmoid) | CKConvMultiheadND (6h, d=64) | **Yes** | No | No | PerHeadRMSNorm | Adds RoPE to multi-head variant |
| `…hyena_cls_row_gated_film_rope_grn_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | **Yes** | **Yes** (32h SIREN, 64d FiLM) | **Yes** | L2Norm | FiLM + RoPE + GRN (inter-channel competition) |
| `…hyena_cls_row_gated_film_rope_wider_grn_ema.py` | Hyena (SiLU/Sigmoid) | CKConvND (depthwise) | **Yes** | **Yes** (64h SIREN, 128d FiLM, 4 layers) | **Yes** | L2Norm | Wider FiLM + RoPE + GRN combined |

All configs are self-contained (no `_pretrain_base` import), use PyTorch default init (Kaiming uniform), share the same training recipe, and include EMA (`decay=0.99996`) with `val/acc_ema` checkpoint monitoring.

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

| Job ID | Job name | Config | W&B run | Node | Status | Epoch | val/acc_ema | Notes |
|--------|----------|--------|---------|------|--------|-------|-------------|-------|
| — | — | `…apex_gated_ema.py` | — | — | Not started | — | — | Baseline gated Hyena |
| — | — | `…multihead_…_gated_ema.py` | — | — | Not started | — | — | Baseline multi-head |
| — | — | `…gated_film_rope_ema.py` | — | — | Not started | — | — | FiLM + RoPE |
| — | — | `…gated_film_rope_wider_ema.py` | — | — | Not started | — | — | Wider FiLM + RoPE |
| — | — | `…gated_rope_ema.py` (multihead) | — | — | Not started | — | — | Multi-head + RoPE |
| — | — | `…gated_film_rope_grn_ema.py` | — | — | Not started | — | — | FiLM + RoPE + GRN |
| — | — | `…gated_film_rope_wider_grn_ema.py` | — | — | Not started | — | — | Wider FiLM + RoPE + GRN |

## How to launch

```bash
sbatch --job-name=<NAME> scripts/submit.sh examples/vit5_imagenet/v3_wessels/<CONFIG>.py
```

## TODOs

- [ ] Launch all three experiments and fill in run table
- [ ] After initial results (~100-200 epochs), compare learning curves against baselines on W&B
- [ ] If FiLM + RoPE shows promise: add FiLM support to `CKConvMultiheadND` (forward `**mixer_kwargs` to kernel) and create a multi-head + FiLM + RoPE config
- [ ] If wider SIREN helps: try even wider (128h) or deeper (5-6 layers) variants
- [ ] **GRN placement revisit**: In ConvNeXt V2, GRN is placed *inside* the MLP (after expansion + activation, before projection) and *replaces* LayerScale (found redundant). Our current config adds GRN after the mixer output while keeping LayerScale. If GRN shows promise, test: (a) moving GRN inside MLP, (b) dropping LayerScale (`layer_scale_init=0`), or (c) both — to match ConvNeXt V2 more closely.
