# v5_patch — Patch-Size Ablation Tracker

W&B group: [`v5_patch_ablation`](https://wandb.ai/implicit-long-convs/nvsubquadratic?nw=nwuserimplicitlongconvs) (entity `implicit-long-convs`, project `nvsubquadratic`)

## Objective

Measure how **Hyena (O(n log n))** and **standard multi-head attention (O(n²))** scale as sequence length increases via patch-size reduction on ViT-5-Small / ImageNet-1k. Patch sizes 16 → 8 → 4 → 2 → 1 span a ~250× range in token count (201 → 50,181). Attention is expected to become infeasible at the longest sequences, while Hyena should handle all five.

## Architecture

### Shared (ViT-5-Small)

- Hidden dim: 384, 12 blocks, CLS readout
- MLP: GELU, expansion 4×, no bias
- Pre-norm: RMSNorm (eps=1e-6)
- LayerScale init: 1e-4, DropPath: 0.05
- Registers: 4 (fixed across all patch sizes)
- Init: trunc_normal (std=0.02)

### Hyena (CLS-row + FiLM + GRN)

- **Layout:** \[CLS, 4 regs, zero-pad, patches\] reshaped to `(num_patches_h + 1) × num_patches_w` 2D grid
- **Mixer:** QKVSequenceMixer → ViT5HyenaAdapter → Hyena
  - Global conv: CKConvND (2D) + SIREN kernel (3 layers, hidden 32, embed 32, ω₀=10, hidden ω₀=1)
  - Short conv: 3×3 depthwise (groups=3·384)
  - Gates: SiLU (1st) + Sigmoid (2nd) — bilinear mixer
  - QK-norm: L2Norm
  - Output norm: RMSNorm
- **FiLM conditioning:** RegisterPooling → KernelFiLMGenerator (hidden 64, 2 FiLM layers) → modulates SIREN hidden layers
- **GRN:** GlobalResponseNorm after mixer output, before residual (ConvNeXt V2-style inter-channel competition)

### Attention

- **Mixer:** ViT5Attention, 6 heads (head_dim=64)
- **RoPE:** base=10000, register base=100
- **QK-norm:** RMSNorm (per-head, eps=1e-6)
- No bias on QKV or output projections

## Experiment matrix

All configs use 4 registers. For Hyena (CLS-row), the first grid row is zero-padded:
T_hyena = `(H+1) × W`. For Attention (appended regs): T_attn = `1 + 4 + H×W`.

| Config                 | Mixer     | Patch | T (attn) | T (hyena) | Feasibility          |
| ---------------------- | --------- | ----: | -------: | --------: | -------------------- |
| `hyena_patch16.py`     | Hyena     |    16 |        — |       210 | ✓                    |
| `hyena_patch8.py`      | Hyena     |     8 |        — |       812 | ✓                    |
| `hyena_patch4.py`      | Hyena     |     4 |        — |     3,192 | ✓                    |
| `hyena_patch2.py`      | Hyena     |     2 |        — |    12,656 | ✓ (memory-intensive) |
| `hyena_patch1.py`      | Hyena     |     1 |        — |    50,400 | ✓ (very intensive)   |
| `attention_patch16.py` | Attention |    16 |      201 |         — | ✓                    |
| `attention_patch8.py`  | Attention |     8 |      789 |         — | ✓                    |
| `attention_patch4.py`  | Attention |     4 |    3,141 |         — | ✓                    |
| `attention_patch2.py`  | Attention |     2 |   12,549 |         — | ⚠ likely OOM on H100 |
| `attention_patch1.py`  | Attention |     1 |   50,181 |         — | ✗ almost certainly   |

## Batch configuration

Effective batch = 8 GPUs × batch/gpu × accum = 2048 for all configs.

| Patch | batch/gpu | accum | compile_mode               |
| ----: | --------: | ----: | :------------------------- |
|    16 |       256 |     1 | max-autotune               |
|     8 |        64 |     4 | max-autotune-no-cudagraphs |
|     4 |        16 |    16 | max-autotune-no-cudagraphs |
|     2 |         4 |    64 | max-autotune-no-cudagraphs |
|     1 |         1 |   256 | max-autotune-no-cudagraphs |

CUDA graphs are incompatible with gradient accumulation, so only patch 16 uses `max-autotune`.

## Training recipe

Identical across all 10 configs:

- **Optimizer:** Apex FusedLAMB, lr=4e-3, wd=0.05
- **Schedule:** Cosine, 800 epochs, 5-epoch warmup (0.625%)
- **Precision:** bf16-mixed
- **EMA:** decay=0.99996, monitored metric: val/acc_ema
- **Augmentation:** ThreeAugment (color_jitter=0.3) + Mixup(0.8) + CutMix(1.0)
- **Loss:** SoftTargetCE
- **Gradient clip:** 1.0
- **Validation:** every 4 epochs
- **Checkpoints:** every 5000 steps
- **Data pipeline:** DALI fused (GPU decode/augment) + local NVMe staging
- **Seed:** 42

______________________________________________________________________

## Hyena runs

| Patch | Config          | Job ID | W&B run | Node | Status  | Epoch | val/loss | val/acc_ema | it/s |
| ----: | --------------- | ------ | ------- | ---- | ------- | ----- | -------- | ----------- | ---- |
|    16 | `hyena_patch16` | —      | —       | —    | Pending | —     | —        | —           | —    |
|     8 | `hyena_patch8`  | —      | —       | —    | Pending | —     | —        | —           | —    |
|     4 | `hyena_patch4`  | —      | —       | —    | Pending | —     | —        | —           | —    |
|     2 | `hyena_patch2`  | —      | —       | —    | Pending | —     | —        | —           | —    |
|     1 | `hyena_patch1`  | —      | —       | —    | Pending | —     | —        | —           | —    |

## Attention runs

| Patch | Config              | Job ID | W&B run | Node | Status  | Epoch | val/loss | val/acc_ema | it/s |
| ----: | ------------------- | ------ | ------- | ---- | ------- | ----- | -------- | ----------- | ---- |
|    16 | `attention_patch16` | —      | —       | —    | Pending | —     | —        | —           | —    |
|     8 | `attention_patch8`  | —      | —       | —    | Pending | —     | —        | —           | —    |
|     4 | `attention_patch4`  | —      | —       | —    | Pending | —     | —        | —           | —    |
|     2 | `attention_patch2`  | —      | —       | —    | ⚠ OOM?  | —     | —        | —           | —    |
|     1 | `attention_patch1`  | —      | —       | —    | ⚠ OOM?  | —     | —        | —           | —    |

______________________________________________________________________

## Feasibility notes

- **attention_patch2:** O(n²) on ~12.5K tokens. Likely to OOM on H100 80GB. If it fits, expect very slow it/s.
- **attention_patch1:** O(n²) on ~50.2K tokens. Almost certainly infeasible on H100 80GB.
- **Hyena patch2/patch1:** Long sequences will be memory-intensive but should be tractable thanks to O(n log n) FFT-based convolution. The grid is `(H+1) × W` with a zero-padded first row. May need to verify that SIREN kernel generation at L_cache=113 (patch2) and L_cache=225 (patch1) remains efficient.

## Key hypotheses

1. **Crossover point:** Hyena should match or beat attention accuracy at patch 16/8, and increasingly dominate at patch 4/2/1 where attention either OOMs or is prohibitively slow.
1. **Throughput scaling:** Hyena it/s should degrade more gracefully than attention as tokens increase. The gap should widen dramatically at patch 2/1.
1. **Accuracy vs resolution:** Smaller patches provide finer spatial resolution — both methods should improve in accuracy as patch size shrinks (until training becomes infeasible or underfitting due to extreme sequence lengths).
1. **FiLM + GRN value:** These components (absent in attention) may give Hyena an edge beyond just feasibility — particularly at long sequences where register conditioning and inter-channel competition have more spatial context to leverage.
1. **Attention OOM boundary:** Attention should run at patch 16/8/4, become borderline at patch 2, and fail at patch 1. This directly demonstrates the practical scaling advantage.

## TODOs

- [ ] Launch Hyena patch16 + Attention patch16 as first pair (sanity check)
- [ ] Launch patch8 pair once patch16 looks healthy
- [ ] Progressively launch patch4, then patch2, then patch1
- [ ] Test attention_patch2 feasibility — if OOM, try reduced batch (batch=2, accum=128)
- [ ] Confirm attention_patch1 is infeasible (document the OOM)
- [ ] Fill in it/s for each run to build the throughput scaling curve
- [ ] Record peak memory usage per config (for the scaling analysis)
- [ ] Add a results summary section once runs complete
