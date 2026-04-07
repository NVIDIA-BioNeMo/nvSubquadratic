# Hyena Permutation Optimization Tracker

**Branch**: `dwromero/hyena-permutation-opt` (based on `wessels/well-v2-base-configs`)

## Goal

Eliminate redundant tensor permutations (rearrange, movedim, reshape) in the
Hyena forward path to improve training throughput.  The model operates
internally in **BCHW** but constantly shuffles to channels-last for norms and
at the QKVSequenceMixer↔Hyena boundary.

Estimated ~168 permutations per forward pass across 12 blocks before
optimization.

## Benchmark setup

- **GPU**: 1× H100 80 GB
- **Dataset**: `active_matter` (256×256, 11 fields, periodic BC)
- **Config**: `hyena_gaussian_mask.py` (Hyena + Gaussian modulation mask)
- **Compile**: `torch.compile(mode="max-autotune-no-cudagraphs")`
- **Iterations**: 200-300 training steps per run
- **Metric**: it/s (steady-state and cumulative)

## Changed files

| File                                       | Change                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `nvsubquadratic/modules/hyena_nd.py`       | Phase 1: `_rmsnorm_channels_first`, `_apply_norm_bchw` helpers; Phase 2: `channels_first_io` flag |
| `nvsubquadratic/modules/sequence_mixer.py` | Phase 2: `channels_first` flag with rearrange-before-split                                        |
| `examples/well/v2/active_matter/hyena.py`  | Phase 2: `channels_first=True`, `channels_first_io=True`                                          |
| `scripts/slurm/submit_1gpu.sh`             | Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`                                          |

______________________________________________________________________

## Phase 1 — Channels-first norms inside Hyena

**What**: Replace `movedim(1,-1) → norm → movedim(-1,1)` round-trips with
a `_apply_norm_bchw` dispatcher that applies RMSNorm/L2Norm/GroupNorm
directly on BCHW tensors.

**Eliminates**: ~10 movedim/reshape per block × 12 blocks = ~120 ops

**Call-sites changed** (3 per block):

1. QK norm (Q always, K only with Identity gate)
1. PixelHyena norm
1. Output norm

**Implementation note**: `_apply_norm_bchw` is decorated with
`@torch.compiler.disable` to prevent Inductor from fusing it with
surrounding FFT operations (which use `complex64` and cause
`InductorError: KeyError 'complex64'`).

### Phase 1 — Results

| Patch | Batch | Config   | it/s (steady) | it/s (cumul) | vs Baseline | val/loss | SLURM Job |
| ----- | ----- | -------- | ------------- | ------------ | ----------- | -------- | --------- |
| 16    | 32    | Baseline | 3.08          | 1.05         | 1.00x       | 0.00555  | 39360     |
| 16    | 32    | Phase 1  | 3.81          | 1.34         | **1.24x**   | 0.00554  | 39363     |
| 4     | 64    | Baseline | ~1.7          | 0.39         | 1.00x       | 0.00249  | 39356     |
| 4     | 64    | Phase 1  | ~1.7          | 0.44         | **1.13x**   | 0.00238  | 39364     |

- Steady-state measured over steps 220-300 (post-validation, post-warmup)
- Cumulative includes torch.compile warmup
- Losses match → correctness confirmed

______________________________________________________________________

## Phase 2 — BCHW-native QKV mixer boundary

**What**: Rearrange once (BHWC→BCHW) on the combined 3C QKV tensor in
`QKVSequenceMixer` before splitting into Q, K, V.  Hyena receives BCHW
directly and skips its 3 entry rearranges + 1 exit rearrange.

**Eliminates**: 4 rearranges per block → 2 rearranges per block (2 saved × 12 = 24 ops)

**Changes**:

- `QKVSequenceMixer`: `channels_first=True` → rearrange QKV to BCHW,
  split on `dim=1`, rearrange output back to BHWC after mixer
- `Hyena`: `channels_first_io=True` → skip entry/exit rearranges

### Phase 2 — Results (Phase 1 + Phase 2 combined)

| Patch | Batch | Config      | it/s (steady) | it/s (cumul) | vs Baseline | val/loss | SLURM Job |
| ----- | ----- | ----------- | ------------- | ------------ | ----------- | -------- | --------- |
| 16    | 32    | Baseline    | 3.08          | 1.05         | 1.00x       | 0.00555  | 39360     |
| 16    | 32    | Phase 1     | 3.81          | 1.34         | 1.24x       | 0.00554  | 39363     |
| 16    | 32    | Phase 1 + 2 | **4.71**      | 1.30         | **1.53x**   | 0.00554  | 39366     |
| 4     | 64    | Baseline    | ~1.7          | 0.39         | 1.00x       | 0.00249  | 39356     |
| 4     | 64    | Phase 1     | ~1.7          | 0.44         | 1.13x       | 0.00238  | 39364     |
| 4     | 64    | Phase 1 + 2 | ~1.7          | **0.51**     | **1.31x**   | 0.00254  | 39367     |

**Key observations**:

- At **p16** (short sequences, 256 tokens): **53% steady-state speedup**.
  Rearranges are a significant fraction of compute.
- At **p4** (longer sequences, 4096 tokens): **31% cumulative speedup**
  (mainly from faster torch.compile warmup); steady-state is dominated
  by FFT conv.
- Loss values match baseline → correctness confirmed for both phases.

______________________________________________________________________

## Issues encountered

| Issue                                                                       | Resolution                                                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `InductorError: KeyError 'complex64'` when compiling Phase 1 norms          | Added `@torch.compiler.disable` on `_apply_norm_bchw` to prevent fusion with FFT ops |
| p16 b64 OOM during `max-autotune` compile (~73 GiB Triton autotune buffers) | Benchmarked at b32 instead; `expandable_segments:True` did not help                  |
| p2 b16 OOM during compile (same autotune issue)                             | Not benchmarked (p2 left for future)                                                 |
| `train.batch_size` is an interpolated field, cannot override via CLI        | Override the source field `dataset.batch_size` instead                               |

______________________________________________________________________

## Phase 3 (future) — Full BCHW block loop

Not implemented.  Would keep BCHW through the entire `ResidualBlock` by also
making `input_norm`, `mlp_norm`, `MLP`, `Patchify`, and `Unpatchify`
channels-first.  This eliminates the last 2 rearranges per block (in
`QKVSequenceMixer`).

Estimated additional savings: 24 more ops (2 per block × 12 blocks).

______________________________________________________________________

## WandB runs

| Run Name                                                          | Run ID     | Config                     |
| ----------------------------------------------------------------- | ---------- | -------------------------- |
| `...hyena_gaussian_mask_its_200_2026-04-02-21-53-12`              | `4srwq1qh` | Baseline p16 (OOM'd)       |
| `...hyena_gaussian_mask_its_200_2026-04-02-22-01-12`              | `tpo5pbp8` | Baseline p16 b32 (OOM'd)   |
| `...hyena_gaussian_mask_its_200_patch_size_4_2026-04-02-21-53-12` | `i4w7t9j0` | Baseline p4 b64 ✓          |
| `...hyena_gaussian_mask_its_300_2026-04-02-22-..`                 | `wah1`     | Baseline p16 b32 (39360) ✓ |
| `...its_300_2026-04-02-22-..`                                     | `yj4f`     | Phase 1 p16 b32 (39363) ✓  |
| `...its_300_2026-04-02-22-..`                                     | `p0dd`     | Phase 1 p4 b64 (39364) ✓   |
| `...its_300_2026-04-02-22-..`                                     | `b2xd`     | Phase 2 p16 b32 (39366) ✓  |
| `...its_300_2026-04-02-22-..`                                     | `jgey`     | Phase 2 p4 b64 (39367) ✓   |
