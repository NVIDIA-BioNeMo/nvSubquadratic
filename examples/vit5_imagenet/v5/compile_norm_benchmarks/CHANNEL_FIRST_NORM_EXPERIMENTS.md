# Channel-First Norm Experiments — ViT-5 Hyena ImageNet

## Motivation

Inside the Hyena mixer, tensors are in channel-first layout `[B, C, H, W]`, but
the normalization layers (`RMSNorm`, `L2Norm`) expect channel-last `[B, T, C]`.
This forces `movedim`/`reshape` round-trips at every norm call site.  We
investigated whether eliminating these transposes via channel-first norms could
improve throughput.

## What we changed

1. **`RMSNormChannelFirst` module** (`nvsubquadratic/modules/rms_norm_channel_first.py`):
   A drop-in norm that normalizes over `dim=1`.  Has two backends:

   - **QuACK CuTe kernel** (`quack.rmsnorm_channel_first`) — fused CUDA kernel
     for Hopper/Blackwell GPUs.  Opaque to `torch.compile`.
   - **Pure PyTorch fallback** — `x.pow(2).mean(1)` + `rsqrt`.  Fully visible
     to `torch.compile`, enabling fusion with adjacent ops.
     Controlled via `use_quack=True|False`.

1. **`is_channels_first_norm` duck-typing** (`nvsubquadratic/modules/_channels_first_utils.py`):
   Detects channel-first norms (via a `channels_first` attribute) so `hyena_nd.py`
   can skip the `movedim`/`reshape` when the norm already expects `[B, C, H, W]`.

1. **`L2Norm(dim=1)`** (`nvsubquadratic/utils/qk_norm.py`):
   Added a `channels_first` property that returns `True` when `self.dim == 1`,
   so QK-norm can also operate in channel-first layout without transposes.

1. **`hyena_nd.py` modifications**:
   Three call sites (`qk_norm`, `pixelhyena_norm`, `output_norm`) now check
   `is_channels_first_norm(module)` and conditionally skip the `movedim`/`reshape`.

## Experiment matrix

All runs: ViT-5-Small Hyena GAP, 8×H100, ImageNet-1K, `torch.compile`
with `max-autotune-no-cudagraphs`.  Speed measured after stabilization
(epochs 4–8+).

| #     | FFT backend    | Norm backend                          | QuACK norm? | it/s      | vs baseline  |
| ----- | -------------- | ------------------------------------- | ----------- | --------- | ------------ |
| 1     | `subq_ops`     | channel-last `RMSNorm`                | N/A         | **~6.05** | — (baseline) |
| 2     | `subq_ops`     | channel-first `RMSNormChannelFirst`   | Yes         | ~5.82     | −3.8%        |
| 3     | `torch_fft`    | channel-first `RMSNormChannelFirst`   | Yes         | ~5.23     | −13.6%       |
| 4     | `torch_fft`    | channel-first (PyTorch, no QuACK)     | No          | ~5.62     | −7.1%        |
| **5** | **`subq_ops`** | **channel-first (PyTorch, no QuACK)** | **No**      | **~6.40** | **+5.8%**    |

## Key findings

### 1. The QuACK channel-first kernel hurts under `torch.compile`

The QuACK CuTe kernel is a `custom_op` — opaque to the compiler.  It acts as a
**fusion barrier**, preventing `torch.compile` from merging the norm into
adjacent pointwise / elementwise operations.  The kernel itself may also have
less optimal memory access patterns for channel-first layout (strided access
across the channel dimension).

### 2. `torch.compile` fuses away transposes for free

In the baseline (run 1), the `movedim` + `reshape` operations around the
channel-last `RMSNorm` look expensive in eager mode, but `torch.compile`
eliminates them by fusing them into the surrounding memory access patterns.
This is why naively adding a channel-first kernel (run 2) doesn't help — the
"problem" it solves doesn't exist under compilation.

### 3. The `subq_ops` FFT kernel is hard to beat

Replacing `subq_ops` with `torch_fft` (runs 3, 4) always hurts, even when
`torch.compile` has full end-to-end visibility.  The hand-tuned CUDA FFT
convolution kernel is simply faster than what Inductor/Triton can generate.

### 4. Best config: `subq_ops` + PyTorch channel-first norms

Run 5 is the winner: keep the fast `subq_ops` FFT kernel, but use **pure
PyTorch** channel-first norms (`use_quack=False`).  This lets `torch.compile`
fuse the norm ops with the surrounding gating / elementwise operations while
still benefiting from the fast custom FFT kernel.  The result is a **~5.8%
speedup** over baseline.

## Optimal config

```
hyena_gap_pretrain_cf_norm.py  (with these settings):
  fft_backend = "subq_ops"
  RMSNormChannelFirst(use_quack=False)  — for pixelhyena_norm, output_norm
  L2Norm(dim=1)                         — for qk_norm
```

## Takeaway

When using `torch.compile`, **avoid opaque custom kernels for small ops** that
the compiler can fuse.  Reserve custom CUDA kernels for large, compute-bound
operations (like FFT convolutions) where hand-tuning genuinely outperforms the
compiler.  For lightweight ops like normalization, letting the compiler see
through them enables better end-to-end fusion.
