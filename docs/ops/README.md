# `nvsubquadratic.ops`: FFT convolution primitives

This folder contains the lowest-level building blocks of the library: FFT-based convolution operators that turn an `O(N · K)` spatial convolution into an `O(N log N)` frequency-domain product. Every subquadratic mixer in the library (Hyena, CKConv, multi-head variants) is built on them. They are kept here as plain functions, with no `nn.Module` state and no learned parameters, so that higher-level modules can compose them freely.

If you are reading the paper alongside this codebase, this is the file to start with.

______________________________________________________________________

## Why FFT convolution?

A standard spatial convolution between an input `x` of length `N` and a kernel `k` of length `K`,

$$
y[n] = \sum_{m} x[n - m] \cdot k[m]
$$

costs `O(N · K)` per channel. When `K` is small (e.g. a 3×3 image kernel) that is fine. The trouble starts when `K` is **comparable to `N`**, the regime Hyena-style models live in, where each layer's effective receptive field can span the whole input. There the spatial cost grows quadratically with sequence length.

The **convolution theorem** lets us replace the spatial convolution with an element-wise product in the frequency domain:

$$
y = \mathcal{F}^{-1}\!\bigl( \mathcal{F}(x) \odot \mathcal{F}(k) \bigr)
$$

The two FFTs and the inverse each cost `O(N log N)`, the element-wise product is `O(N)`, and the total cost is **independent of kernel size**. That is what makes "global-kernel" convolutional sequence models subquadratic.

Two flavours show up throughout the folder:

| Flavour                            | What it computes                                                                         | When to use                                                                                                                               |
| ---------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Linear** (`fftconv*`)            | Standard convolution, zero-padded so no wrap-around occurs, then cropped to "same" size. | Default choice; matches `torch.nn.ConvNd` semantics.                                                                                      |
| **Circular** (`circular_fftconv*`) | Periodic convolution where the kernel wraps around the input boundary.                   | When you want global mixing under periodic boundary conditions, or when input and kernel are the same size (no padding needed → cheaper). |

______________________________________________________________________

## File map

| File                                               | Precision | Conv type     | Channel mixing | When you'd reach for it                                                                                                                                                                                                                          |
| -------------------------------------------------- | --------- | ------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| {ref}`fftconv.py <ops-fftconv-fp32>`               | fp32      | linear        | depthwise      | The default. 1D/2D/3D, causal & non-causal.                                                                                                                                                                                                      |
| {ref}`circular_fftconv.py <ops-circular-fftconv>`  | fp32      | circular      | depthwise      | Periodic boundaries (e.g. PDEs, ARC grids), or when `K = N` so padding is wasteful.                                                                                                                                                              |
| {ref}`mixed_fftconv.py <ops-mixed-fftconv>`        | fp32      | per-axis BC   | depthwise      | **Mixed boundaries**: periodic on some spatial axes, zero-padded on others (e.g. Well's `rayleigh_benard`, `viscoelastic_instability`, `turbulent_radiative_layer`). Routes to the existing linear/circular ops in the all-False/all-True cases. |
| {ref}`fftconv_chunked.py <ops-chunking>`           | fp32      | linear        | depthwise      | Memory-constrained training; processes channels in chunks. Has a global flag so models can opt in transparently.                                                                                                                                 |
| {ref}`fftconv_custom.py <ops-fftconv-custom>`      | fp32 / native | linear    | depthwise      | Wraps optional fused CUDA kernels (`subquadratic_ops_torch.fft_conv2d` for 2D non-causal, `fft_causal_conv1d` for 1D causal) behind the same API as `fftconv.py`. Also hosts the `fused_fftconv2d_*` wrappers around `fused_fft_conv2d`, which run natively in fp32/fp16/bf16 but cap spatial dims at 64 per axis. |
| {ref}`fftconv_lowering.py <ops-fftconv-lowering>`  | native    | linear        | depthwise      | `torch.compile` pre-grad pass that rewrites `fftconv.py`'s 2D chain onto `fused_fft_conv2d`. Use when you want the fused kernel without changing `fft_backend` on an existing model.                                                             |
| {ref}`causal_conv1d_custom.py <ops-causal-conv1d>` | fp32      | direct causal | depthwise      | Non-FFT 1D causal kernels (`causal_conv1d` short conv, `b2b_causal_conv1d` fused proj-gate-mixer-gate). Use for kernels short enough that FFT overhead dominates, or as a fused-Hyena building block.                                            |

The [FP16 circular FFT convolution report](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/reports/fp16_fft_convolution/REPORT.md) contains the full derivation of the numerically stable fp16 circular conv (dual mean-centering + inclusion-exclusion geometric correction). Read it if you are touching the fp16 path or want to understand the math behind those `T1, T2, T3, T4` terms in the code.

______________________________________________________________________

## Naming convention

Every function name encodes its contract:

```
[causal_] fftconv {1d|2d|3d} _ {fp32|fp16} _ {bhl|blh} [_w_reshape] [_chunked]
```

| Part               | Meaning                                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `causal_`          | Output at position `n` only sees inputs at positions `≤ n`. 1D only.                                                                           |
| `1d` / `2d` / `3d` | Spatial rank.                                                                                                                                  |
| `fp32` / `fp16`    | Internal compute precision. The output dtype always matches `x.dtype` regardless.                                                              |
| `bhl` / `blh`      | Memory layout. `bhl` = channels-first (`[B, H, *spatial]`). `blh` = channels-last (`[B, *spatial, H]`).                                        |
| `_w_reshape`       | Wrapper that accepts BLH input, internally reshapes to BHL (faster), and reshapes back. The recommended entry point for channels-last callers. |
| `_chunked`         | Processes channels in groups to reduce peak GPU memory.                                                                                        |

So `causal_fftconv1d_fp32_bhl_w_reshape` is: causal 1D FFT conv, fp32 internal, accepts channels-last input, internally uses the channels-first kernel.

The CUDA-accelerated wrappers in `fftconv_custom.py` drop the `_fp32_` / `_fp16_` token because the underlying kernel manages its own precision internally, so the same name in `fftconv_custom` is `causal_fftconv1d_bhl_w_reshape`. The direct-conv wrappers in `causal_conv1d_custom.py` (`causal_conv1d`, `b2b_causal_conv1d`) do not follow this scheme because they are thin pass-throughs to the upstream API; see their docstrings for shapes.

______________________________________________________________________

## Shape conventions

Everything in this folder follows two layouts. Pick whichever matches your surrounding module:

- **BHL** (channels-first): `x: [B, H, *spatial]`, `kernel: [1|B, H, *K_dims]`. Standard for `torch.nn.ConvNd`-style modules. Faster under the hood because FFT runs on contiguous spatial axes without a transpose.
- **BLH** (channels-last): `x: [B, *spatial, H]`, `kernel: [1|B, *K_dims, H]`. Common in transformer-style code. Use the `_w_reshape` variants.

The kernel's leading dim is either `1` (shared kernel across the batch, the standard depthwise case) or `B` (per-sample kernel, e.g. FiLM-conditioned Hyena where each sample gets its own kernel).

### The shortcut term

Every operator accepts an optional `shortcut: [H]` tensor and computes

$$
y \leftarrow y + \mathrm{shortcut} \odot x
$$

i.e. a per-channel residual scale. This is *not* a generic skip connection; it fuses a specific algebraic shortcut that shows up repeatedly in Hyena-style gating, saving a separate kernel launch. Pass `None` if you don't need it.

______________________________________________________________________

## Choosing a function: a decision tree

1. **Do I need periodic boundaries?**

   - Yes → `circular_fftconv*`. The kernel wraps around the input; useful for PDE-like signals or whenever the input is naturally periodic.
   - No → `fftconv*`. The default.

1. **Is my model causal (1D sequence)?**

   - Yes → use the `causal_*` variant. Slightly more padding (`L + K` instead of `L + K/2`), but enforces no information leak from the future.
   - No → use the non-causal variant. Cheaper, since you only pad by `K/2`.

1. **What's my hidden layout?**

   - Channels-first (`[B, H, …]`) → use `_bhl` directly.
   - Channels-last (`[B, …, H]`) → use `_bhl_w_reshape`. Benchmarks show this is faster than a true `_blh` op because the FFT runs on contiguous spatial axes.

1. **What's my precision budget?**

   - fp32 is the default, always correct.
   - For aggressive memory/throughput savings on **power-of-2 spatial dims**, use the `*_fp16_*` variant. The fp16 ops use `norm="ortho"` and (for circular) dual mean-centering to stay within fp16 dynamic range; see the [FP16 circular FFT convolution report](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/reports/fp16_fft_convolution/REPORT.md).
   - If your spatial dims aren't powers of two, stay in fp32 (cuFFT requires power-of-2 for fp16 transforms).

1. **Am I OOMing?**

   - Try `fftconv_chunked`, which splits the channel dim into groups to cap peak memory. Default chunk size 128 gives ~26% memory savings for ~11% overhead.
   - Or combine: `fftconv_fp16.py` already provides `_chunked` variants that stack both savings.

1. **Is there a fused CUDA kernel for my shape?**

   - 2D non-causal or 1D causal long-conv → `fftconv_custom.py` exposes the upstream fused FFT kernels (`fft_conv2d`, `fft_causal_conv1d`) through the same API. Wire in via the `fft_backend="subq_ops"` flag on `CKConvND`. The 1D path requires `data_dim=1, is_causal=True`; the 2D path requires `data_dim=2, is_causal=False`.
   - 2D non-causal with spatial dims **≤ 64 per axis** → prefer `fft_backend="subq_ops_fused"`, which uses the newer `fused_fft_conv2d` kernel. It runs the whole rfft2/multiply/irfft2 pipeline in one launch **natively in fp32/fp16/bf16** instead of upcasting to fp32 — roughly 3–4× faster than `torch_fft` end-to-end in bf16 and ~1.2–2.4× faster than `subq_ops`. Requires `subquadratic-ops-torch >= 0.2.2`. The 64×64 cap comes from the kernel's largest FFT tile (128) combined with its `max(X, Y) ≤ fft_size // 2` requirement, and is enforced on the first forward pass rather than at construction. Beyond that, fall back to `subq_ops` or `torch_fft`.
   - Already have a model on `fft_backend="torch_fft"` and want the fused kernel without a config change → `fftconv_lowering.py` provides a `torch.compile` pre-grad pass that detects the FFT-conv chain and rewrites it. See [the lowering section below](#torchcompile-lowering).
   - 1D causal short conv (typical short_conv slot in a Hyena block) → `causal_conv1d_custom.py` exposes `causal_conv1d` directly, and `nvsubquadratic.modules.subq_ops_causal_conv1d.SubqOpsCausalConv1d` wraps it as a depthwise `nn.Conv1d`-compatible module.
   - 1D causal fused proj+gate+mixer+gate block → `b2b_causal_conv1d` in `causal_conv1d_custom.py`. Not yet wired into a `Hyena` variant; exposed as a building block.

______________________________________________________________________

## torch.compile lowering

Inductor cannot generate code for complex operators — it warns
`Torchinductor does not support code generation for complex operators` and falls
back to eager cuFFT for the whole `rfft2 → multiply → irfft2` chain. So
`torch.compile` on its own buys essentially nothing for `fftconv.py`.

`fftconv_lowering.py` closes that gap. It registers an inductor **pre-grad**
pass (before AOTAutograd, so autograd comes from the custom op's registered
backward) that detects the chain and swaps in the fused kernel:

```python
import torch
from nvsubquadratic.ops.fftconv_lowering import fused_fftconv2d_options, lowering_stats

compiled = torch.compile(model, options=fused_fftconv2d_options())
out = compiled(x)

print(lowering_stats())  # {'rewritten': 1}
```

`fused_fftconv2d_options()` is scoped to that one compiled callable and leaves
global inductor config untouched. When a trainer or framework owns the
`torch.compile` call and you cannot pass `options`, the
`fused_fftconv2d_lowering()` context manager does the same thing by patching
global config for its duration:

```python
from nvsubquadratic.ops.fftconv_lowering import fused_fftconv2d_lowering

with fused_fftconv2d_lowering():
    out = trainer.fit(model)  # whatever compiles internally
```

The pass only fires when the graph is *exactly* `fftconv.py`'s recipe — the
`min(N + (K+1)//2, 2N)` padding, the `K // 2` crop, shapes within the fused
kernel's limits, CUDA device, and a compute capability that supports the
required FFT tile (the 128 tile needs SM90+; SM80/SM86 lack the shared memory).
Anything else is left on the eager path.

Two things to know:

- **Setting `fft_backend="subq_ops_fused"` is the more predictable route** and
  needs no pass at all. Reach for the lowering when you want the kernel under an
  existing `torch_fft` model without touching its config.
- **`lowering_stats()` only counts passes that actually ran.** Inductor caches
  compiled artifacts on disk, and a cache hit skips every pre-grad pass, so
  empty counters can mean "served from cache" rather than "did not fire". Set
  `torch._inductor.config.force_disable_caches = True` when verifying.

By default the pass also rewrites fp16/bf16 graphs, which moves the convolution
from fp32-internal to native-dtype — that is where most of the speedup comes
from, and it is a real numerics change (~2e-3 normwise in bf16). Pass
`fused_fftconv2d_lowering(allow_reduced_precision=False)` to restrict it to
fp32, where the rewrite is numerically neutral.

______________________________________________________________________

## Numerical notes

- All operators **accept any input dtype** but cast to the internal compute precision (fp32 or fp16) before the FFT. The output is returned in the **original dtype of `x`**, so no manual cast is needed on the caller side. The one exception is the `fused_fftconv2d_*` family, which runs in `x.dtype` end to end — that is the point of it. Measured against the fp32 reference, its normwise relative error is ~3e-7 in fp32, ~3e-4 in fp16, and ~3e-3 in bf16.
- The fp32 ops are correct for any input range. The fp16 ops impose two constraints: spatial dims must be powers of two (cuFFT), and the dynamic range is handled by mean-centering both `x` and `k` (see derivation doc).
- The non-causal linear ops match a standard `torch.nn.ConvNd(padding='same')` up to floating-point rounding. The circular ops match `torch.nn.functional.conv*d` after a circular pad. Both are exercised in `tests/`.

______________________________________________________________________

## Related modules

- **[`nvsubquadratic/modules/kernels_nd.py`](../../nvsubquadratic/modules/kernels_nd.py)**: learned kernel parametrisations that produce the kernels these ops consume.
- **[`nvsubquadratic/modules/hyena_nd.py`](../../nvsubquadratic/modules/hyena_nd.py)**: the Hyena operator, the main consumer of these ops.
- **[`nvsubquadratic/modules/ckconv_nd.py`](../../nvsubquadratic/modules/ckconv_nd.py)** / **[`ckconv_multihead_nd.py`](../../nvsubquadratic/modules/ckconv_multihead_nd.py)**: CKConv variants that compose these primitives.

```{toctree}
---
maxdepth: 1
caption: Further reading
---
mixed_boundary_conditions
```
