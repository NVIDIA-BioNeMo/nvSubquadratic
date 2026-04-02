# FP16 FFT Convolution: Benchmark Results

Benchmark results for the dual-centered FP16 circular FFT convolution
(`use_fp16_fft=True`) compared to the default FP32 path.

All measurements on **Gray-Scott Reaction Diffusion** (Hyena, 2D, 128x128
spatial resolution, BF16-mixed precision training) on a single **H100 SXM
80 GB** GPU.

## End-to-End Training Throughput

Measured with the `IterationSpeedCallback` (windowed steady-state after
`torch.compile` warmup) and wall-clock tqdm parsing, 600 training steps.

### patch_size=1 (full-resolution 128x128 sequences)

| Configuration           | Steady it/s | fwd (ms) | bwd (ms)   | other (ms) | Peak GPU (MB) |
| ----------------------- | ----------- | -------- | ---------- | ---------- | ------------- |
| FP32 fftconv, bs=16     | 3.45        | 98.2     | 185.9      | 6.0        | 60,565        |
| **FP16 fftconv, bs=16** | **3.58**    | 108.8    | 163.7      | 6.7        | **58,049**    |
| Delta                   | **+3.8%**   | +10.8%   | **-12.0%** | —          | **-4.2%**     |

At full resolution, the FFT convolution is a small fraction of total
compute. The centering overhead slows down the forward pass (+10ms), but
smaller `complex32` intermediates speed up the backward pass (-22ms).
Net: a modest 3.8% throughput improvement and 2.5 GB memory savings —
not enough to increase batch size (bs=24 OOMs at 78.6/79.3 GB).

### patch_size=2 (downsampled 64x64 sequences)

| Configuration       | Steady it/s | fwd (ms) | bwd (ms) | other (ms) | Peak GPU (MB) |
| ------------------- | ----------- | -------- | -------- | ---------- | ------------- |
| FP32 fftconv, bs=16 | 11.85       | 28.8     | 50.4     | 5.2        | 15,293        |
| FP16 fftconv, bs=16 | 11.24       | 36.6     | 47.2     | 5.2        | 14,651        |
| Delta               | **-5.1%**   | +27.1%   | -6.3%    | —          | -4.2%         |

At lower resolution the FFTs are already fast (~29ms fwd), so the
centering overhead dominates: the forward pass increases by 8ms (+27%)
while the backward saves only 3ms. **FP16 fftconv is not beneficial
at patch_size=2.**

### Varying batch size at patch_size=2

| Configuration | Steady it/s | fwd (ms) | bwd (ms) | Peak GPU (MB) |
| ------------- | ----------- | -------- | -------- | ------------- |
| ps=2, bs=16   | 11.85       | 28.8     | 50.4     | 15,293        |
| ps=2, bs=32   | 6.56        | 51.5     | 95.1     | 30,225        |
| ps=2, bs=64   | 3.54        | 96.7     | 180.1    | 60,113        |

Memory and compute scale linearly with batch size.  At ps=2 the GPU is
under-utilized at bs=16 (15 GB / 80 GB), leaving room for much larger
batches.

## Data Loading: Not the Bottleneck

| Configuration      | Steady it/s | other (ms) | Peak GPU (MB) |
| ------------------ | ----------- | ---------- | ------------- |
| nw=12, no preload  | 3.44        | 6.3        | 60,565        |
| nw=12, RAM preload | 3.46        | 5.4        | 60,565        |
| nw=4, no preload   | 3.46        | 5.5        | 60,564        |
| nw=0, RAM preload  | 3.37        | 12.5       | 60,565        |
| nw=0, no preload   | 3.06        | 43.0       | 60,565        |

With `num_workers >= 4`, data loading (`other_ms`) is 5-6ms regardless
of RAM preloading. The GPU compute (~284ms/step) is the sole bottleneck.
RAM preloading (126 GB for Gray-Scott) provides no steady-state benefit.

## Correctness

The dual-centered FP16 implementation was validated against the FP32
reference on a trained Euler Hyena checkpoint (177M parameters):

- **Relative error** (vs FP32 reference): \< 1e-3 mean absolute error
- **Validation loss**: identical to 4 significant figures
- **No NaNs or Infs** across all tested configurations (1D, 2D, 3D)

See `tests/test_circular_fftconv_fp16.py` for the automated test suite.

## Summary

| Scenario                       | FP16 fftconv recommendation                          |
| ------------------------------ | ---------------------------------------------------- |
| ps=1 (full res), memory-tight  | Marginal: +3.8% speed, -4.2% memory                  |
| ps=1, need to fit larger batch | Not enough savings to change batch size              |
| ps=2 (lower res)               | **Do not use** — centering overhead hurts throughput |
| Correctness-critical           | Safe — validated against FP32 reference              |

The primary value of the FP16 fftconv work is **fixing the NaN bug** in
the original implementation (which was unusable). The centering technique
is mathematically sound and numerically stable, but the practical
throughput gains are modest because FFT convolutions are a small fraction
of total model compute.

## Environment

- GPU: NVIDIA H100 SXM 80 GB
- PyTorch 2.6.0+cu129, CUDA 12.9
- Precision: BF16-mixed (`bf16-mixed` Lightning trainer)
- `torch.compile` enabled (default mode)
- Dataset: Gray-Scott Reaction Diffusion (The Well)
- Model: Hyena with Gaussian mask, 12 blocks
