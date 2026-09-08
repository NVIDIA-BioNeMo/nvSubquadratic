# nvSubquadratic forward-time benchmarks — results digest

Single-layer forward-pass timings for HyenaND vs attention kernels vs Mamba2,
swept from a 16-wide grid up to ~16.7M tokens in 1D, 2D and 3D.

This file is generated from the JSONL sweep outputs in this directory; every number
below is read from that data rather than transcribed. Raw per-point records live in
`forward_time_{1,2,3}d.jsonl` and `forward_time_flash_{1,2,3}d.jsonl`, with matching
`.png`/`.pdf` plots and `*-<jobid>.out` run logs.

---

## What was run

Two sweep configurations, each across 1D/2D/3D — six jobs total. They answer
different questions and are **not** comparable to each other:

| Config | hidden_dim | head_dim | Mixers | Reaches | Purpose |
|---|---|---|---|---|---|
| **Reach** | 8 | 4 | hyena, attention, mamba | 16.7M tokens | How far each operator scales. Small width keeps the qkv tensor under torch's 2^31 index limit at 16.7M. |
| **Flash-kernel** | 512 | 128 | hyena, attention, flex, fa4, mamba | ~1M tokens | SDPA vs FlexAttention vs FlashAttention-4 vs HyenaND at the width flash kernels are optimised for. Walls at ~1.4M on the same index limit. |

`attention`, `flex` and `fa4` are three interchangeable kernels on one shared q/k/v +
RoPE path. `flex` and `fa4` require head_dim >= 16, so they cannot run in the Reach
config and appear only in the Flash-kernel sweeps.

---

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA GB200 (aarch64), driver 580.173.02 |
| torch | 2.12.1+cu130 (CUDA 13.0) |
| subquadratic-ops-torch-cu13 | 0.2.2 |
| mamba-ssm / causal-conv1d | 2.3.2.post1 / 1.6.2.post1 |
| flash-attn-4 | 4.0.0b23 (cutlass-dsl 4.6.0.dev0) |
| apex | 0.1 |
| Image | `/lustre/fsw/healthcareeng_bionemo/farhadr/enroot/nvsubquadratic-arm64.sqsh` |
| Branch | `farhadr/2d_bench` (merge of PR #137 + PR #138) |

---

## Results

### Reach 1D  (`forward_time_1d`)

- SLURM job: `2559702`  |  device: `NVIDIA GB200`  |  dtype: `bf16`
- hidden_dim=8, num_heads=2, batch_size=1, data_dim=1 (L = R^1)
- Hyena short conv: subq_ops_causal_conv1d
- Hyena FFT backend: `subq_ops` at R=16..16777216

| R | L | hyena | attention | mamba |
|---|---|---|---|---|
| 16 | 16 | 1.667 | 0.652 | 2.237 |
| 32 | 32 | 1.508 | 1.144 | 2.410 |
| 64 | 64 | 1.515 | 1.143 | 2.408 |
| 128 | 128 | 1.470 | 1.108 | 2.411 |
| 256 | 256 | 1.431 | 1.149 | 2.412 |
| 512 | 512 | 1.412 | 1.134 | 2.272 |
| 1,024 | 1,024 | 1.453 | 1.178 | 2.391 |
| 2,048 | 2,048 | 1.450 | 0.643 | 2.415 |
| 4,096 | 4,096 | 1.452 | 1.188 | 2.279 |
| 8,192 | 8,192 | 1.453 | 1.139 | 2.425 |
| 16,384 | 16,384 | 1.405 | 1.092 | 2.445 |
| 32,768 | 32,768 | 1.420 | 1.395 | 2.439 |
| 65,536 | 65,536 | 1.497 | 5.024 | 2.437 |
| 131,072 | 131,072 | 1.501 | 17.164 | 2.446 |
| 262,144 | 262,144 | 1.480 | 67.812 | 3.067 |
| 524,288 | 524,288 | 1.921 | 261.968 | 5.956 |
| 1,048,576 | 1,048,576 | 4.348 | 1046.639 | 11.772 |
| 2,097,152 | 2,097,152 | 8.865 | 4185.474 | 23.352 |
| 4,194,304 | 4,194,304 | 18.639 | 16788.312 | error |
| 8,388,608 | 8,388,608 | 37.347 | 67089.927 | error |
| 16,777,216 | 16,777,216 | 74.507 | 268037.938 | error |

All values are ms per forward pass, lower is better.

- max attention/hyena: **3597x** at L=16,777,216
- max mamba/hyena: **3.1x** at L=524,288

Points without a timing:
- `mamba` error at R=[4194304, 8388608, 16777216]

---

### Reach 2D  (`forward_time_2d`)

- SLURM job: `2559620`  |  device: `NVIDIA GB200`  |  dtype: `bf16`
- hidden_dim=8, num_heads=2, batch_size=1, data_dim=2 (L = R^2)
- Hyena FFT backend: `subq_ops` at R=128..4096; `subq_ops_fused` at R=16..64

| R | L | hyena | attention | mamba |
|---|---|---|---|---|
| 16 | 256 | 1.299 | 0.803 | 2.289 |
| 32 | 1,024 | 1.113 | 1.407 | 2.154 |
| 64 | 4,096 | 1.093 | 1.483 | 2.177 |
| 128 | 16,384 | 1.143 | 1.443 | 2.332 |
| 256 | 65,536 | 1.046 | 5.096 | 2.338 |
| 512 | 262,144 | 1.055 | 67.892 | 3.198 |
| 1,024 | 1,048,576 | 2.361 | 1046.963 | 12.224 |
| 2,048 | 4,194,304 | 10.156 | 16787.928 | error |
| 4,096 | 16,777,216 | 41.292 | 268220.812 | error |

All values are ms per forward pass, lower is better.

- max attention/hyena: **6496x** at L=16,777,216
- max mamba/hyena: **5.2x** at L=1,048,576

Points without a timing:
- `mamba` error at R=[2048, 4096]

---

### Reach 3D  (`forward_time_3d`)

- SLURM job: `2559621`  |  device: `NVIDIA GB200`  |  dtype: `bf16`
- hidden_dim=8, num_heads=2, batch_size=1, data_dim=3 (L = R^3)
- Hyena FFT backend: `torch_fft` at R=16..256

| R | L | hyena | attention | mamba |
|---|---|---|---|---|
| 16 | 4,096 | 1.118 | 0.579 | 2.401 |
| 32 | 32,768 | 1.115 | 1.334 | 2.414 |
| 64 | 262,144 | 1.122 | 67.674 | 3.142 |
| 128 | 2,097,152 | 7.579 | 4178.602 | 23.841 |
| 256 | 16,777,216 | 62.031 | 267676.531 | error |

All values are ms per forward pass, lower is better.

- max attention/hyena: **4315x** at L=16,777,216
- max mamba/hyena: **3.1x** at L=2,097,152

Points without a timing:
- `mamba` error at R=[256]

---

### Flash-kernel 1D  (`forward_time_flash_1d`)

- SLURM job: `2559703`  |  device: `NVIDIA GB200`  |  dtype: `bf16`
- hidden_dim=512, num_heads=4, batch_size=1, data_dim=1 (L = R^1)
- Hyena short conv: subq_ops_causal_conv1d
- Hyena FFT backend: `subq_ops` at R=16..1048576

| R | L | hyena | attention | flex | fa4 | mamba |
|---|---|---|---|---|---|---|
| 16 | 16 | 1.956 | 0.551 | 0.997 | 0.825 | 2.286 |
| 32 | 32 | 1.428 | 0.558 | 0.893 | 0.950 | 2.291 |
| 64 | 64 | 1.366 | 0.562 | 1.069 | 0.949 | 2.299 |
| 128 | 128 | 1.332 | 0.548 | 0.981 | 0.955 | 2.321 |
| 256 | 256 | 1.328 | 0.617 | 1.024 | 0.759 | 2.297 |
| 512 | 512 | 1.322 | 0.617 | 1.049 | 0.782 | 2.187 |
| 1,024 | 1,024 | 1.326 | 0.614 | 1.043 | 0.938 | 2.331 |
| 2,048 | 2,048 | 1.289 | 0.611 | 1.036 | 0.961 | 2.358 |
| 4,096 | 4,096 | 1.286 | 0.632 | 1.046 | 0.960 | 2.190 |
| 8,192 | 8,192 | 1.292 | 0.631 | 1.056 | 0.959 | 2.297 |
| 16,384 | 16,384 | 1.290 | 0.822 | 2.425 | 0.996 | 2.322 |
| 32,768 | 32,768 | 1.286 | 2.228 | 7.597 | 2.658 | 2.310 |
| 65,536 | 65,536 | 2.395 | 6.965 | 28.295 | 8.120 | 3.007 |
| 131,072 | 131,072 | 4.761 | 24.981 | 114.488 | 30.295 | 5.819 |
| 262,144 | 262,144 | 9.971 | 95.353 | 460.909 | 119.907 | 11.451 |
| 524,288 | 524,288 | 24.203 | 379.858 | 1831.237 | 505.537 | 22.792 |
| 1,048,576 | 1,048,576 | 76.215 | 1587.445 | 7299.468 | 2085.962 | - |

All values are ms per forward pass, lower is better.

- max attention/hyena: **21x** at L=1,048,576
- max flex/hyena: **96x** at L=1,048,576
- max fa4/hyena: **27x** at L=1,048,576
- max mamba/hyena: **1.8x** at L=2,048

---

### Flash-kernel 2D  (`forward_time_flash_2d`)

- SLURM job: `2559623`  |  device: `NVIDIA GB200`  |  dtype: `bf16`
- hidden_dim=512, num_heads=4, batch_size=1, data_dim=2 (L = R^2)
- Hyena FFT backend: `subq_ops` at R=128..1024; `subq_ops_fused` at R=16..64

| R | L | hyena | attention | flex | fa4 | mamba |
|---|---|---|---|---|---|---|
| 16 | 256 | 1.625 | 0.986 | 1.242 | 1.065 | 2.562 |
| 32 | 1,024 | 1.226 | 0.949 | 1.391 | 1.037 | 2.511 |
| 64 | 4,096 | 1.211 | 0.966 | 1.544 | 1.448 | 2.551 |
| 128 | 16,384 | 1.359 | 0.999 | 2.609 | 1.458 | 2.736 |
| 256 | 65,536 | 3.555 | 7.263 | 28.950 | 8.412 | 2.725 |
| 512 | 262,144 | 14.075 | 96.786 | 467.332 | 122.081 | 8.659 |
| 1,024 | 1,048,576 | 89.187 | 1648.564 | 7381.654 | 2080.850 | - |

All values are ms per forward pass, lower is better.

- max attention/hyena: **18x** at L=1,048,576
- max flex/hyena: **83x** at L=1,048,576
- max fa4/hyena: **23x** at L=1,048,576
- max mamba/hyena: **2.1x** at L=4,096

---

### Flash-kernel 3D  (`forward_time_flash_3d`)

- SLURM job: `2559624`  |  device: `NVIDIA GB200`  |  dtype: `bf16`
- hidden_dim=512, num_heads=4, batch_size=1, data_dim=3 (L = R^3)
- Hyena FFT backend: `torch_fft` at R=16..64

| R | L | hyena | attention | flex | fa4 | mamba |
|---|---|---|---|---|---|---|
| 16 | 4,096 | 1.136 | 0.496 | 0.730 | 0.708 | 2.707 |
| 32 | 32,768 | 7.897 | 1.764 | 7.329 | 2.150 | 2.929 |
| 64 | 262,144 | 70.852 | 96.806 | 459.914 | 118.767 | 8.324 |

All values are ms per forward pass, lower is better.

- max attention/hyena: **1.4x** at L=262,144
- max flex/hyena: **6.5x** at L=262,144
- max fa4/hyena: **1.7x** at L=262,144
- max mamba/hyena: **2.4x** at L=4,096

---

## How to read these numbers

### 1. Below ~65K tokens the sweep does not measure attention

In the Reach config (head_dim 4) attention sits flat at ~1.1 ms from R=32 to R=32768.
That floor is fixed per-iteration dispatch overhead, not attention: the quadratic term
is real but far below the floor until L=65536, where attention first departs (5.02 ms)
and then scales cleanly (17.2 -> 67.8 -> 262 -> 1047 ms, ~4x per doubling).

**Do not draw conclusions from points below L=65536.** For a figure, truncating the
x-axis there is the honest presentation.

### 2. There is a reproducible dip at R=2048 in the 1D Reach plot

Investigated directly. It is **not** hardware and **not** a one-off:

- Clean ECC (0 uncorrected, all 4 GPUs), no throttling of any kind, 27 C.
- Reproduces on a different node (`ptyche0178`) than the production run (`ptyche0077`),
  ruling out a node-specific fault.
- 6 repetitions at R=2048: min 0.585, median 0.649, max 1.189 ms. Five of six land at
  ~0.65 ms, so the production value of 0.643 is the *typical* reading, not an outlier.

Timings in this region are **bimodal**, landing near ~0.6 ms or ~1.2 ms, and which
resolutions land where changes with node and process context (in production only R=16
and R=2048 were fast; on the rerun node R=1024 and R=2048 were fast while R=4096 was
slow). The likely mechanism is SDPA backend selection: at head_dim 4 the flash path is
ineligible, so PyTorch chooses between math and mem-efficient kernels by heuristic,
giving two stable states. Re-running relocates the dip rather than removing it.

### 3. Mamba's `error` points are kernel limits, NOT out-of-memory

No point in any sweep ran out of memory. The `error` entries here are
`cudaErrorInvalidValue` raised by Dao-AILab's `causal_conv1d`: its channels-last path
tiles the sequence at 64 elements per block and hits the CUDA grid-dimension cap of
65,535, giving a hard ceiling of **4,194,240 tokens**. That is a launch-configuration
limit, so it occurs on any GPU regardless of memory (peak use here was under 5 GB).

At wider configurations a *different* and earlier limit binds first: 32-bit index
overflow in the Triton SSD scan (`ssd_chunk_state.py`), which fails at ~2^31 elements
and therefore moves inversely with model width (~600K tokens at hidden 768). These
sweeps use hidden_dim 8, which is why they reach 4M before failing.

Both limits are implementation properties of the shipping kernels, not architectural
bounds on Mamba-2, and both are documented with reproductions in
`docs/mamba2_limits.md`. The Mamba baseline intentionally keeps its own
`causal_conv1d`; it was not switched to the subq_ops kernel.

### 4. The fused 2D kernel covers only part of the 2D sweep

`fft_backend=subq_ops_fused` (subq_ops >= 0.2.2) is **2D-only and capped at 64 per
axis** — its largest FFT tile is 128 and it requires max(X,Y) <= fft_size/2. Across a
16..16M sweep it therefore covers 2D at R=16/32/64 only; R>=128 falls back to
`subq_ops`, and 3D to `torch_fft` (no 3D CUDA kernel exists). The backend actually used
is recorded per row in the JSONL as `backend`, so no point is misattributed.

### 5. 1D HyenaND is not comparable to runs before 2026-08-10

HyenaND's short conv now follows the operator's causality. Previously the causal 1D
config paired a causal long conv with a symmetric-padded short conv, so the operator
could see one token of future context. 1D now uses the fused
`subquadratic_ops_torch.causal_conv1d` (left-only padding), recorded per row as
`short_conv`. This is a behaviour change, not only a speedup. 2D/3D are unaffected:
they are non-causal and still use symmetric `torch.nn.ConvNd`, because the fused kernel
is causal and 1D-only.

### 6. The Flash-kernel 3D sweep is too short to show a crossover

At hidden_dim 512 in 3D the 2^31 index limit bites at R=64 (262K tokens), leaving only
three points. HyenaND is still *behind* attention at the first two and only pulls ahead
at the last (70.9 vs 96.8 ms, 1.4x). Treat that sweep as a kernel-availability check
rather than evidence about scaling; the 3D scaling story is in the Reach 3D sweep,
which reaches 16.7M.

### 7. Points in the tail are averaged over very few iterations

Each datapoint is a single measurement: CUDA events around one back-to-back loop of N
forwards, divided by N. It is **not** an average of repeated runs. N adapts to the cost
(target ~5 s per point, capped by `--num-iters` and `--max-seconds-per-point`), so the
largest points have very few iterations — the 16.7M attention point is N=1. Those are
still reliable because a 268-second measurement swamps jitter, but they are single-shot.

1D attention iteration counts:

| L | iterations |
|---|---|
| 16 | 30 |
| 32 | 30 |
| 64 | 30 |
| 128 | 30 |
| 256 | 30 |
| 512 | 30 |
| 1,024 | 30 |
| 2,048 | 30 |
| 4,096 | 30 |
| 8,192 | 30 |
| 16,384 | 30 |
| 32,768 | 30 |
| 65,536 | 30 |
| 131,072 | 30 |
| 262,144 | 30 |
| 524,288 | 19 |
| 1,048,576 | 5 |
| 2,097,152 | 3 |
| 4,194,304 | 3 |
| 8,388,608 | 3 |
| 16,777,216 | 1 |

---

## JSONL schema

One record per (mixer, resolution):

```json
{
  "mixer": "attention",
  "resolution": 16,
  "seq_len": 16,
  "data_dim": 1,
  "backend": null,
  "short_conv": null,
  "batch_size": 1,
  "hidden_dim": 8,
  "num_heads": 2,
  "dtype": "bf16",
  "device": "NVIDIA GB200",
  "status": "ok",
  "ms": 0.6519936243693034,
  "mem_gb": 0.03125905990600586,
  "iters": 30
}
```

| Field | Meaning |
|---|---|
| `status` | `ok`, `oom`, `timeout`, `error`, or `unavailable` (dependency missing). Only `ok` rows have `ms`. |
| `ms` | Milliseconds per forward pass. |
| `iters` | Iterations the timed loop averaged over (adaptive; see note 6). |
| `backend` | Hyena FFT-conv backend actually used at this point (null for non-hyena). |
| `short_conv` | Hyena short-conv implementation actually used (null for non-hyena). |
| `mem_gb` | Peak CUDA memory allocated during the point. |

## Reproducing

```bash
# Reach sweeps (16 -> 16.7M):
for D in 1 2 3; do DATA_DIM=$D sbatch --time=04:30:00 --export=ALL \
    scripts/slurm/submit_forward_time_nd.sh; done

# Flash-kernel sweeps:
for D in 1 2 3; do DATA_DIM=$D scripts/slurm/submit_forward_time_flash_kernels.sh; done
```

Both wrap `benchmarks/benchmark_forward_time_nd_resolution.py`. Relevant flags:
`--fft-backend {subq_ops,subq_ops_fused,torch_fft}`, `--short-conv {subq_ops,torch}`,
`--mixers`, `--resolutions`, `--hidden-dim`, `--num-iters`, `--max-seconds-per-point`.

