# ViT-5-Small Throughput Benchmarks

Single-GPU (H100 SXM 80GB) throughput benchmarks for ViT-5-Small (22M params, 224x224 input, batch size 256, BF16).

## Results Summary

| Configuration | Time/step (ms) | Throughput (samples/sec) | MFU |
|---|---:|---:|---:|
| **Before optimizations** | | | |
| Eager (original) | 159.2 | 1,608 | 4.6% |
| torch.compile (default) | 46.0 | 5,560 | 15.9% |
| torch.compile (max-autotune) | CRASH | — | — |
| **After optimizations** | | | |
| Eager (optimized) | 111.1 | 2,305 | 6.6% |
| torch.compile (default) | 33.2 | 7,716 | 22.0% |
| torch.compile (max-autotune) | 32.0 | 8,003 | 22.9% |

**Theoretical maximum**: ~34,800 samples/sec (100% MFU on H100 SXM @ 989 TFLOPS BF16).

## What Changed

### Model optimizations (`vit5_attention.py`)

1. **RoPE precomputation** — Replaced per-forward dict-based RoPE cache with `register_buffer` for precomputed cos/sin. Eliminates graph breaks, enables CUDA Graphs, and removes redundant `rearrange`/`torch.cat` ops per step.

2. **SDPA backend auto-selection** — Removed explicit `SDPBackend.FLASH_ATTENTION` preference. PyTorch now auto-selects the fastest backend (CuDNN on H100).

3. **Removed redundant dtype casts** — Eliminated `.to(torch.bfloat16)` / `.to(in_dtype)` around SDPA calls. Autocast handles precision.

4. **QuACK fused RMSNorm** — `quack.rmsnorm` replaces the manual float32-upcast-then-downcast RMSNorm with a single fused Triton kernel. Falls back to PyTorch on CPU or when QuACK is unavailable.

### Optimizer

5. **Apex FusedLAMB** — Multi-tensor fused LAMB optimizer replaces `torch_optimizer.Lamb`. Batches all parameter updates into 1-2 kernel launches.

### Compilation

6. **`torch.compile` support** — Added `compile` and `compile_mode` config flags. The RoPE buffer refactoring (item 1) was required to unblock `max-autotune` mode, which previously crashed with CUDA Graph errors.

## FLOP Analysis

ViT-5-Small per-sample FLOPs (12 blocks, dim 384, 6 heads, 201 tokens):

| Component | GFLOPs (fwd) |
|---|---:|
| Patch embed | 0.06 |
| Attention (QKV + proj) | 5.72 |
| Attention (softmax) | 0.12 |
| MLP | 5.72 |
| Head | 0.001 |
| **Total (fwd)** | **11.6** |
| **Total (train: fwd + 2x bwd)** | **34.9** |

## Benchmark Scripts

| Script | Purpose |
|---|---|
| `bench_vit5_baseline.py` | Original model eager throughput + FLOPs/MFU calculation |
| `bench_vit5_compile.py` | Eager vs torch.compile (default) vs max-autotune + profiler |
| `bench_vit5_profile.py` | Single-step CUDA kernel profiling (top-30 by time) |
| `bench_vit5_optimized.py` | Correctness checks + throughput for the optimized model |

### Running

Submit via SLURM (from the repo root):

```bash
sbatch benchmarks/scripts/bench_optimized.sh
sbatch benchmarks/scripts/bench_compile.sh
sbatch benchmarks/scripts/bench_profile.sh
```

Logs go to `logs/`.

## Environment

- GPU: NVIDIA H100 SXM 80GB
- PyTorch 2.10.0+cu129
- CUDA 12.9
- Apex (FusedLAMB)
- QuACK (fused RMSNorm)
