# ViT-5-Small Throughput Benchmarks

Single-GPU and multi-GPU (8x H100 SXM 80GB) throughput benchmarks for ViT-5-Small (22M params, 224x224 input, batch size 256, BF16).

## Single-GPU Model Throughput

Pure model throughput (synthetic data, no data loading overhead):

| Configuration                | Time/step (ms) | Throughput (samples/sec) |   MFU |
| ---------------------------- | -------------: | -----------------------: | ----: |
| **Before optimizations**     |                |                          |       |
| Eager (original)             |          159.2 |                    1,608 |  4.6% |
| torch.compile (default)      |           46.0 |                    5,560 | 15.9% |
| torch.compile (max-autotune) |          CRASH |                        — |     — |
| **After optimizations**      |                |                          |       |
| Eager (optimized)            |          111.1 |                    2,305 |  6.6% |
| torch.compile (default)      |           33.2 |                    7,716 | 22.0% |
| torch.compile (max-autotune) |           32.0 |                    8,003 | 22.9% |

**Theoretical maximum**: ~34,800 samples/sec (100% MFU on H100 SXM @ 989 TFLOPS BF16).

## Multi-GPU DDP Training Throughput (8x H100)

End-to-end training throughput including data loading, augmentations, compute, and DDP allreduce:

| Version             | Dataloader                     | Storage        |     it/s | ms/step |  Speedup |
| ------------------- | ------------------------------ | -------------- | -------: | ------: | -------: |
| CPU baseline        | torchvision                    | Network FS     |     ~2.5 |    ~400 |     1.0x |
| v1 (DALI)           | DALI                           | Network FS     |      5.3 |     189 |     2.1x |
| v2 (DALI optimized) | DALI + compiled aug            | Network FS     |      6.3 |     159 |     2.5x |
| **optimized_plus**  | **DALI + compiled aug**        | **Local NVMe** | **12.1** |  **83** | **4.8x** |
| **fused**           | **DALI (all aug in pipeline)** | **Local NVMe** | **12.6** |  **79** | **5.0x** |

### Step breakdown (fused DALI, NVMe, compiled)

| Component              | Time (ms) | % of step |
| ---------------------- | --------: | --------: |
| DALI fetch             |        ~2 |        3% |
| Mixup/CutMix + permute |      0.35 |      0.5% |
| Forward + Backward     |       ~66 |       94% |
| Optimizer (FusedLAMB)  |        ~2 |        3% |

After fusing all augmentations into the DALI pipeline and staging data on NVMe, the pipeline is **fully compute-bound**. Data loading accounts for \<3% of step time.

### Fused vs optimized DALI (DDP x8, profiling script)

| Pipeline       | Model     | Full step (ms) | Agg throughput (img/s) |  Speedup |
| -------------- | --------- | -------------: | ---------------------: | -------: |
| DALI-optimized | Small     |           80.6 |                 25,401 |        — |
| **DALI-fused** | **Small** |       **70.4** |             **29,103** | **+15%** |
| DALI-optimized | Base      |          131.1 |                 15,622 |        — |
| **DALI-fused** | **Base**  |      **120.8** |             **16,950** |  **+9%** |

## What Changed

### Model optimizations (`vit5_attention.py`)

1. **RoPE precomputation** — Replaced per-forward dict-based RoPE cache with `register_buffer` for precomputed cos/sin. Eliminates graph breaks, enables CUDA Graphs, and removes redundant `rearrange`/`torch.cat` ops per step.

1. **SDPA backend auto-selection** — Removed explicit `SDPBackend.FLASH_ATTENTION` preference. PyTorch now auto-selects the fastest backend (CuDNN on H100).

1. **Removed redundant dtype casts** — Eliminated `.to(torch.bfloat16)` / `.to(in_dtype)` around SDPA calls. Autocast handles precision.

1. **QuACK fused RMSNorm** — `quack.rmsnorm` replaces the manual float32-upcast-then-downcast RMSNorm with a single fused Triton kernel.

### Optimizer

5. **Apex FusedLAMB** — Multi-tensor fused LAMB optimizer replaces `torch_optimizer.Lamb`. Batches all parameter updates into 1-2 kernel launches.

### Compilation

6. **`torch.compile` support** — Added `compile` and `compile_mode` config flags. The RoPE buffer refactoring (item 1) was required to unblock `max-autotune` mode, which previously crashed with CUDA Graph errors.

### Data loading pipeline

7. **NVIDIA DALI** — GPU-pipelined JPEG decode + crop + flip, replacing torchvision CPU pipeline.

1. **Fused DALI augmentations** — ThreeAugment, ColorJitter, and normalization moved entirely into the DALI pipeline using `enable_conditionals=True`. Eliminates ~25ms of serial GPU augmentation per step.

1. **Local NVMe staging** — `prepare_data()` copies ImageNet to node-local `/scratch` with sentinel-based idempotency. Eliminates cross-rank I/O variance that caused DDP allreduce stalls on shared FS.

1. **Training recipe tuning** — Validation every 4 epochs (not every epoch), 12 workers, prefetch_factor=3.

## FLOP Analysis

ViT-5-Small per-sample FLOPs (12 blocks, dim 384, 6 heads, 201 tokens):

| Component                       | GFLOPs (fwd) |
| ------------------------------- | -----------: |
| Patch embed                     |         0.06 |
| Attention (QKV + proj)          |         5.72 |
| Attention (softmax)             |         0.12 |
| MLP                             |         5.72 |
| Head                            |        0.001 |
| **Total (fwd)**                 |     **11.6** |
| **Total (train: fwd + 2x bwd)** |     **34.9** |

## Benchmark & Profiling Scripts

| Script                                         | Purpose                                                              |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| `vit5_imagenet/bench_vit5_baseline.py`         | Original model eager throughput + FLOPs/MFU calculation              |
| `vit5_imagenet/bench_vit5_compile.py`          | Eager vs torch.compile (default) vs max-autotune + profiler          |
| `vit5_imagenet/bench_vit5_profile.py`          | Single-step CUDA kernel profiling (top-30 by time)                   |
| `vit5_imagenet/bench_vit5_optimized.py`        | Correctness checks + throughput for the optimized model              |
| `vit5_imagenet/profile_training_bottleneck.py` | Single-GPU component profiling (data / compute / optimizer)          |
| `vit5_imagenet/profile_step_breakdown.py`      | Per-phase step breakdown with CUDA events (data / fwd / bwd / optim) |
| `vit5_imagenet/verify_dali_fused.py`           | DALI fused pipeline correctness verification                         |
| `vit5_imagenet/validate_checkpoint.py`         | Checkpoint validation against W&B metrics                            |

### Running

Submit via SLURM (from the repo root):

```bash
sbatch benchmarks/vit5_imagenet/bench_optimized.sh
sbatch benchmarks/vit5_imagenet/bench_compile.sh
sbatch benchmarks/vit5_imagenet/bench_profile.sh
```

Logs go to `logs/`.

## Environment

- GPU: NVIDIA H100 SXM 80GB
- PyTorch 2.6.0+cu129
- CUDA 12.9
- Apex 0.1 (FusedLAMB)
- QuACK 0.2.10 (fused RMSNorm)
- NVIDIA DALI 1.53.0
- PyTorch Lightning 2.6.1
