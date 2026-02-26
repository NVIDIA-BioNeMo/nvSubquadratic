# ViT-5 ImageNet Data Loading & Training Optimization Tracker

All experiments use ViT-5-Small (22M params), ImageNet-1k (224x224), batch size 256 per GPU, 8x H100 SXM 80GB, BF16-mixed, Apex FusedLAMB, `torch.compile(mode="max-autotune")`.

---

## 1. Model & Optimizer Optimizations (prior work)

| Change | File | Impact |
|---|---|---|
| RoPE precomputation via `register_buffer` | `vit5_attention.py` | Eliminated graph breaks, enabled CUDA Graphs |
| SDPA auto-select (CuDNN on H100) | `vit5_attention.py` | Faster attention kernel |
| Removed redundant dtype casts around SDPA | `vit5_attention.py` | Fewer GPU ops |
| QuACK fused RMSNorm (Triton kernel) | `rms_norm.py` | Replaced float32 upcast RMSNorm |
| Apex FusedLAMB | config | Multi-tensor fused optimizer (0.6ms/step) |
| `torch.compile(max-autotune)` | config | 2x compute speedup: 78ms → 39ms fwd+bwd |

**Single-GPU model throughput** (before data loading work):

| Mode | Step time | Throughput |
|---|---:|---:|
| Eager (original) | 159.2 ms | 1,608 img/s |
| torch.compile (default) | 46.0 ms | 5,560 img/s |
| torch.compile (max-autotune) | 32.0 ms | 8,003 img/s |

---

## 2. Data Loading Experiments

### Configurations tested

| Version | Dataloader | Augmentations | Storage | Workers | Prefetch | Val freq | Config file |
|---|---|---|---|---|---|---|---|
| CPU baseline | PyTorch DataLoader | torchvision on CPU | Network FS | 14 | 2 | every epoch | `vit5_small_pretrain_apex.py` |
| **v1** (DALI) | NVIDIA DALI | PyTorch on GPU | Network FS | 8 | 2 | every epoch | `vit5_small_pretrain_apex_dali.py` |
| **v2** (DALI optimized) | NVIDIA DALI | torch.compile-friendly GPU | Network FS | 8 | 2 | every epoch | `vit5_small_pretrain_apex_dali_optimized_v2.py` |
| **v3** (DALI optimized v3) | NVIDIA DALI | bf16 + single compiled fn | Network FS | 8 | 2 | every epoch | `vit5_small_pretrain_apex_dali_optimized_v3.py` |
| **optimized_plus** | NVIDIA DALI | torch.compile-friendly GPU | **Local NVMe** | **12** | **3** | **every 4 epochs** | `vit5_small_pretrain_apex_dali_optimized_plus.py` |

### v2 augmentation changes (`dali_imagenet_optimized.py`)

1. Module-level `gaussian_blur` import (no per-forward import)
2. Device-cached normalization tensors via `register_buffer`
3. `torch.where`-based blending in `_BatchThreeAugment` (no boolean-index scatter)
4. Vectorised random permutations via `argsort` in `_BatchColorJitter`
5. Fused uint8→float + normalization for validation path (`_fused_val_normalize`, compiled)
6. `_BatchColorJitter` wrapped in `torch.compile`

### v3 additional changes (`dali_imagenet_optimized_v3.py`)

1. Attempted DALI `fn.transpose` for CHW output → **reverted** (added ~47ms due to explicit memory copy)
2. bf16 precision for augmentation pipeline
3. Single compiled function for ColorJitter + normalization

### optimized_plus additional changes

1. **Local NVMe staging** — `prepare_data()` copies ImageNet to `/scratch/$USER/imagenet_dataset` with sentinel file for idempotency
2. **Validation every 4 epochs** — reduces DALI pipeline interruptions
3. **12 workers** (up from 8) — better utilization of 16 CPU cores/GPU
4. **Prefetch factor 3** (up from 2) — deeper buffer for micro-stall absorption
5. `/scratch` mounted into container via `--container-mounts`

---

## 3. Single-GPU Profiling Results

All from `profile_training_bottleneck.py`, H100 SXM, batch 256. Measured components: data loading, forward+backward, optimizer.

### Compiled mode (head-to-head on same node, b65c909e-24)

| Dataloader | Data (ms) | Compute (ms) | Optim (ms) | Full step (ms) | Bottleneck |
|---|---:|---:|---:|---:|---|
| DALI v1 | 41.7 | 39.4 | 0.7 | 91.7 | balanced |
| **DALI v2** | **42.0** | **39.4** | **0.6** | **74.0** | balanced |
| DALI v3 | 89.4 | 39.4 | 0.7 | 123.8 | data (fn.transpose overhead) |

### Eager mode baselines

| Dataloader | Data (ms) | Compute (ms) | Full step (ms) | Bottleneck |
|---|---:|---:|---:|---|
| CPU (torchvision) | 105.4 | 78.2 | 223.9 | data + compute |
| DALI v1 (eager) | 46.4 | 78.9 | 136.7 | compute |

---

## 4. Multi-GPU Training Throughput (8x H100 DDP)

Measured from live training logs. All use `torch.compile(max-autotune)`.

| Version | Storage | it/s (per GPU) | ms/step | Epoch time | Current epoch | Est. 800ep wall time |
|---|---|---:|---:|---|---:|---|
| v1 (DALI) | Network FS | **5.3** | 189 | ~2m00s | 420 | ~37h |
| v2 (DALI optimized) | Network FS | **6.3** | 159 | ~1m40s | 400 | ~31h |
| **optimized_plus** | **Local NVMe** | **12.1** | 83 | ~52s | **736** | **~12h** |

### Speedup summary

| Comparison | Speedup |
|---|---:|
| optimized_plus vs v1 | **2.3x** |
| optimized_plus vs v2 | **1.9x** |
| optimized_plus vs CPU baseline (estimated) | **~5x** |

---

## 5. Step Time Breakdown (optimized_plus, 83ms/step)

```
[DALI prefetch (overlapped)]  [GPU augmentations]  [fwd + bwd]  [DDP allreduce + optim]
                               |---- ~25ms ---------|-- 39ms ----|------ ~10ms ---------|
```

| Component | Time (ms) | Notes |
|---|---:|---|
| DALI batch delivery | ~15-20 | Overlapped with previous step (prefetch) |
| GPU augmentations (`on_before_batch_transfer`) | ~25 | Permute, ThreeAugment, ColorJitter (compiled), normalize — **serial, not overlapped** |
| Forward + backward | 39.4 | `torch.compile(max-autotune)` |
| DDP allreduce tail | ~9 | Partially overlapped with backward |
| Optimizer (FusedLAMB) | 0.7 | Multi-tensor fused |

**Current bottleneck**: GPU augmentations (~25ms) run serially before forward pass. Next optimization: double-buffered augmentation on a separate CUDA stream to overlap with previous step's compute.

---

## 6. Key Files

| File | Purpose |
|---|---|
| `experiments/datamodules/dali_imagenet.py` | Original DALI datamodule (v1) |
| `experiments/datamodules/dali_imagenet_optimized.py` | Optimized DALI datamodule (v2 + local staging) |
| `experiments/datamodules/dali_imagenet_optimized_v3.py` | v3 experiments (bf16, single compiled fn) |
| `scripts/profile_training_bottleneck.py` | Single-GPU component profiling |
| `scripts/stage_imagenet.sh` | SLURM job to pre-stage ImageNet on all nodes |
| `benchmarks/dataloader_profile_2026-02-25.jsonl` | Raw profiling data (JSON lines) |

---

## 7. Lessons Learned

1. **Network FS was the #1 bottleneck** — Local NVMe staging gave 2-3x speedup by eliminating I/O variability.
2. **DALI `fn.transpose` is not free** — It performs an explicit memory copy, adding ~47ms. Keeping HWC→NCHW in PyTorch is faster.
3. **v2 vs v1 augmentation optimizations mattered less than expected** — With `torch.compile`, the augmentation overhead shrank, making the compute-side gains modest. The I/O fix was far more impactful.
4. **Validation frequency matters** — Validating every epoch adds ~3s of overhead per epoch (DALI pipeline rebuild + checkpoint callback). At 52s/epoch, that's ~6% overhead eliminated by going to every 4 epochs.
5. **Container mounts are easy to miss** — `/scratch` must be explicitly mounted in `--container-mounts` for Pyxis/enroot containers.
6. **`torch.compile` warmup needs 15-20 iterations** — Initial profiling with 5-iteration warmup gave misleading results (massive "other overhead").
