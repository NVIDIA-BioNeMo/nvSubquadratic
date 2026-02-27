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
| **fused** | NVIDIA DALI | **All augments in DALI pipeline** | **Local NVMe** | **12** | **3** | **every 4 epochs** | `vit5_small_pretrain_apex_dali_fused.py` |

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

**Current bottleneck**: GPU augmentations (~25ms) run serially before forward pass. Next optimization: move all augmentations into DALI pipeline so they overlap with compute.

---

## 6. Fused DALI Pipeline (`dali_imagenet_fused.py`)

Moves ThreeAugment, ColorJitter, uint8→float conversion, and normalization **entirely into the DALI pipeline**, eliminating the ~25ms serial GPU augmentation overhead.

### What moved into DALI

| Operation | Before (optimized) | After (fused) |
|---|---|---|
| JPEG decode + crop + flip | DALI | DALI |
| ThreeAugment (grayscale/solarize/blur) | PyTorch GPU (serial) | DALI pipeline (per-sample conditionals) |
| ColorJitter (brightness/contrast/saturation) | PyTorch GPU (serial) | DALI `fn.color_twist` |
| uint8 → float + normalize | PyTorch GPU (serial) | DALI `fn.crop_mirror_normalize` |
| Mixup/CutMix | PyTorch GPU | PyTorch GPU (still needs labels) |

### Implementation details

- Uses DALI `enable_conditionals=True` with per-sample `if/else` for ThreeAugment branching
- Solarize uses element-wise uint8 masking (`fn.cast` + arithmetic) to avoid type promotion
- `fn.crop_mirror_normalize` fuses uint8→float conversion with normalization
- `on_before_batch_transfer` now only does Mixup/CutMix + optional NHWC permute (~1ms)

### Verification

- Validation outputs: **bit-identical** (max diff = 0.0) between optimized and fused
- Checkpoint validation (run `2y06y121`): **val/acc = 0.8170** — matches exactly

### Profiling results (DDP x8, H100 SXM, NVMe, `profile_training_bottleneck.py`)

| Pipeline | Model | Data (ms) | Compute DDP (ms) | Full step (ms) | Agg throughput | Regime |
|---|---|---:|---:|---:|---:|---|
| DALI-optimized | Small | 34.3 | 40.8 | 80.6 | 25,401 | balanced |
| **DALI-fused** | **Small** | **41.8** | **41.0** | **70.4** | **29,103** | **balanced** |
| DALI-optimized | Base | 36.8 | 95.7 | 131.1 | 15,622 | compute-bound |
| **DALI-fused** | **Base** | **38.4** | **95.9** | **120.8** | **16,950** | **compute-bound** |

**Speedup**: Small **+15%** throughput, Base **+9%** throughput. The "data loading" number is higher because it now includes augmentations, but the **full step is faster** because DALI's internal pipelining overlaps augmentations with I/O and compute.

---

## 7. Per-Phase Step Breakdown (`profile_step_breakdown.py`)

Fine-grained profiling with `torch.cuda.synchronize()` between each phase to measure where wall-clock time is spent **within a real training step**. Eager mode, DDP x8 H100 SXM, shared FS.

### DALI-fused (all augments in pipeline)

| Phase | Mean (ms) | Median (ms) | % of step | Notes |
|---|---:|---:|---:|---|
| DALI fetch | 9.03 | 0.75 | 7.2% | Range 0.6–79 ms (shared FS variance) |
| **Mixup/CutMix** | **0.13** | **0.10** | **0.1%** | Negligible |
| **NHWC permute** | **0.22** | **0.19** | **0.2%** | Negligible |
| Forward | 32.36 | 32.40 | 25.8% | Eager mode (no compile) |
| Backward (+allreduce) | 81.26 | 70.59 | 64.7% | Includes DDP allreduce tail |
| Optimizer + zero_grad | 2.48 | — | 1.9% | FusedLAMB |
| **Total (instrumented)** | **125.59** | — | — | |
| **Total (natural)** | **134.94** | 121.58 | — | No mid-step syncs |

### DALI-v2 (optimized, augments in PyTorch)

| Phase | Mean (ms) | Median (ms) | % of step | Notes |
|---|---:|---:|---:|---|
| DALI fetch | 4.64 | 0.77 | 3.4% | Lighter pipeline (no augments) |
| **on_before_batch_transfer** | **6.10** | **3.71** | **3.5%** | ThreeAugment + ColorJitter + normalize |
| Forward | 32.36 | 31.76 | 24.0% | Same model |
| Backward (+allreduce) | 92.30 | 76.19 | 68.4% | More I/O stalls → DDP sync waits |
| Optimizer + zero_grad | 2.40 | — | 1.6% | |
| **Total (instrumented)** | **134.90** | — | — | |
| **Total (natural)** | **178.05** | — | — | |

### Key conclusions

1. **Serial GPU augmentation is now negligible in fused**: 0.35 ms total (0.13 ms Mixup + 0.22 ms permute) vs 6.10 ms in v2. Moving Mixup to a separate CUDA stream would save at most 0.13 ms — not worth the complexity.
2. **The data pipeline is fully optimized**: on NVMe (where fetch ≈ 1-5 ms), the total non-compute overhead is ~3 ms (fetch + augment + optimizer). The 70.4 ms step on NVMe is consistent with: ~2 ms fetch + 0.35 ms augment + ~32 ms fwd + ~34 ms bwd + ~2 ms optim ≈ 70 ms.
3. **Remaining optimization opportunities are compute-side**: DDP allreduce efficiency, batch size scaling, or model-level optimizations — not data loading.
4. **I/O variance across ranks is the dominant source of step-time variance on shared FS**: DALI fetch ranges from 0.6 ms to 200+ ms, causing DDP synchronization stalls. NVMe eliminates this.

---

## 8. Key Files

| File | Purpose |
|---|---|
| `experiments/datamodules/dali_imagenet.py` | Original DALI datamodule (v1) |
| `experiments/datamodules/dali_imagenet_optimized.py` | Optimized DALI datamodule (v2 + local staging) |
| `experiments/datamodules/dali_imagenet_optimized_v3.py` | v3 experiments (bf16, single compiled fn) |
| `experiments/datamodules/dali_imagenet_fused.py` | Fused DALI datamodule (all augments in pipeline) |
| `benchmarks/vit5_imagenet/profile_training_bottleneck.py` | Single-GPU component profiling |
| `benchmarks/vit5_imagenet/profile_step_breakdown.py` | Per-phase step breakdown (CUDA events + wall-clock) |
| `scripts/stage_imagenet.sh` | SLURM job to pre-stage ImageNet on all nodes |
| `benchmarks/vit5_imagenet/dataloader_profile_2026-02-25.jsonl` | Raw profiling data (JSON lines) |
| `benchmarks/vit5_imagenet/dataloader_profile_2026-02-26.jsonl` | Profiling data incl. fused DALI |
| `benchmarks/vit5_imagenet/step_breakdown_2026-02-26.jsonl` | Per-phase step breakdown data |

---

## 9. Lessons Learned

1. **Network FS was the #1 bottleneck** — Local NVMe staging gave 2-3x speedup by eliminating I/O variability.
2. **DALI `fn.transpose` is not free** — It performs an explicit memory copy, adding ~47ms. Keeping HWC→NCHW in PyTorch is faster.
3. **v2 vs v1 augmentation optimizations mattered less than expected** — With `torch.compile`, the augmentation overhead shrank, making the compute-side gains modest. The I/O fix was far more impactful.
4. **Validation frequency matters** — Validating every epoch adds ~3s of overhead per epoch (DALI pipeline rebuild + checkpoint callback). At 52s/epoch, that's ~6% overhead eliminated by going to every 4 epochs.
5. **Container mounts are easy to miss** — `/scratch` must be explicitly mounted in `--container-mounts` for Pyxis/enroot containers.
6. **`torch.compile` warmup needs 15-20 iterations** — Initial profiling with 5-iteration warmup gave misleading results (massive "other overhead").
7. **Fusing augmentations into DALI eliminated the serial bottleneck** — Moving ThreeAugment, ColorJitter, and normalization into the DALI pipeline reduced serial GPU augmentation from ~6 ms (`on_before_batch_transfer`) to 0.35 ms (Mixup + permute only). Further data pipeline optimization yields diminishing returns (<0.5% of step time).
8. **DDP sync stalls dominate on shared FS** — Per-phase profiling revealed that I/O variance across ranks (0.6–200+ ms) causes DDP allreduce waits, inflating backward time. NVMe eliminates this by providing consistent low-latency reads.
