# ViT-5-Small ImageNet dataloader profiling — 2026-02

Investigation that diagnosed the CPU-decode bottleneck in the
ViT-5-Small ImageNet training pipeline and motivated the move to the
DALI-fused dataloader whose steady-state numbers now live in the
headline tables of [`benchmarks/README.md`](../../benchmarks/README.md).

## Context

- Model: **ViT-5-Small** (22M params, 224x224 input, batch size 256, BF16).
- Hardware: **H100 SXM 80GB**.
- Goal: identify the wall-clock bottleneck in the training step and
  cut it.

## Pipeline at the time

`torchvision.datasets.ImageFolder` -> CPU PIL JPEG decode -> CPU
transforms (RandomResizedCrop, ThreeAugment, ColorJitter, Normalize)
-> `pin_memory` -> H2D copy.

## Day 1 (Feb 25) — bottleneck identification

Per-component profiling with [`profile_training_bottleneck.py`](profile_training_bottleneck.py)
([`profile_training_bottleneck.sh`](profile_training_bottleneck.sh)
drives it via SLURM).  All numbers in
[`dataloader_profile_2026-02-25.jsonl`](dataloader_profile_2026-02-25.jsonl).

### Previous runs (before today)

| Mode  | Workers | Prefetch | Data ms | Compute ms | Optim ms | Full ms | Throughput | Data % |
| ----- | ------: | -------: | ------: | ---------: | -------: | ------: | ---------: | -----: |
| eager |      16 |        2 |   111.6 |      230.7 |     17.3 |   354.5 |        722 |  31.5% |
| eager |      14 |        2 |    98.8 |      231.2 |     19.3 |   371.7 |        689 |  26.6% |

*The runs above used `torch_optimizer.Lamb` (non-fused).  Apex
FusedLAMB is the optimiser below.*

**Key insight.**  In eager mode, compute (231 ms) > data
(99-112 ms), so data is "only" ~30% of the step.  Under
`torch.compile(max-autotune)`, compute drops to ~32 ms, making data
**~3x slower than compute** — the step is then **fully data-bound**.

### Today's runs (Apex FusedLAMB, 14 workers, single GPU)

| Mode     | Workers | Prefetch | Data ms | Compute ms | Optim ms | Full ms | Throughput | Data % |
| -------- | ------: | -------: | ------: | ---------: | -------: | ------: | ---------: | -----: |
| eager    |      14 |        2 |   104.1 |      231.3 |      0.7 |   365.8 |        700 |  28.4% |
| compiled |      14 |        2 |    97.6 |      196.4 |      0.7 |   324.2 |        790 |  30.1% |

The compute numbers above are FP32 (autocast was missing from the
profiling script).  Combining BF16 + compile (from
`bench_vit5_optimized.py`) drops compute to ~32 ms, making data ~75%
of the step — **fully data-bound**.

### Prefetch factor sweep (14 workers)

| prefetch_factor | Data ms (eager) | Throughput | Data ms (compiled) | Throughput |
| --------------: | --------------: | ---------: | -----------------: | ---------: |
|               2 |            87.7 |      2,919 |               85.7 |      2,988 |
|               4 |           108.8 |      2,354 |               99.0 |      2,586 |
|               8 |           105.3 |      2,431 |               98.3 |      2,604 |
|              16 |           111.4 |      2,299 |              104.4 |      2,453 |

**Verdict:** `prefetch_factor=2` is consistently best.  Higher values
add memory pressure without benefit.

### GPU decode experiment (nvJPEG + GPU transforms)

Raw JPEG bytes from CPU workers -> batch `decode_jpeg` on GPU (nvJPEG)
-> GPU transforms.  Per-image `RandomResizedCrop` is unavoidable
(variable input sizes); flip / jitter / three-augment can be batched.

| Variant                  | Data I/O ms | Compute ms | Optim ms | Overhead ms | Full ms | Throughput |
| ------------------------ | ----------: | ---------: | -------: | ----------: | ------: | ---------: |
| CPU baseline (PIL, 14w)  |       108.0 |       78.5 |      0.6 |        42.8 |   229.8 |      1,114 |
| GPU decode (per-img all) |        53.2 |       78.2 |      0.6 |       392.4 |   524.4 |        488 |
| GPU decode (batched aug) |        54.6 |       78.3 |      0.6 |       298.7 |   432.2 |        592 |

**Verdict.**  GPU decode halves the I/O time (108 -> 54 ms), but
per-image `RandomResizedCrop` on GPU (256 sequential bicubic resize
kernels) adds ~300 ms overhead.  Net result is **~2x slower** than
the CPU pipeline, where 14 workers parallelise crop+resize across
CPU cores.  Batching the post-crop augmentations saved ~94 ms but
wasn't enough.

## Day 2 (Feb 26) — step breakdown under compiled BF16

Re-instrumented with [`profile_step_breakdown.py`](profile_step_breakdown.py)
([`profile_step_breakdown.sh`](profile_step_breakdown.sh) drives the
DDP variant); raw measurements in
[`step_breakdown_2026-02-26.jsonl`](step_breakdown_2026-02-26.jsonl).
The two follow-up sweeps under
[`dataloader_profile_2026-02-26.jsonl`](dataloader_profile_2026-02-26.jsonl)
cross-checked the prefetch_factor sweep on DALI-v2 across small/base
model sizes.

Representative numbers (DDP=8, batch=256, model=small,
compiled+BF16, DALI fused):

| Phase            |   Wall ms |
| ---------------- | --------: |
| dali_fetch       |      32.4 |
| forward          |      17.5 |
| backward         |      88.8 |
| optim_step       |       2.4 |
| **natural step** | **178.8** |

GPU-event view ("gpu_phases_ms" key in the JSONL) puts backward at
~142 ms, fetch at ~19 ms, optim at ~0.5 ms — the gap between the
wall-clock view and the GPU-event view confirms the step is still
data-bound under DALI v2 (the CPU prefetch isn't fully overlapping
the backward).  Theoretical minimum on this run was ~109 ms; the
~70 ms gap from observed (~179 ms) to theoretical (~109 ms) is where
the eventual DALI-fused pipeline closed up.

Subsequent DALI variants ("dali-v2" vs "fused") and small / base
model sweeps in
[`dataloader_profile_2026-02-26.jsonl`](dataloader_profile_2026-02-26.jsonl)
confirmed the fused pipeline both reduces fetch ms and removes the
per-step CPU augmentation overhead — the change that took the step
from ~179 ms to the ~79-83 ms reported in the headline table of
[`benchmarks/README.md`](../../benchmarks/README.md).

## Outcome

These captures motivated the DALI-fused dataloader and the
local-NVMe storage move.  The follow-up steady-state numbers are
reported in [`benchmarks/README.md`](../../benchmarks/README.md) under
"v2 (DALI optimized) -> optimized_plus -> fused" (4.8x – 5.0x over the
CPU baseline).

A snapshot of the GPU-decode experiment is preserved at
[`experiments/datamodules/_tmp_imagenet_gpu_decode.py`](../../experiments/datamodules/_tmp_imagenet_gpu_decode.py).

## Files

- [`profile_training_bottleneck.py`](profile_training_bottleneck.py),
  [`profile_training_bottleneck.sh`](profile_training_bottleneck.sh)
  — the Day 1 per-component profiler and SLURM driver.
- [`profile_step_breakdown.py`](profile_step_breakdown.py),
  [`profile_step_breakdown.sh`](profile_step_breakdown.sh) — the Day 2
  in-step phase breakdown and SLURM driver (DDP-aware).
- [`dataloader_profile_2026-02-25.jsonl`](dataloader_profile_2026-02-25.jsonl)
  — raw Day 1 runs.
- [`dataloader_profile_2026-02-26.jsonl`](dataloader_profile_2026-02-26.jsonl)
  — Day 2 prefetch sweep across model sizes.
- [`step_breakdown_2026-02-26.jsonl`](step_breakdown_2026-02-26.jsonl)
  — Day 2 in-step phase breakdown.
