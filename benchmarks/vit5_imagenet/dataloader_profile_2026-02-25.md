# Dataloader Profiling — 2026-02-25

ViT-5-Small (22M params, 224x224, BS=256, BF16) on H100 SXM 80GB.
Goal: identify and reduce dataloading bottleneck in the training loop.

## Current Pipeline

`torchvision.datasets.ImageFolder` → CPU PIL JPEG decode → CPU transforms
(RandomResizedCrop, ThreeAugment, ColorJitter, Normalize) → pin_memory → H2D copy.

## Results

Results are stored in `dataloader_profile_2026-02-25.jsonl` (one JSON object per run).

### Previous runs (before today)

| Mode  | Workers | Prefetch | Data ms | Compute ms | Optim ms | Full ms | Throughput | Data % |
| ----- | ------: | -------: | ------: | ---------: | -------: | ------: | ---------: | -----: |
| eager |      16 |        2 |   111.6 |      230.7 |     17.3 |   354.5 |        722 |  31.5% |
| eager |      14 |        2 |    98.8 |      231.2 |     19.3 |   371.7 |        689 |  26.6% |

*Note: Above used `torch_optimizer.Lamb` (non-fused). Runs below use Apex FusedLAMB.*

**Key insight:** In eager mode, compute (231ms) > data (99-112ms), so data is "only" ~30% of the step.
But with `torch.compile(max-autotune)`, compute drops to ~32ms, making data **~3x slower than compute** → **fully data-bound**.

### Today's runs (Apex FusedLAMB, 14 workers, single GPU)

| Mode     | Workers | Prefetch | Data ms | Compute ms | Optim ms | Full ms | Throughput | Data % |
| -------- | ------: | -------: | ------: | ---------: | -------: | ------: | ---------: | -----: |
| eager    |      14 |        2 |   104.1 |      231.3 |      0.7 |   365.8 |        700 |  28.4% |
| compiled |      14 |        2 |    97.6 |      196.4 |      0.7 |   324.2 |        790 |  30.1% |

*Note: compute numbers above are FP32 (missing autocast in profiling script). With BF16+compile,
compute drops to ~32ms (from bench_vit5_optimized.py), making data ~75% of the step → fully data-bound.*

## Prefetch Factor Sweep (14 workers)

| prefetch_factor | Data ms (eager) | Throughput | Data ms (compiled) | Throughput |
| --------------: | --------------: | ---------: | -----------------: | ---------: |
|               2 |            87.7 |      2,919 |               85.7 |      2,988 |
|               4 |           108.8 |      2,354 |               99.0 |      2,586 |
|               8 |           105.3 |      2,431 |               98.3 |      2,604 |
|              16 |           111.4 |      2,299 |              104.4 |      2,453 |

**Verdict:** `prefetch_factor=2` is consistently best. Higher values add memory pressure without benefit.

## GPU Decode Experiment (nvJPEG + GPU transforms)

Approach: raw JPEG bytes from CPU workers → batch `decode_jpeg` on GPU (nvJPEG) → GPU transforms.
Per-image `RandomResizedCrop` is unavoidable (variable input sizes); flip/jitter/three-augment batched.

| Variant                  | Data I/O ms | Compute ms | Optim ms | Overhead ms | Full ms | Throughput |
| ------------------------ | ----------: | ---------: | -------: | ----------: | ------: | ---------: |
| CPU baseline (PIL, 14w)  |       108.0 |       78.5 |      0.6 |        42.8 |   229.8 |      1,114 |
| GPU decode (per-img all) |        53.2 |       78.2 |      0.6 |       392.4 |   524.4 |        488 |
| GPU decode (batched aug) |        54.6 |       78.3 |      0.6 |       298.7 |   432.2 |        592 |

**Verdict:** GPU decode halves the I/O time (108 → 54ms), but per-image `RandomResizedCrop` on GPU
(256 sequential bicubic resize kernels) adds ~300ms overhead. Net result is **~2x slower** than the
CPU pipeline where 14 workers parallelize crop+resize across CPU cores. Batching the post-crop
augmentations (flip, jitter, three-augment) saved ~94ms but is not enough.

Code preserved in `experiments/datamodules/_tmp_imagenet_gpu_decode.py`.

## Ideas Under Evaluation

1. ~~Reduce num_workers to match CPU count~~ — Done. ~13% data loading improvement.
1. ~~GPU JPEG decode + GPU transforms~~ — Tested. **2x slower** due to per-image resize overhead.
1. NVIDIA DALI pipeline (truly pipelined GPU decode + transforms)
1. WebDataset / sharded tar format
1. ffcv `.beton` format
1. Pre-decoded dataset on local NVMe
