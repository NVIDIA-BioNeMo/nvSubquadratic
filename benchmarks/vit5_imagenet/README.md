# ViT-5-Small ImageNet benchmarks

Steady-state throughput and memory measurements for ViT-5-Small on
ImageNet-1k.  Headline numbers live in
[`benchmarks/README.md`](../README.md); the scripts here are the
benchmarks that produce them.

Hardware target: **H100 SXM 80GB**, BF16, batch size 256.

## Throughput scripts

### `bench_vit5_baseline.py`

Forward+backward throughput of the unoptimized ViT-5-Small (eager,
torchvision dataloader).  Reference point for every optimisation that
follows.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/vit5_imagenet/bench_vit5_baseline.py
```

### `bench_vit5_compile.py`

Probes `torch.compile` configurations (default / `max-autotune` /
selective component compilation) on top of the baseline pipeline.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/vit5_imagenet/bench_vit5_compile.py
```

### `bench_vit5_hyena.py`

End-to-end Attention vs Hyena vs Hyena-FiLM throughput.  Each variant
shares the ViT-5-Small chassis; only the sequence mixer swaps.  The
Hyena variants can route their FFT through `torch.fft` (default) or
the `subquadratic_ops_torch` CUDA kernel via monkey-patching.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/vit5_imagenet/bench_vit5_hyena.py
```

### `bench_vit5_optimized.py`

Optimised pipeline (BF16 + compile + DALI-fused dataloader): the
production configuration whose numbers populate the headline tables
in [`../README.md`](../README.md).

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/vit5_imagenet/bench_vit5_optimized.py
```

### `bench_vit5_profile.py`

Per-phase timing breakdown for a single forward+backward step —
forward, attention/mixer, MLP, backward, optimizer.  Useful for
diagnosing where any new regression came from.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/vit5_imagenet/bench_vit5_profile.py
```

### `benchmark_imagenet_throughput.py`

Inference-only throughput benchmark for ImageNet-1k ViT-5 configs —
images/sec on a single GPU, comparable to the VMamba Table 1 numbers.
Builds the network directly from `nvsubquadratic` modules without
loading the full training-side config (avoids apex / DALI / Lightning),
so it runs on a vanilla setup.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/vit5_imagenet/benchmark_imagenet_throughput.py
```

## Verification scripts

- `verify_dali_fused.py` — sanity-checks output shapes/dtypes, value
  ranges, and visual augmentation distributions for the DALI fused
  pipeline; saves a side-by-side PNG vs the optimized pipeline.
- `validate_checkpoint.py` — downloads a W&B "best" checkpoint, loads
  it into the matching architecture, and runs a validation + test
  pass on ImageNet-1k.

## SLURM submit scripts

`scripts/` holds the SLURM drivers that invoke the `bench_vit5_*.py`
scripts with the standard ViT-5-Small configuration.

## Historical profiling

Historical profiling artefacts that motivated the current DALI-fused
pipeline live under
[`reports/vit5_imagenet_dataloader_profiling/`](../../reports/vit5_imagenet_dataloader_profiling/REPORT.md).
