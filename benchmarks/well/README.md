# The Well benchmarks

Throughput, dataloader, and end-to-end training-step benchmarks for
ViT-5 / Hyena on the
[The Well](https://github.com/PolymathicAI/the_well) PDE benchmark
suite.  Each script targets one of the WELL sub-datasets via a
`--config` pointing at an experiment under
[`examples/well/`](../../examples/well/).

Hardware target: **H100 SXM 80GB**, BF16.

## Throughput scripts

### `bench_ab_comparison.py`

A/B sweep that runs a baseline configuration and an optimised one
back-to-back on the same GPU.  Optimisations covered:
`persistent_workers=True`, `prefetch_factor=4`, and direct
`output_fields` extraction instead of `formatter.process_input()`.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/well/bench_ab_comparison.py \
    --config examples/well/supernova_explosion_64/cfg_vit5_attention.py
```

### `bench_dataloader.py`

Isolated dataloader throughput — how fast batches can leave disk and
arrive on GPU, with the model out of the way.  Useful for diagnosing
I/O bottlenecks before adding the compute path.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/well/bench_dataloader.py \
    --config examples/well/supernova_explosion_64/cfg_vit5_attention.py
```

### `bench_training_step.py`

End-to-end training step throughput (forward + loss + backward +
optimiser), no W&B / no checkpointing — what `iter/s` will look like
on the real training loop, minus logging overhead.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/well/bench_training_step.py \
    --config examples/well/supernova_explosion_64/cfg_vit5_attention.py --compile
```

## Profiling scripts

- `profile_timing.py` — per-phase forward/backward/optimiser breakdown
  for the Gray-Scott Hyena config.  Compiled vs uncompiled with
  proper CUDA synchronisation.
- `profile_training_loop.py` — diagnoses the gap between pure compute
  and PyTorch-Lightning-reported iteration time (DataLoader,
  preprocessing, overhead).

## Verification

- `verify_vrmse.py` — cross-checks the VRMSE metric implementation by
  running multiple independent computation paths against a saved
  checkpoint.

## Drivers & parsers

- `parse_bench.py` — summarises the output of the SLURM sweep driver.
- `submit_bench_ivi.sh` — IVI cluster SLURM driver.

## Targeted WELL sub-datasets

The configs under [`examples/well/`](../../examples/well/) cover the
WELL sub-datasets we routinely benchmark (Gray-Scott, supernova
explosion, etc.).  See [`examples/well/README.md`](../../examples/well/README.md)
for the active list.
