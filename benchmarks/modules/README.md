# Module-level benchmarks

## TriHyena (`bench_tri_hyena.py`)

Times `QKVSequenceMixer(TriangleHyena)` on a pair tensor `[B, L, L, C]`
(default `B=1`, `C=128`, `L` in 128, 256, 384, 512, 1024, 2048, bf16, hard
window `max_offset=256`) with CUDA events, an L2 flush before every timed region
and a one-second warm-up so the GPU sits at sustained clocks. Arch names are
`<window>-<torch|subq_ops>-<eager|compiled>`; `compiled` applies
`torch.compile(mode="reduce-overhead")` from the outside. `--mode train`
(default) reports forward, backward and full-step time plus peak memory;
`--mode infer` reports the forward under `torch.inference_mode()`. Results are
written incrementally to a CSV. Pin the GPU with `CUDA_VISIBLE_DEVICES=<uuid>`
on multi-GPU hosts.

```bash
# window x backend x eager/compiled, starting node
PYTHONPATH=. python benchmarks/modules/bench_tri_hyena.py --mode infer

# ending-node update
PYTHONPATH=. python benchmarks/modules/bench_tri_hyena.py --mode infer --end-node
```

### Memory columns

All figures come from the PyTorch caching allocator with the persistent L2 flush
buffer subtracted (its 2 MiB segment rounding for the reserved figures). Compile
state and cached segments are reset before every (arch, L) measurement
(`torch._dynamo.reset()` is the only way to return a cudagraph private pool, so
without it one arch's pool would sit in the next arch's baseline and reserved
peak). Peak columns are the max over the `--n-bench` iterations.

| column                     | meaning                                                                                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `setup_ms`                 | wall time of the warm-up: compile and graph capture included                                                                                                                                           |
| `fwd_peak_mb`              | peak allocated during the forward: resident memory (module, input, anything alive) plus its transient                                                                                                  |
| `fwd_peak_delta_mb`        | `fwd_peak_mb` minus the memory allocated right before that forward: the forward's own working set                                                                                                      |
| `step_peak_mb`             | peak allocated during a training step (forward + backward), resident memory included                                                                                                                   |
| `step_peak_delta_mb`       | `step_peak_mb` minus the memory allocated before the step, previous gradients already freed                                                                                                            |
| `act_mb`                   | memory allocated after the forward of a training step, before backward (resident + saved activations)                                                                                                  |
| `setup_peak_delta_mb`      | peak allocated during the warm-up minus the memory allocated before it                                                                                                                                 |
| `setup_peak_reserved_mb`   | peak reserved during the warm-up                                                                                                                                                                       |
| `steady_reserved_delta_mb` | reserved after the timed calls + `empty_cache()` minus reserved before warm-up: what setup leaves behind (read last: an `empty_cache()` before timing puts cudaMalloc re-growth into the timed region) |

Under `reduce-overhead` a replay allocates nothing, so `fwd_peak_delta_mb` reads 0
and `fwd_peak_mb` is the resident baseline; the graph's static buffers show up in
`setup_peak_*` and `steady_reserved_delta_mb` (the cudagraph private pool, i.e. the
steady-state working set) instead, while for eager archs
`steady_reserved_delta_mb` is close to 0. To compare peak memory across archs use
`fwd_peak_delta_mb` / `step_peak_delta_mb` for eager and compile-default,
`steady_reserved_delta_mb` for `reduce-overhead`. In `--mode train` the warm-up
runs full steps, so `steady_reserved_delta_mb` also contains the parameter
gradients left by the last warm-up step.

### Results (GB300, 1300 W cap, bf16, B=1, C=128, hard window 256, `--mode infer`, 2026-11-03)

Inference forward time in ms, starting node / ending node (`--end-node`):

| L    | box-subq_ops-compiled | gaussian-subq_ops-compiled | none-subq_ops-compiled | box-subq_ops-eager | box-torch-eager |
| ---- | --------------------- | -------------------------- | ---------------------- | ------------------ | --------------- |
| 128  | 0.35 / 0.46           | 0.67 / 0.43                | 0.65 / 0.34            | 1.63 / 2.52        | 1.61 / 1.94     |
| 256  | 0.41 / 0.51           | 0.48 / 0.37                | 0.39 / 0.34            | 2.82 / 2.08        | 1.76 / 2.29     |
| 384  | 0.80 / 0.45           | 0.48 / 0.65                | 0.51 / 0.45            | 1.16 / 1.78        | 2.27 / 2.15     |
| 512  | 0.57 / 0.56           | 0.56 / 0.63                | 0.74 / 0.53            | 1.29 / 1.89        | 3.66 / 3.41     |
| 1024 | 1.21 / 1.34           | 1.26 / 1.32                | 1.30 / 1.39            | 4.48 / 5.15        | 14.54 / 13.76   |
| 2048 | 4.86 / 5.07           | 4.82 / 5.23                | 4.79 / 5.13            | 18.07 / 20.77      | 59.53 / 58.31   |

Compiled timings below L=1024 are launch-bound and move by up to 0.15 ms between
runs. The window choice (box / gaussian / none) does not change the compiled
timings; at L=2048 the compiled `subq_ops` forward is 3.7x faster than eager
`subq_ops` and 12x faster than the eager torch backends.

Memory in MB, starting node / ending node: `fwd_peak_delta_mb` for the eager
archs (the forward's working set) and `steady_reserved_delta_mb` for the
compiled ones (the cudagraph working set):

| L    | box-subq_ops-eager `fwd_peak_delta_mb` | box-torch-eager `fwd_peak_delta_mb` | box-subq_ops-compiled `steady_reserved_delta_mb` | gaussian-subq_ops-compiled `steady_reserved_delta_mb` |
| ---- | -------------------------------------- | ----------------------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| 128  | 52 / 40                                | 101 / 89                            | 54 / 170                                         | 138 / 138                                             |
| 256  | 208 / 160                              | 401 / 353                           | 114 / 330                                        | 182 / 182                                             |
| 384  | 468 / 360                              | 902 / 794                           | 214 / 410                                        | 262 / 262                                             |
| 512  | 832 / 640                              | 1603 / 1411                         | 354 / 522                                        | 374 / 374                                             |
| 1024 | 3329 / 2561                            | 6409 / 5641                         | 1314 / 1430                                      | 1398 / 1398                                           |
| 2048 | 13316 / 10244                          | 25613 / 22541                       | 5156 / 5272                                      | 5240 / 5240                                           |

The eager `subq_ops` starting-node forward holds the `3C`-wide fused short-conv
output (two extra `[B*L, L, C]` buffers at the peak, 1 GB each at L=2048); the
ending node convolves the three streams separately and peaks lower.
