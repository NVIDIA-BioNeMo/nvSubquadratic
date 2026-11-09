# Module-level benchmarks

## TriHyena (`bench_tri_hyena.py`)

Times `QKVSequenceMixer(TriangleHyena)` on a pair tensor `[B, L, L, C]`
(default `B=1`, `C=128`, `L` in 128, 256, 384, 512, 1024, 2048, bf16, hard
window `max_offset=256`, `right_padding=True` with the last 12.5% of every
sample padded, `--right-pad-frac`) with CUDA events, an L2 flush before every
timed region and a one-second warm-up so the GPU sits at sustained clocks. Arch names are
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

All `_mb` columns are MiB and come from the PyTorch caching allocator with the
persistent L2 flush buffer subtracted (its 2 MiB segment rounding for the
reserved figures). Compile
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
gradients left by the last warm-up step, and for compiled archs `act_mb` and
`step_peak_*` leave out the activations held in the cudagraph pool (L=512 starting
node: `act_mb` 194 compiled vs 964 eager `subq_ops`).

### Results (GB300 CC 10.3, 1300 W cap, bf16, B=1, C=128, hard window 256, `--mode infer`, 2026-11-09)

torch 2.13.0+cu132, subquadratic-ops-torch-cu13 0.3.1, cuDNN 9.27. At every `L`
in the tables below the `subq_ops` long conv dispatches to `explicit_conv1d` (fp32 would use
`fft_conv1d`); the `torch` archs use `torch.fft`. Eager `subq_ops` runs the short
conv with `causal_conv1d`, compiled archs with Inductor.

Inference forward time in ms, starting node / ending node (`--end-node`), median of
five runs (each the mean of 10 calls):

| L    | box-subq_ops-compiled | gaussian-subq_ops-compiled | none-subq_ops-compiled | box-subq_ops-eager | box-torch-eager |
| ---- | --------------------- | -------------------------- | ---------------------- | ------------------ | --------------- |
| 128  | 0.32 / 0.31           | 0.32 / 0.36                | 0.35 / 0.38            | 1.64 / 1.86        | 1.49 / 1.81     |
| 256  | 0.29 / 0.37           | 0.34 / 0.35                | 0.39 / 0.42            | 1.60 / 1.83        | 1.59 / 2.51     |
| 384  | 0.42 / 0.47           | 0.45 / 0.50                | 0.49 / 0.47            | 1.62 / 2.14        | 2.35 / 2.40     |
| 512  | 0.62 / 0.55           | 0.58 / 0.59                | 0.52 / 0.57            | 1.95 / 1.88        | 3.78 / 3.50     |
| 1024 | 1.30 / 1.39           | 1.31 / 1.38                | 1.30 / 1.35            | 4.83 / 5.48        | 14.84 / 14.09   |
| 2048 | 4.83 / 5.12           | 4.83 / 5.10                | 4.89 / 5.08            | 19.27 / 21.97      | 60.83 / 59.61   |

Single compiled runs below L=1024 are launch-bound and move by up to 0.35 ms around
these medians (up to 0.2 ms at L>=1024); single eager runs below L=1024 by up to
2.7 ms (0.1 ms at L>=1024). The medians of the three windows (box / gaussian / none) stay within about 0.1 ms
of each other at every `L`. At L=2048 the compiled `subq_ops` forward is 4.0x / 4.3x
faster than eager `subq_ops` and 12.6x / 11.6x faster than eager `torch` (starting /
ending node).

Memory in MiB, starting node / ending node: `fwd_peak_delta_mb` for the eager
archs (the forward's working set) and `steady_reserved_delta_mb` for the
compiled ones (the cudagraph working set). Runs 2-5 agree exactly; the first run
after a cold Inductor cache reads 0-216 MiB higher for the compiled archs.

| L    | box-subq_ops-eager `fwd_peak_delta_mb` | box-torch-eager `fwd_peak_delta_mb` | box-subq_ops-compiled `steady_reserved_delta_mb` | gaussian-subq_ops-compiled `steady_reserved_delta_mb` |
| ---- | -------------------------------------- | ----------------------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| 128  | 40 / 40                                | 89 / 89                             | 54 / 54                                          | 22 / 22                                               |
| 256  | 160 / 160                              | 353 / 353                           | 114 / 114                                        | 82 / 82                                               |
| 384  | 360 / 360                              | 794 / 794                           | 214 / 214                                        | 182 / 182                                             |
| 512  | 640 / 640                              | 1411 / 1411                         | 354 / 354                                        | 322 / 322                                             |
| 1024 | 2561 / 2561                            | 5640 / 5640                         | 1314 / 1314                                      | 1282 / 1282                                           |
| 2048 | 10244 / 10244                          | 22541 / 22541                       | 5156 / 5156                                      | 5124 / 5124                                           |

One `[B, L, L, C]` bf16 buffer is 1 GiB at L=2048, and the eager `subq_ops` peak
(10244 MiB) holds 10 of them at both nodes: the LayerNorm output (1), the fused
`qkv` projection (3), then at the starting node the `3C`-wide fused short-conv
output (3), `v` and the masked `x1`, `x2` (3), at the ending node the masked `x1`,
`x2`, `v`, `z`, the long-conv output and the gated product (6). Eager `torch`
(22541 MiB, 22 buffers) adds the fp32 FFT buffers.
