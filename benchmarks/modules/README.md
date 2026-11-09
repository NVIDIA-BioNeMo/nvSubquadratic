# Module-level benchmarks

## TriHyena (`bench_tri_hyena.py`)

Times a pre-`LayerNorm` followed by `QKVSequenceMixer(TriangleHyena)` (`PairMixer`)
on a pair tensor `[B, L, L, C]`
(default `B=1`, `C=128`, `L` in 128, 256, 384, 512, 1024, 2048, bf16, box
window `max_offset=256`, `right_padding=True` (fixed) with the last
`round(0.125·L)` tokens of every sample padded (`--right-pad-frac 0.125`), i.e.
about 23% of pair positions masked) with CUDA events, an L2 flush before every
timed region and a warm-up of 3 calls, then until 1 s of wall time (compile and
capture included) has passed. Arch names are
`<window>-<torch|subq_ops>-<eager|compiled>`; `compiled` applies
`torch.compile(mode="reduce-overhead")` from the outside. `--mode train`
(default) reports forward, backward and full-step time plus peak memory;
`--mode infer` reports the forward under `torch.inference_mode()`. Results are
written incrementally to a CSV. Pin the GPU with `CUDA_VISIBLE_DEVICES=<uuid>`
on multi-GPU hosts.

```bash
# window x backend x eager/compiled, starting node
PYTHONPATH=. python benchmarks/modules/bench_tri_hyena.py --mode infer --csv tri_hyena_start.csv

# ending-node update
PYTHONPATH=. python benchmarks/modules/bench_tri_hyena.py --mode infer --end-node --csv tri_hyena_end.csv
```

### Memory columns

All `_mb` columns are MiB and come from the PyTorch caching allocator with the
persistent L2 flush buffer subtracted (its 2 MiB segment rounding for the
reserved figures). Compile
state and cached segments are reset before every (arch, L) measurement
(`torch._dynamo.reset()` is the only way to return a cudagraph private pool, so
without it one arch's pool would sit in the next arch's baseline and reserved
peak); this does not fully isolate archs sharing a process (see the memory
results). Peak columns are the max over the `--n-bench` iterations.

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

In `--mode infer` a `reduce-overhead` replay allocates nothing, so `fwd_peak_delta_mb` reads 0
and `fwd_peak_mb` is the resident baseline; the graph's static buffers show up in
`setup_peak_*` and `steady_reserved_delta_mb` (the cudagraph private pool, i.e. the
steady-state working set) instead, while for eager archs
`steady_reserved_delta_mb` is 0 (`torch`) or a constant 32 MiB (`subq_ops`). To compare peak memory across archs use
`fwd_peak_delta_mb` / `step_peak_delta_mb` for the eager archs,
`steady_reserved_delta_mb` for the compiled ones. In `--mode train` the warm-up
runs full steps, so `steady_reserved_delta_mb` also contains the parameter
gradients left by the last warm-up step, and for compiled archs `act_mb` and
`step_peak_*` leave out the activations held in the cudagraph pool (L=512 starting
node: `act_mb` 194 compiled vs 1092 eager `subq_ops`).

### Results (GB300 CC 10.3, 1300 W cap, bf16, B=1, C=128, box window 256, `--mode infer`, 2026-11-09)

torch 2.13.0+cu132, subquadratic-ops-torch-cu13 0.3.1, cuDNN 9.27. Measured on
`TriangleHyena` with `right_padding=True` and a single padding fill of the long-conv
input `x1 * v`; timings are medians of 5 runs of each command above, compiled memory
one run per `--archs <arch>` (starting and ending node). At every `L`
in the tables below the `subq_ops` long conv dispatches to `explicit_conv1d` (probed in eager mode; fp32 would use
`fft_conv1d`); the `torch` archs use `torch.fft`. Eager `subq_ops` runs the short
conv with `causal_conv1d`, eager `torch` with `F.conv1d`, compiled archs with Inductor.

Inference forward time in ms, starting node / ending node (`--end-node`), median of
five runs (each the mean of 10 calls):

| L    | box-subq_ops-compiled | gaussian-subq_ops-compiled | none-subq_ops-compiled | box-subq_ops-eager | box-torch-eager |
| ---- | --------------------- | -------------------------- | ---------------------- | ------------------ | --------------- |
| 128  | 0.32 / 0.29           | 0.26 / 0.37                | 0.33 / 0.39            | 1.58 / 1.94        | 1.32 / 1.91     |
| 256  | 0.40 / 0.37           | 0.36 / 0.36                | 0.37 / 0.32            | 1.51 / 1.81        | 1.46 / 1.81     |
| 384  | 0.43 / 0.42           | 0.45 / 0.47                | 0.45 / 0.44            | 1.57 / 1.90        | 2.29 / 2.14     |
| 512  | 0.54 / 0.61           | 0.56 / 0.62                | 0.50 / 0.57            | 1.55 / 2.02        | 3.62 / 3.30     |
| 1024 | 1.26 / 1.41           | 1.30 / 1.40                | 1.27 / 1.40            | 4.40 / 5.10        | 14.24 / 13.31   |
| 2048 | 4.87 / 5.10           | 4.89 / 5.10                | 4.83 / 5.11            | 17.82 / 20.51      | 58.45 / 56.48   |

Single compiled runs below L=1024 are launch-bound and move by up to 0.43 ms around
these medians, more than 100% at L=128 (ending-node gaussian 0.80 vs median 0.37),
and by up to 0.23 ms at L>=1024; single eager runs below L=1024 by up to
2.7 ms (0.06 ms at L>=1024). The medians of the three windows (box / gaussian / none) stay within about 0.1 ms
of each other at every `L`. At L=2048 the compiled `subq_ops` forward is 3.7x / 4.0x
faster than eager `subq_ops` and 12.0x / 11.1x faster than eager `torch` (starting /
ending node).

Memory in MiB, starting node / ending node: `fwd_peak_delta_mb` for the eager
archs (the forward's working set) and `steady_reserved_delta_mb` for the
compiled ones (the cudagraph working set). The eager columns agree exactly across
all five runs; the first run after a cold Inductor cache reads up to 116 MiB higher
for the compiled archs (0 for box at L=2048). Compiled columns are a single run
(not a median) of each arch in its own process (`--archs <arch>`): in the five-arch
sweep gaussian reads 32 MiB lower at every `L` despite the per-measurement reset
(5124 vs 5156 at L=2048), while box and none match. `none-subq_ops-compiled`
reads the same as box (54 / 114 / 214 / 354 / 1314 / 5156).

| L    | box-subq_ops-eager `fwd_peak_delta_mb` | box-torch-eager `fwd_peak_delta_mb` | box-subq_ops-compiled `steady_reserved_delta_mb` | gaussian-subq_ops-compiled `steady_reserved_delta_mb` |
| ---- | -------------------------------------- | ----------------------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| 128  | 40 / 36                                | 85 / 85                             | 54 / 54                                          | 54 / 54                                               |
| 256  | 160 / 144                              | 337 / 337                           | 114 / 114                                        | 114 / 114                                             |
| 384  | 360 / 324                              | 758 / 758                           | 214 / 214                                        | 214 / 214                                             |
| 512  | 640 / 576                              | 1347 / 1347                         | 354 / 354                                        | 354 / 354                                             |
| 1024 | 2561 / 2305                            | 5384 / 5384                         | 1314 / 1314                                      | 1314 / 1314                                           |
| 2048 | 10244 / 9220                           | 21518 / 21517                       | 5156 / 5156                                      | 5156 / 5156                                           |

One `[B, L, L, C]` bf16 buffer is 1 GiB at L=2048. The eager `subq_ops` peak
holds 10 of them at the starting node (10244 MiB) and 9 at the ending node
(9220 MiB): the LayerNorm output (1), the fused `qkv` projection (3), then at the
starting node the `3C`-wide fused short-conv output (3), `v`, `x1 * v` and its
masked copy (3), at the ending node `v`, the masked `x2`, the transposed `z`, the
long-conv output and the gated product (5). Eager `torch` (21518 / 21517 MiB, 21
buffers) peaks inside the long conv, when the starting node's fused short-conv output is
already freed: the LayerNorm output (1), `qkv` (3), `v`, the masked `x2` and the
transposed `z` (3), plus 14 in `torch.fft`: the fp32 input (2), its padded
spectrum (4) and two padded inverse-FFT buffers (8).
