# Op-level benchmarks

Microbenchmarks for the convolution and MLP primitives that ViT-5 and
Hyena models compose.  Each script targets a single op, varies a few
realistic workload shapes, and prints latency / memory / throughput
side-by-side for the variants we care about.

## Scripts

### `bench_fftconv2d.py`

Latency comparison between the reference
`torch.fft`-based `fftconv2d_fp32_bhl` and the CUDA-accelerated
`subquadratic_ops_torch.fft_conv2d`.  Covers eager, `torch.compile`
(default), and `max-autotune` variants across realistic ViT/Hyena
shapes.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/ops/bench_fftconv2d.py
```

Output: stdout summary table.  Targets H100 SXM 80GB (numbers were
collected there); any Ampere+ GPU will work for relative comparisons.

### `bench_mlp.py`

Forward + backward correctness check and microbenchmark for the
`nvsubquadratic.modules.mlp.MLP` block (`torch` vs QuACK fused
kernel).  Uses the well/euler training-config shapes
(`dim=384`, `activation="glu"`, `batch=24`, `seq_len=1024`, bf16).

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/ops/bench_mlp.py
```

Output: stdout.  Hopper/Blackwell only for the QuACK path (H100 / B200).

### `bench_subquadratic_fftconv.py`

Quick correctness + forward-speed check for the
`subquadratic_ops_torch.fft_causal_conv1d` CUDA kernel.  Useful as a
sanity gate after kernel changes.

```bash
PYTHONPATH=. conda run -n nv-subq python benchmarks/ops/bench_subquadratic_fftconv.py --device cuda
```

Output: stdout.
