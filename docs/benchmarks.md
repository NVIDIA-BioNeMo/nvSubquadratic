# Benchmarks

Throughput numbers, FLOP scaling, and FP16 op-level results.  The
tables below are included verbatim from the
[`benchmarks/README.md`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/benchmarks/README.md)
single-source — edits should land there, not here.

## FLOP scaling

![FLOP scaling for Hyena / attention / CKConv mixers](_static/flop_scaling.png)

See [`benchmarks/compare_flops.py`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/benchmarks/compare_flops.py)
for the script that produced the plot.

## ViT-5-Small throughput

```{include} ../benchmarks/README.md
---
start-after: '# ViT-5-Small Throughput Benchmarks'
---
```

## Op-level results

- [FP16 FFT convolution results](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/benchmarks/ops/FP16_FFTCONV_RESULTS.md)
  — accuracy and throughput of the FP16 path against the FP32 reference,
  with the dual-mean-centering derivation summarised in
  [FP16 Circular FFT Convolution: Derivation](ops/FP16_FFTCONV_DERIVATION.md).
