# Benchmarks

Throughput numbers and FLOP scaling.  The
tables below are included verbatim from the
[`benchmarks/README.md`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/benchmarks/README.md)
single-source — edits should land there, not here.

## FLOP scaling

![FLOP scaling for Hyena / attention / CKConv mixers](_static/flop_scaling.png)

See [`benchmarks/compare_flops.py`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/benchmarks/compare_flops.py)
for the script that produced the plot.

## ViT-5-Small throughput

```{include} ../benchmarks/README.md
---
start-after: '# ViT-5-Small Throughput Benchmarks'
---
```
