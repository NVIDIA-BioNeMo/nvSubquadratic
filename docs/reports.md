# Reports

Long-form technical reports backed by code, figures, and reproducible
scripts.  Each topic lives in its own subfolder under
[`reports/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/tree/main/reports)
with one `REPORT.md` plus the supporting `.py` and `.png` files it
cites.  See the
[source README](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/reports/README.md)
for the regeneration conventions (snake_case topic names, scripts take
`--output-dir`, runs from repo root with `PYTHONPATH=.`).

## Current topics

| Topic                                                                                                                                                            | Summary                                                                                                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`ckconv_block_diagonal_kernel/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/reports/ckconv_block_diagonal_kernel/REPORT.md)             | Block-diagonal multi-ω₀ SIREN kernel + block-aligned Gaussian mask for ViT-5 hybrid Hyena.  Resolution scaling rule (`ω₀ ← m·ω₀`) verified across 1×/2×/4× grids. |
| [`siren_omega0_dimensional_scaling/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/reports/siren_omega0_dimensional_scaling/REPORT.md)     | SIREN ω₀ dimensional scaling rule and supporting figures.                                                                                                         |
| [`spatial_recall/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/reports/spatial_recall/REPORT.md)                                         | Qualitative target-vs-prediction snapshots for the 1D/2D/3D EMNIST spatial-recall task suite (simple copy, mask selection, color selection, color conditioning).  |
| [`vit5_imagenet_dataloader_profiling/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/reports/vit5_imagenet_dataloader_profiling/REPORT.md) | Feb-2026 investigation that diagnosed the CPU-decode bottleneck on ViT-5-Small ImageNet and motivated the move to the DALI-fused dataloader.                      |

## Adding a new report

1. `mkdir reports/<topic>/` (descriptive snake_case name).
1. Drop a `REPORT.md` plus any scripts and figures inside; keep
   image links relative to the topic folder.
1. Add a row to the index table in
   [`reports/README.md`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/reports/README.md)
   and mirror it here.
1. Re-run every script once before committing so the figures match
   the reported numbers.
