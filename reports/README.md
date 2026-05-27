# Reports

Long-form technical reports backed by code, figures, and reproducible
scripts. Each topic gets its own subfolder containing **one `REPORT.md`**
plus the supporting `.py` scripts and `.png` figures it cites.

## Conventions

- One subfolder per topic. Use a descriptive `snake_case` name
  (`ckconv_block_diagonal_kernel`, `well_evaluation_pipeline`, …).

- Each topic folder is **self-contained**: scripts live with the figures and
  the report so the whole thing can be re-run from a checkout of this branch.
  Markdown image links are relative — copy or move the folder freely.

- Scripts take `--output-dir` (defaulting to their own folder) so re-running
  them refreshes the figures next to the report.

- Run scripts from the repo root with `PYTHONPATH=.` so the
  `nvsubquadratic.*` imports resolve, e.g.

  ```bash
  PYTHONPATH=. conda run -n nv-subq python \
      reports/<topic>/<script>.py
  ```

- Use the `nv-subq` conda environment (CPU-friendly variants of GPU-only
  tests are baked into the smoke-test scripts when available; otherwise run
  via SLURM).

## Index

| topic                                                                                 | summary                                                                                                                                                                                                                        | added   |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| [`ckconv_block_diagonal_kernel/`](ckconv_block_diagonal_kernel/REPORT.md)             | Block-diagonal multi-ω₀ SIREN kernel + block-aligned Gaussian mask for ViT-5 hybrid Hyena. Default selection, **resolution scaling rule** (`ω₀ ← m·ω₀`) verified across 1×/2×/4× grids, prod-vs-prototype bit-identical check. | 2026-04 |
| [`spatial_recall/`](spatial_recall/REPORT.md)                                         | Qualitative target-vs-prediction snapshots for the 1D/2D/3D EMNIST spatial-recall task suite (simple copy, mask selection, color selection, color conditioning).                                                               | 2026-05 |
| [`vit5_imagenet_dataloader_profiling/`](vit5_imagenet_dataloader_profiling/REPORT.md) | Feb-2026 investigation that diagnosed the CPU-decode bottleneck on ViT-5-Small ImageNet and motivated the move to the DALI-fused dataloader.                                                                                   | 2026-02 |

## Adding a new report

1. `mkdir reports/<topic>/`.
1. Drop a `REPORT.md` plus any scripts and figures inside; keep image links
   relative to the topic folder.
1. Add a row to the index table above.
1. Re-run every script once before committing so the figures committed to
   the repo match the reported numbers.
