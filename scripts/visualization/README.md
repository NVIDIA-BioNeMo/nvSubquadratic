# Visualization scripts

## Kernel viewers

Two viewers exist for inspecting SIREN kernels, Gaussian masks, and
masked kernels across runs and blocks.  They consume different file
formats and have different feature sets — pick whichever matches your
data and workflow.

### `visualize_kernels.py` (Streamlit) — consumes `.json`

- Sidebar with multi-select for runs, view selector, shared-scale toggle.
- Four tabs: Summary, Detail, Compare across runs, Profiles.
- Good for ad-hoc poking around during a kernel-debug session.

```bash
conda run -n nv-subq streamlit run scripts/visualization/visualize_kernels.py -- --data-dir tmp/kernel_data
```

### `visualize_kernels_app.py` (Gradio) — consumes `.npz`

- Block slider + three tabs: Channel Grid, Raw/Mask/Masked, Overview.
- Headless-friendly (sets `matplotlib.use("Agg")`, configurable
  `MPLCONFIGDIR`) — works in containers without `DISPLAY`.
- PIL image output, easier to drop into a `REPORT.md` or share via
  Gradio's `share=True` URL.

```bash
conda run -n nv-subq python scripts/visualization/visualize_kernels_app.py --data-dir tmp/kernel_data
```

### Producing data for the viewers

Both viewers consume output of
[`scripts/data/extract_kernel_data.py`](../data/extract_kernel_data.py)
(Streamlit reads JSON, Gradio reads NPZ — the extractor produces both
formats).

## Other tools

- `visualize_repeated_aug.py` — DALI repeated-augmentation sanity
  check.  Saves `outputs/repeated_aug_grid.png` with a grid of
  augmented views per source image.
- `visualize_patch_size_throughput.py` — patch-size sweep throughput
  plot (consumes `.jsonl` output of
  [`scripts/benchmark_patch_size_2d.py`](../benchmark_patch_size_2d.py)).
