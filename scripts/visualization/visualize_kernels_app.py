"""Interactive Kernel + Mask Visualizer (Gradio).

Loads ``.npz`` files produced by ``extract_kernel_data.py`` and provides
interactive side-by-side exploration of learned SIREN kernels, Gaussian masks,
and masked kernels across runs and blocks.

All sampled channels are shown at once in a grid — only the block needs switching.

Usage:
    conda run -n nv-subq python scripts/visualize_kernels_app.py --data-dir tmp/kernel_data
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib


matplotlib.use("Agg")

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_CMAP_KERNEL = "RdBu_r"
_CMAP_MASK = "viridis"
_CMAP_NORM = "inferno"

_CH_GRID_ROWS = 4
_CH_GRID_COLS = 4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_runs(data_dir: str) -> dict[str, Path]:
    d = Path(data_dir)
    return {p.stem: p for p in sorted(d.glob("*.npz"))}


def load_run(path: Path) -> dict:
    data = dict(np.load(str(path), allow_pickle=True))
    block_ids = data["block_ids"]
    blocks = {}
    for bid in block_ids:
        blocks[int(bid)] = {
            "kernel": data[f"block_{bid}_kernel"],
            "mask": data[f"block_{bid}_mask"],
            "masked": data[f"block_{bid}_masked"],
        }
    return {
        "blocks": blocks,
        "block_ids": [int(b) for b in block_ids],
        "grid": data["grid"],
        "channel_indices": data["channel_indices"],
        "kernel_size": tuple(data["kernel_size"]),
        "hidden_dim": int(data["hidden_dim"]),
    }


_cache: dict[str, dict] = {}


def get_run(name: str, runs: dict[str, Path]) -> dict:
    if name not in _cache:
        _cache[name] = load_run(runs[name])
    return _cache[name]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def fig_to_image(fig: plt.Figure, dpi: int = 120) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def _placeholder(text: str = "Select runs and click Render.") -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color="0.5")
    ax.set_axis_off()
    return fig_to_image(fig, dpi=80)


# ---------------------------------------------------------------------------
# Panel: Channel Grid — all channels in 4×4 per run, runs side-by-side
# ---------------------------------------------------------------------------


def plot_channel_grid(
    run_names: list[str],
    runs: dict[str, Path],
    block_idx: int,
    view: str,
) -> Image.Image:
    """4×4 channel grid per run, arranged horizontally.

    view: "kernel", "mask", or "masked"
    """
    if not run_names:
        return _placeholder()

    n_runs = len(run_names)
    total_cols = _CH_GRID_COLS * n_runs
    total_rows = _CH_GRID_ROWS
    cell = 1.3

    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(cell * total_cols + 0.3 * n_runs, cell * total_rows + 1),
        squeeze=False,
    )

    for ri, run_name in enumerate(run_names):
        d = get_run(run_name, runs)
        if block_idx not in d["blocks"]:
            continue

        arr = d["blocks"][block_idx][view]  # [K_h, K_w, n_ch]
        n_ch = arr.shape[-1]
        ch_indices = d["channel_indices"]

        is_mask = view == "mask"
        if is_mask:
            vmin, vmax = 0.0, 1.0
            cmap = _CMAP_MASK
        else:
            abs_max = max(np.abs(arr).max(), 1e-8)
            vmin, vmax = -abs_max, abs_max
            cmap = _CMAP_KERNEL

        for ch_i in range(min(n_ch, _CH_GRID_ROWS * _CH_GRID_COLS)):
            row = ch_i // _CH_GRID_COLS
            col = ri * _CH_GRID_COLS + ch_i % _CH_GRID_COLS
            ax = axes[row, col]

            ax.imshow(arr[:, :, ch_i], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0 and ch_i % _CH_GRID_COLS == _CH_GRID_COLS // 2:
                ax.set_title(run_name, fontsize=9, fontweight="bold")
            if ri == 0 and ch_i % _CH_GRID_COLS == 0:
                ax.set_ylabel(f"ch {ch_indices[ch_i]}", fontsize=7)

    # Turn off unused axes
    for r in range(total_rows):
        for c in range(total_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    view_label = {"kernel": "Raw Kernel", "mask": "Mask", "masked": "Masked Kernel"}[view]
    fig.suptitle(f"Block {block_idx} — {view_label} (all channels)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig_to_image(fig)


# ---------------------------------------------------------------------------
# Panel: Three-View — raw | mask | masked, all channels, per run
# ---------------------------------------------------------------------------


def plot_three_view(run_names: list[str], runs: dict[str, Path], block_idx: int) -> Image.Image:
    """For each run: 3 columns (raw, mask, masked), rows = channels."""
    if not run_names:
        return _placeholder()

    n_runs = len(run_names)
    d0 = get_run(run_names[0], runs)
    n_ch = d0["blocks"][block_idx]["kernel"].shape[-1] if block_idx in d0["blocks"] else 16

    total_cols = 3 * n_runs
    total_rows = n_ch
    cell_w, cell_h = 1.1, 0.9

    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(cell_w * total_cols + 1, cell_h * total_rows + 1.5),
        squeeze=False,
    )

    views = ["kernel", "mask", "masked"]
    view_labels = ["Raw", "Mask", "Masked"]

    for ri, run_name in enumerate(run_names):
        d = get_run(run_name, runs)
        if block_idx not in d["blocks"]:
            continue

        blk = d["blocks"][block_idx]
        ch_indices = d["channel_indices"]

        for vi, view_key in enumerate(views):
            arr = blk[view_key]  # [K_h, K_w, n_ch]
            col_base = ri * 3 + vi

            is_mask = view_key == "mask"
            if is_mask:
                vmin, vmax = 0.0, 1.0
                cmap = _CMAP_MASK
            else:
                abs_max = max(np.abs(arr).max(), 1e-8)
                vmin, vmax = -abs_max, abs_max
                cmap = _CMAP_KERNEL

            for ch_i in range(arr.shape[-1]):
                ax = axes[ch_i, col_base]
                ax.imshow(arr[:, :, ch_i], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])

                if ch_i == 0:
                    label = f"{run_name}\n{view_labels[vi]}" if vi == 1 else view_labels[vi]
                    ax.set_title(label, fontsize=7)

            if vi == 0:
                for ch_i in range(arr.shape[-1]):
                    axes[ch_i, col_base].set_ylabel(f"ch{ch_indices[ch_i]}", fontsize=6, rotation=0, ha="right")

    fig.suptitle(f"Block {block_idx} — Raw / Mask / Masked (all channels)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig_to_image(fig, dpi=100)


# ---------------------------------------------------------------------------
# Panel: Mask Overview — per-block extent across runs
# ---------------------------------------------------------------------------


def plot_mask_overview(run_names: list[str], runs: dict[str, Path]) -> Image.Image:
    if not run_names:
        return _placeholder()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for run_name in run_names:
        d = get_run(run_name, runs)
        block_ids = d["block_ids"]
        boundary_means = []
        center_step_means = []

        for bid in block_ids:
            mask = d["blocks"][bid]["mask"]
            boundary_means.append(mask[0, 0, :].mean())
            ch, cw = mask.shape[0] // 2, mask.shape[1] // 2
            center_step_means.append(mask[ch, min(cw + 1, mask.shape[1] - 1), :].mean())

        axes[0].plot(block_ids, boundary_means, "o-", label=run_name, markersize=4)
        axes[1].plot(block_ids, center_step_means, "o-", label=run_name, markersize=4)

    axes[0].set_title("Mean Mask at Boundary Corner", fontsize=12)
    axes[0].set_xlabel("Block")
    axes[0].set_ylabel("Mask value")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Mean Mask One Step from Center", fontsize=12)
    axes[1].set_xlabel("Block")
    axes[1].set_ylabel("Mask value")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Mask Extent Overview Across Blocks", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig_to_image(fig)


# ---------------------------------------------------------------------------
# Panel: Mask Profiles — 1D center-row for all channels, per run
# ---------------------------------------------------------------------------


def plot_mask_profiles(run_names: list[str], runs: dict[str, Path], block_idx: int) -> Image.Image:
    if not run_names:
        return _placeholder()

    n_runs = len(run_names)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 4), squeeze=False)

    for col, run_name in enumerate(run_names):
        ax = axes[0, col]
        d = get_run(run_name, runs)
        if block_idx not in d["blocks"]:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue

        mask = d["blocks"][block_idx]["mask"]
        center_row = mask.shape[0] // 2
        ch_indices = d["channel_indices"]

        for i in range(mask.shape[-1]):
            profile = mask[center_row, :, i]
            ax.plot(profile, alpha=0.4, linewidth=1, label=f"ch {ch_indices[i]}" if i < 4 else None)

        mean_profile = mask[center_row, :, :].mean(axis=-1)
        ax.plot(mean_profile, color="black", linewidth=2.5, label="mean")

        ax.set_title(run_name, fontsize=11)
        ax.set_xlabel("Grid position")
        ax.set_ylabel("Mask value")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc="lower center", ncol=3)

    fig.suptitle(f"Block {block_idx} — Mask Center-Row Profiles (all channels)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig_to_image(fig)


# ---------------------------------------------------------------------------
# Panel: Kernel Magnitude — channel-mean |kernel| before/after mask
# ---------------------------------------------------------------------------


def plot_kernel_norms(run_names: list[str], runs: dict[str, Path], block_idx: int) -> Image.Image:
    if not run_names:
        return _placeholder()

    n_runs = len(run_names)
    fig, axes = plt.subplots(2, n_runs, figsize=(4 * n_runs, 7), squeeze=False)

    for col, run_name in enumerate(run_names):
        d = get_run(run_name, runs)
        if block_idx not in d["blocks"]:
            continue

        blk = d["blocks"][block_idx]
        raw_norm = np.abs(blk["kernel"]).mean(axis=-1)
        masked_norm = np.abs(blk["masked"]).mean(axis=-1)
        vmax = max(raw_norm.max(), masked_norm.max(), 1e-8)

        im0 = axes[0, col].imshow(raw_norm, cmap=_CMAP_NORM, vmin=0, vmax=vmax, origin="lower", aspect="equal")
        axes[0, col].set_title(run_name, fontsize=10)
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        im1 = axes[1, col].imshow(masked_norm, cmap=_CMAP_NORM, vmin=0, vmax=vmax, origin="lower", aspect="equal")
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        for r in range(2):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    axes[0, 0].set_ylabel("Raw |kernel|", fontsize=10)
    axes[1, 0].set_ylabel("Masked |kernel|", fontsize=10)

    fig.suptitle(f"Block {block_idx} — Channel-Mean Kernel Magnitude", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig_to_image(fig)


# ---------------------------------------------------------------------------
# Panel: All Blocks — masked kernel, channel-mean, one row per block
# ---------------------------------------------------------------------------


def plot_all_blocks(run_names: list[str], runs: dict[str, Path]) -> Image.Image:
    if not run_names:
        return _placeholder()

    d0 = get_run(run_names[0], runs)
    block_ids = d0["block_ids"]
    n_blocks = len(block_ids)
    n_runs = len(run_names)

    fig, axes = plt.subplots(n_blocks, n_runs, figsize=(3.2 * n_runs, 2.2 * n_blocks), squeeze=False)

    for col, run_name in enumerate(run_names):
        d = get_run(run_name, runs)
        for row, bid in enumerate(block_ids):
            ax = axes[row, col]
            if bid not in d["blocks"]:
                ax.axis("off")
                continue

            masked = d["blocks"][bid]["masked"]
            norm = np.abs(masked).mean(axis=-1)
            ax.imshow(norm, cmap=_CMAP_NORM, origin="lower", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(run_name, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Blk {bid}", fontsize=8)

    fig.suptitle("All Blocks — Channel-Mean |Masked Kernel|", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig_to_image(fig, dpi=100)


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------


def build_app(data_dir: str) -> gr.Blocks:
    runs = discover_runs(data_dir)
    if not runs:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    all_names = list(runs.keys())

    # Auto-select init_extent runs
    default_sel = [n for n in all_names if n.startswith("gmask-e")]
    if not default_sel:
        default_sel = all_names[:4]

    d0 = get_run(default_sel[0] if default_sel else all_names[0], runs)
    block_ids = d0["block_ids"]

    with gr.Blocks(title="Kernel & Mask Visualizer") as app:
        gr.Markdown(
            "# SIREN Kernel & Mask Visualizer\n\nAll sampled channels shown at once. Switch blocks to explore depth."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Settings")
                run_selector = gr.CheckboxGroup(
                    choices=all_names,
                    value=default_sel,
                    label="Runs",
                )
                block_slider = gr.Slider(
                    minimum=min(block_ids),
                    maximum=max(block_ids),
                    step=1,
                    value=0,
                    label="Block",
                )
                view_radio = gr.Radio(
                    choices=["masked", "kernel", "mask"],
                    value="masked",
                    label="Channel Grid View",
                )
                render_btn = gr.Button("Render", variant="primary")

            with gr.Column(scale=4):
                with gr.Tab("Channel Grid"):
                    gr.Markdown(
                        "4x4 grid of all sampled channels per run. "
                        "Runs arranged side-by-side. Pick view type in sidebar."
                    )
                    grid_img = gr.Image(label="Channel Grid", type="pil")

                with gr.Tab("Raw / Mask / Masked"):
                    gr.Markdown(
                        "All three views (raw kernel, mask, masked kernel) for every channel. Columns grouped by run."
                    )
                    three_img = gr.Image(label="Three View", type="pil")

                with gr.Tab("Overview"):
                    gr.Markdown(
                        "Per-block mask extent across runs. Left: boundary corner. Right: one step from center."
                    )
                    overview_img = gr.Image(label="Mask Overview", type="pil")

                with gr.Tab("All Blocks"):
                    gr.Markdown("Channel-mean |masked kernel| for all 12 blocks, side-by-side across runs.")
                    all_blocks_img = gr.Image(label="All Blocks", type="pil")

                with gr.Tab("Kernel Magnitude"):
                    gr.Markdown("Channel-mean absolute kernel value before and after masking.")
                    norm_img = gr.Image(label="Kernel Norms", type="pil")

                with gr.Tab("Mask Profiles"):
                    gr.Markdown("1D center-row mask profiles for all channels. Black line = channel mean.")
                    profile_img = gr.Image(label="Mask Profiles", type="pil")

        # --- Callbacks ---
        def refresh_all(selected_runs, block_idx, view):
            block_idx = int(block_idx)
            return (
                plot_channel_grid(selected_runs, runs, block_idx, view),
                plot_three_view(selected_runs, runs, block_idx),
                plot_mask_overview(selected_runs, runs),
                plot_all_blocks(selected_runs, runs),
                plot_kernel_norms(selected_runs, runs, block_idx),
                plot_mask_profiles(selected_runs, runs, block_idx),
            )

        all_outputs = [grid_img, three_img, overview_img, all_blocks_img, norm_img, profile_img]
        all_inputs = [run_selector, block_slider, view_radio]

        render_btn.click(fn=refresh_all, inputs=all_inputs, outputs=all_outputs)
        run_selector.change(fn=refresh_all, inputs=all_inputs, outputs=all_outputs)
        block_slider.change(fn=refresh_all, inputs=all_inputs, outputs=all_outputs)
        view_radio.change(fn=refresh_all, inputs=all_inputs, outputs=all_outputs)

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive Kernel & Mask Visualizer")
    parser.add_argument("--data-dir", type=str, default="tmp/kernel_data")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.data_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
