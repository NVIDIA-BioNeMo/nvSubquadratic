"""Streamlit app to visualize extracted SIREN kernels, masks, and masked kernels.

Usage:
    conda run -n nv-subq streamlit run scripts/visualization/visualize_kernels.py -- --data-dir tmp/kernel_data

Expects JSON files produced by ``scripts/data/extract_kernel_data.py`` in --data-dir.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_run(path: str) -> dict:
    """Load a single run JSON file."""
    with open(path) as f:
        return json.load(f)


def discover_runs(data_dir: str) -> dict[str, Path]:
    """Find all .json files in data_dir, return {display_name: Path}."""
    d = Path(data_dir)
    runs = {}
    for p in sorted(d.glob("*.json")):
        runs[p.stem] = p
    return runs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def make_heatmap(arr: np.ndarray, title: str, vmin: float | None = None, vmax: float | None = None):
    """Create a matplotlib figure with a single heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_channel_grid(block_data: dict, view: str, ncols: int = 4, shared_scale: bool = True):
    """Create a grid of heatmaps for all sampled channels in a block."""
    import matplotlib.pyplot as plt

    channels = block_data["channels"]
    n = len(channels)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False)

    all_data = [np.array(ch[view]) for ch in channels]

    if shared_scale:
        vmin = min(d.min() for d in all_data)
        vmax = max(d.max() for d in all_data)
        if view == "mask":
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = None, None

    cmap = "viridis" if view == "mask" else "RdBu_r"

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx < n:
            ch = channels[idx]
            arr = all_data[idx]
            im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(f"ch {ch['ch_idx']}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    kernel_size = block_data["kernel_size"]
    fig.suptitle(
        f"Block {block_data['block_id']} — {view} ({kernel_size[0]}x{kernel_size[1]})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def make_mask_profile(block_data: dict):
    """1D radial profile of mask values at the center row for all sampled channels."""
    import matplotlib.pyplot as plt

    channels = block_data["channels"]
    kernel_size = block_data["kernel_size"]
    center_row = kernel_size[0] // 2

    fig, ax = plt.subplots(figsize=(8, 4))
    for ch in channels:
        mask = np.array(ch["mask"])
        profile = mask[center_row, :]
        ax.plot(profile, label=f"ch {ch['ch_idx']}", alpha=0.7)

    ax.set_xlabel("Grid position (pixels)")
    ax.set_ylabel("Mask value")
    ax.set_title(f"Block {block_data['block_id']} — Mask center-row profile")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(ncol=4, fontsize=7, loc="lower center")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_run_summary(data: dict):
    """Per-block mask statistics summary figure."""
    import matplotlib.pyplot as plt

    blocks = data["blocks"]
    block_ids = [b["block_id"] for b in blocks]
    n_blocks = len(blocks)

    # For each block, get the range of mask values across channels at the boundary
    min_at_boundary = []
    max_at_boundary = []
    min_at_center_step = []
    max_at_center_step = []

    for b in blocks:
        ks = b["kernel_size"]
        center = ks[0] // 2
        # Boundary: corners of the 2D grid
        boundary_vals = []
        step_vals = []
        for ch in b["channels"]:
            mask = np.array(ch["mask"])
            boundary_vals.append(mask[0, 0])
            if center + 1 < ks[0]:
                step_vals.append(mask[center, center + 1])
            else:
                step_vals.append(mask[center, center])

        min_at_boundary.append(min(boundary_vals))
        max_at_boundary.append(max(boundary_vals))
        min_at_center_step.append(min(step_vals))
        max_at_center_step.append(max(step_vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(n_blocks)
    width = 0.35

    ax1.bar(x - width / 2, min_at_boundary, width, label="Min (narrowest)", color="steelblue", alpha=0.8)
    ax1.bar(x + width / 2, max_at_boundary, width, label="Max (widest)", color="coral", alpha=0.8)
    ax1.set_xlabel("Block")
    ax1.set_ylabel("Mask value at boundary")
    ax1.set_title("Mask at boundary corner")
    ax1.set_xticks(x)
    ax1.set_xticklabels(block_ids)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x - width / 2, min_at_center_step, width, label="Min (narrowest)", color="steelblue", alpha=0.8)
    ax2.bar(x + width / 2, max_at_center_step, width, label="Max (widest)", color="coral", alpha=0.8)
    ax2.set_xlabel("Block")
    ax2.set_ylabel("Mask value at first step")
    ax2.set_title("Mask at first step from center")
    ax2.set_xticks(x)
    ax2.set_xticklabels(block_ids)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Mask coverage summary across blocks", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="Kernel Visualizer", layout="wide")
    st.title("SIREN Kernel + Mask Visualizer")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="tmp/kernel_data")
    args, _ = parser.parse_known_args()

    runs = discover_runs(args.data_dir)
    if not runs:
        st.error(f"No JSON files found in `{args.data_dir}`. Run extract_kernel_data.py first.")
        return

    # --- Sidebar ---
    st.sidebar.header("Settings")
    selected_runs = st.sidebar.multiselect("Runs", list(runs.keys()), default=list(runs.keys()))
    if not selected_runs:
        st.warning("Select at least one run.")
        return

    view = st.sidebar.selectbox("View", ["masked_kernel", "mask", "kernel"])
    shared_scale = st.sidebar.checkbox("Shared color scale", value=True)

    # Load first run for block selection
    first_data = load_run(str(runs[selected_runs[0]]))
    block_ids = [b["block_id"] for b in first_data["blocks"]]
    selected_block = st.sidebar.selectbox("Block", block_ids, format_func=lambda x: f"Block {x}")

    # --- Run summary tab and detail tab ---
    tab_summary, tab_detail, tab_compare, tab_profiles = st.tabs(
        ["Summary", "Channel Detail", "Cross-Run Compare", "Mask Profiles"]
    )

    # Summary tab
    with tab_summary:
        for run_name in selected_runs:
            data = load_run(str(runs[run_name]))
            st.subheader(run_name)
            fig = make_run_summary(data)
            st.pyplot(fig)

    # Detail tab
    with tab_detail:
        for run_name in selected_runs:
            data = load_run(str(runs[run_name]))
            block_data = next((b for b in data["blocks"] if b["block_id"] == selected_block), None)
            if block_data is None:
                st.warning(f"Block {selected_block} not found in {run_name}")
                continue
            st.subheader(f"{run_name} — Block {selected_block}")
            fig = make_channel_grid(block_data, view, shared_scale=shared_scale)
            st.pyplot(fig)

    # Compare tab: same block, same channel across runs
    with tab_compare:
        if len(selected_runs) < 2:
            st.info("Select 2+ runs to compare.")
        else:
            first_block = next((b for b in first_data["blocks"] if b["block_id"] == selected_block), None)
            if first_block:
                ch_indices = [ch["ch_idx"] for ch in first_block["channels"]]
                sel_ch = st.selectbox("Channel", ch_indices, format_func=lambda x: f"Channel {x}")

                cols = st.columns(len(selected_runs))
                for i, run_name in enumerate(selected_runs):
                    data = load_run(str(runs[run_name]))
                    block_data = next((b for b in data["blocks"] if b["block_id"] == selected_block), None)
                    if block_data is None:
                        continue
                    ch_data = next((c for c in block_data["channels"] if c["ch_idx"] == sel_ch), None)
                    if ch_data is None:
                        continue
                    arr = np.array(ch_data[view])
                    with cols[i]:
                        vmin, vmax = (0.0, 1.0) if view == "mask" else (None, None)
                        fig = make_heatmap(arr, f"{run_name}", vmin=vmin, vmax=vmax)
                        st.pyplot(fig)

    # Profiles tab
    with tab_profiles:
        for run_name in selected_runs:
            data = load_run(str(runs[run_name]))
            block_data = next((b for b in data["blocks"] if b["block_id"] == selected_block), None)
            if block_data is None:
                continue
            st.subheader(f"{run_name} — Block {selected_block}")
            fig = make_mask_profile(block_data)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
