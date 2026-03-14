"""Interactive FiLM-Hyena kernel explorer (Streamlit).

Loads pre-exported kernel data (.npz) and provides interactive controls
to browse blocks, channels, and images.

Usage:
    streamlit run scripts/app_film_kernels.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent.parent / "outputs" / "film_kernel_viz"
DATASETS = {
    "FiLM bound-all γ_max=4 ω₀=10 ep143": DATA_DIR / "kernel_data_bound_all_ep143.npz",
    "FiLM bound-all γ_max=4 ω₀=10 ep135": DATA_DIR / "kernel_data_bound_all_ep135.npz",
    "FiLM bound-all γ_max=4 ω₀=10 ep95": DATA_DIR / "kernel_data_bound_all_ep95.npz",
    "FiLM + pos-embed (wd=1e-3, ω₀=10, ep71)": DATA_DIR / "kernel_data_posemb_wd1e3_w10.npz",
    "FiLM residual (no pos-embed)": DATA_DIR / "kernel_data_residual.npz",
    "FiLM + pos-embed (wd=1e-3, ω₀=1, ep63)": DATA_DIR / "kernel_data_posemb_wd1e3.npz",
    "FiLM (original/direct)": DATA_DIR / "kernel_data.npz",
    "FiLM + pos_embed warping": DATA_DIR / "kernel_data_posemb.npz",
}


@st.cache_data
def load_data(path: str):
    """Load and unpack the .npz file into a structured dict."""
    raw = np.load(path, allow_pickle=True)

    labels = list(raw["labels"])
    predictions = list(raw["predictions"])
    thumbnails = raw["thumbnails"]  # [n_images, 112, 112, 3]

    static, conditioned, reg_weights = {}, {}, {}
    for b in range(12):
        static[b] = raw[f"static_{b}"]  # [1, H, W, C]
        conditioned[b] = raw[f"cond_{b}"]  # [n_images, H, W, C]
        reg_weights[b] = raw[f"regw_{b}"]  # [13]

    n_images = conditioned[0].shape[0]
    spatial_h, spatial_w = conditioned[0].shape[1], conditioned[0].shape[2]
    n_channels = conditioned[0].shape[3]

    return {
        "labels": labels,
        "predictions": predictions,
        "thumbnails": thumbnails,
        "static": static,
        "conditioned": conditioned,
        "reg_weights": reg_weights,
        "n_images": n_images,
        "n_channels": n_channels,
        "spatial_h": spatial_h,
        "spatial_w": spatial_w,
    }


def make_heatmap(arr_2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Convert a 2D array to an RGB image using RdBu_r colormap."""
    cmap = plt.cm.RdBu_r
    if vmax == vmin:
        normed = np.full_like(arr_2d, 0.5)
    else:
        normed = (arr_2d - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0, 1)
    rgba = cmap(normed)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def get_slice(kernel: np.ndarray, channel: int | None) -> np.ndarray:
    """Extract a 2D slice from a kernel array.

    kernel: [H, W, C] or [1, H, W, C]
    Returns [H, W].
    """
    if kernel.ndim == 4:
        kernel = kernel[0]
    if channel is None:
        return kernel.mean(axis=-1)
    return kernel[:, :, channel]


def main():
    st.set_page_config(page_title="FiLM-Hyena Kernel Explorer", layout="wide")
    st.title("FiLM-Hyena Kernel Explorer")

    available = {name: p for name, p in DATASETS.items() if p.exists()}
    if not available:
        st.error("No data files found. Run `scripts/export_film_kernels.py` first.")
        return

    # ── Sidebar controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        dataset_name = st.selectbox("Model", list(available.keys()))
        data = load_data(str(available[dataset_name]))

        compare_mode = False
        if len(available) > 1:
            compare_mode = st.checkbox("Side-by-side comparison", value=False)
            if compare_mode:
                other_names = [n for n in available if n != dataset_name]
                compare_name = st.selectbox("Compare with", other_names)
                data_cmp = load_data(str(available[compare_name]))

        block = st.slider("Block", 0, 11, 0)

        use_mean = st.checkbox("Channel mean", value=True)
        if use_mean:
            channel = None
        else:
            channel = st.slider("Channel", 0, data["n_channels"] - 1, 0)

        view_mode = st.radio("View mode", ["Conditioned", "Static", "Difference (Cond − Static)"])

        st.markdown("---")
        st.subheader(f"Register Weights (Block {block})")
        fig_reg, ax_reg = plt.subplots(figsize=(4, 2.5))
        w = data["reg_weights"][block]
        colors = ["#d73027" if v == w.max() else "#4575b4" for v in w]
        ax_reg.bar(range(len(w)), w, color=colors)
        ax_reg.set_xlabel("Register")
        ax_reg.set_ylabel("Weight")
        ax_reg.set_xticks(range(len(w)))
        ax_reg.set_ylim(0, min(1.0, w.max() * 1.2))
        fig_reg.tight_layout()
        st.pyplot(fig_reg)
        plt.close(fig_reg)

        st.caption(f"Max: reg {w.argmax()} ({w.max():.3f})")
        entropy = -np.sum(w * np.log(w + 1e-10))
        st.caption(f"Entropy: {entropy:.3f} / {np.log(len(w)):.3f}")

    # ── Channel label ────────────────────────────────────────────────────
    ch_label = "channel mean" if channel is None else f"channel {channel}"
    st.markdown(f"**{dataset_name}** | **Block {block}** | **{ch_label}** | **{view_mode}**")

    # ── Input image strip ────────────────────────────────────────────────
    st.subheader("Input Images")
    cols = st.columns(data["n_images"])
    for i, col in enumerate(cols):
        with col:
            st.image(data["thumbnails"][i], caption=data["labels"][i], width=80)

    # ── Helper to compute heatmaps from a data dict ──────────────────────
    def compute_heatmaps(d):
        s_slice = get_slice(d["static"][block], channel)
        hms = []
        for img_idx in range(d["n_images"]):
            c_slice = get_slice(d["conditioned"][block][img_idx], channel)
            if view_mode == "Conditioned":
                hms.append(c_slice)
            elif view_mode == "Static":
                hms.append(s_slice)
            else:
                hms.append(c_slice - s_slice)
        return hms, s_slice

    def compute_color_range(hms):
        all_vals = np.concatenate([h.ravel() for h in hms])
        am = max(abs(all_vals.min()), abs(all_vals.max()), 1e-8)
        return -am, am

    def render_kernel_row(d, hms, vmin, vmax, label, s_slice):
        """Render a single model's kernel heatmaps."""
        if view_mode != "Static":
            s_abs = max(abs(s_slice.min()), abs(s_slice.max()), 1e-8)
            s_img = make_heatmap(s_slice, -s_abs, s_abs)
            st.image(s_img, width=200, caption=f"Static kernel (max={s_abs:.5f})")

        if view_mode == "Static":
            s_abs = max(abs(s_slice.min()), abs(s_slice.max()), 1e-8)
            s_img = make_heatmap(s_slice, -s_abs, s_abs)
            st.image(s_img, width=300, caption=f"max={s_abs:.5f}")
        else:
            st.caption(f"Color range: [{vmin:.5f}, {vmax:.5f}]")
            imgs_per_row = min(10, d["n_images"])
            for row_start in range(0, d["n_images"], imgs_per_row):
                row_end = min(row_start + imgs_per_row, d["n_images"])
                cols = st.columns(row_end - row_start)
                for i, col in enumerate(cols):
                    idx = row_start + i
                    img = make_heatmap(hms[idx], vmin, vmax)
                    with col:
                        st.image(img, caption=d["labels"][idx], use_container_width=True)

    # ── Compute heatmaps ─────────────────────────────────────────────────
    heatmaps, static_slice = compute_heatmaps(data)
    vmin, vmax = compute_color_range(heatmaps)

    if compare_mode:
        heatmaps_cmp, static_slice_cmp = compute_heatmaps(data_cmp)
        # Use shared color range for fair comparison
        vmin_cmp, vmax_cmp = compute_color_range(heatmaps_cmp)
        shared_abs = max(abs(vmin), abs(vmax), abs(vmin_cmp), abs(vmax_cmp))
        vmin = vmin_cmp = -shared_abs
        vmax = vmax_cmp = shared_abs

        title = (
            "Conditioned Kernels"
            if view_mode == "Conditioned"
            else ("Static Kernels" if view_mode == "Static" else "FiLM Effect (Cond − Static)")
        )
        st.subheader(title)

        st.markdown(f"#### {dataset_name}")
        render_kernel_row(data, heatmaps, vmin, vmax, dataset_name, static_slice)

        st.markdown("---")
        st.markdown(f"#### {compare_name}")
        render_kernel_row(data_cmp, heatmaps_cmp, vmin_cmp, vmax_cmp, compare_name, static_slice_cmp)
    else:
        if view_mode != "Static":
            st.subheader("Static Kernel (no FiLM)")
            static_abs = max(abs(static_slice.min()), abs(static_slice.max()), 1e-8)
            static_img = make_heatmap(static_slice, -static_abs, static_abs)
            st.image(static_img, width=200, caption=f"max={static_abs:.5f}")

        if view_mode == "Static":
            st.subheader("Static Kernel (same for all images)")
            static_abs = max(abs(static_slice.min()), abs(static_slice.max()), 1e-8)
            static_img = make_heatmap(static_slice, -static_abs, static_abs)
            st.image(static_img, width=300, caption=f"max={static_abs:.5f}")
        else:
            st.subheader(f"{'Conditioned Kernels' if view_mode == 'Conditioned' else 'FiLM Effect (Cond − Static)'}")
            st.caption(f"Color range: [{vmin:.5f}, {vmax:.5f}]")
            imgs_per_row = min(10, data["n_images"])
            for row_start in range(0, data["n_images"], imgs_per_row):
                row_end = min(row_start + imgs_per_row, data["n_images"])
                cols = st.columns(row_end - row_start)
                for i, col in enumerate(cols):
                    idx = row_start + i
                    img = make_heatmap(heatmaps[idx], vmin, vmax)
                    with col:
                        st.image(img, caption=data["labels"][idx], use_container_width=True)

    # ── Stats ────────────────────────────────────────────────────────────
    with st.expander("Channel statistics"):
        cond_full = data["conditioned"][block]  # [n_images, H, W, C]
        static_full = data["static"][block][0]  # [H, W, C]

        diff_all = cond_full - static_full[None, ...]  # [n_images, H, W, C]
        var_across_images = diff_all.var(axis=0).mean(axis=(0, 1))  # [C]

        fig_var, ax_var = plt.subplots(figsize=(10, 3))
        ax_var.bar(range(len(var_across_images)), var_across_images, width=1.0, color="#4575b4")
        ax_var.set_xlabel("Channel")
        ax_var.set_ylabel("FiLM effect variance")
        ax_var.set_title(f"Block {block}: Per-channel variance of FiLM effect across {data['n_images']} images")
        fig_var.tight_layout()
        st.pyplot(fig_var)
        plt.close(fig_var)

        top_k = 10
        top_channels = var_across_images.argsort()[::-1][:top_k]
        st.markdown(f"**Top-{top_k} most varying channels:** {', '.join(str(c) for c in top_channels)}")


if __name__ == "__main__":
    main()
