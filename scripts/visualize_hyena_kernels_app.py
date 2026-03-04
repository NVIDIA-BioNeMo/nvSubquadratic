"""Interactive Hyena Kernel Visualization App (Gradio).

Provides an interactive UI for exploring learned Hyena kernels from trained
ViT-5 + Multi-Head Hyena models. Supports switching between layers, heads,
and input images on-the-fly.

Usage:
    PYTHONPATH=. python scripts/visualize_hyena_kernels_app.py --port 7860
    PYTHONPATH=. python scripts/visualize_hyena_kernels_app.py --share  # public tunnel
"""

from __future__ import annotations

import argparse
import io
import os
import sys

# Redirect caches before any heavy imports.
_CACHE_ROOT = os.environ.get("NVSQ_CACHE_ROOT", "/ivi/zfs/s0/original_homes/dknigge/.cache")
os.environ.setdefault("MPLCONFIGDIR", f"{_CACHE_ROOT}/matplotlib")
os.environ.setdefault("WANDB_CACHE_DIR", f"{_CACHE_ROOT}/wandb")
os.environ.setdefault("WANDB_DATA_DIR", f"{_CACHE_ROOT}/wandb")

import matplotlib

matplotlib.use("Agg")  # Thread-safe, non-interactive backend.

import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image

# Ensure the project root is on the path so we can import from scripts/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.visualize_hyena_kernels import (
    extract_all_kernels,
    forward_with_hooks,
    load_and_preprocess_image,
    load_model,
    plot_activation_maps,
    plot_activation_on_image,
    plot_channel_correlation,
    plot_gaussian_masks,
    plot_kernel_norm_heatmaps,
    plot_kernel_on_image,
    plot_kernel_pca,
    plot_kernel_slices,
    plot_mixing_svd_map,
    plot_raw_vs_masked,
    plot_spectral_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fig_to_image(fig: plt.Figure, dpi: int = 120) -> Image.Image:
    """Render a matplotlib Figure to a PIL Image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def _placeholder(text: str = "Load a model first.") -> Image.Image:
    """Return a small placeholder image with the given text."""
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color="0.5")
    ax.set_axis_off()
    return fig_to_image(fig, dpi=80)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def on_load_model(run_path: str, alias: str, device: str):
    """Load model + extract kernels. Returns states and updated slider ranges."""
    try:
        model = load_model(None, run_path, alias, device)
    except Exception as e:
        return (
            None,
            None,
            gr.Slider(value=0),
            gr.Slider(value=0),
            f"Error loading model: {e}",
        )

    kernels = extract_all_kernels(model)
    num_layers = len(model.blocks)
    num_heads = kernels[0]["masked_kernel"].shape[0]
    status = f"Loaded: {num_layers} layers, {num_heads} heads/layer, device={device}"

    return (
        model,
        kernels,
        gr.Slider(minimum=0, maximum=num_layers - 1, step=1, value=0),
        gr.Slider(minimum=0, maximum=num_heads - 1, step=1, value=0),
        status,
    )


def on_render_overview(kernels):
    """Full kernel norm grid across all layers and heads."""
    if kernels is None:
        return _placeholder()
    num_layers = len(kernels)
    fig = plot_kernel_norm_heatmaps(kernels, list(range(num_layers)), None)
    return fig_to_image(fig)


def on_render_receptive_field(kernels, layer_idx):
    """Receptive field panels: Raw vs Masked + Gaussian mask for ALL heads at the selected layer."""
    if kernels is None:
        ph = _placeholder()
        return ph, ph
    layer_idx = int(layer_idx)
    layers = [layer_idx]

    fig_rm = plot_raw_vs_masked(kernels, layers, None)
    fig_gauss = plot_gaussian_masks(kernels, layers, None)

    return fig_to_image(fig_rm), fig_to_image(fig_gauss)


def on_render_kernel_structure(kernels, layer_idx, head_idx, all_heads):
    """Kernel structure panels for a single layer and head (or all heads)."""
    if kernels is None:
        ph = _placeholder()
        return ph, ph, ph, ph, ph
    layer_idx = int(layer_idx)
    layers = [layer_idx]
    heads = None if all_heads else [int(head_idx)]

    fig_pca = plot_kernel_pca(kernels, layers, heads)
    fig_spec = plot_spectral_analysis(kernels, layers, heads)
    fig_corr = plot_channel_correlation(kernels, layers, heads)
    fig_slices = plot_kernel_slices(kernels, layers, heads)
    fig_svd = plot_mixing_svd_map(kernels, layers, heads)

    return (
        fig_to_image(fig_pca),
        fig_to_image(fig_spec),
        fig_to_image(fig_corr),
        fig_to_image(fig_slices),
        fig_to_image(fig_svd),
    )


def on_image_upload(image_path, model, device):
    """Load and preprocess image, run forward pass with hooks."""
    if image_path is None or model is None:
        return None, None, "No image or model."
    try:
        img_tensor = load_and_preprocess_image(image_path)
        activations = forward_with_hooks(model, img_tensor, device)
        return img_tensor, activations, "Forward pass complete."
    except Exception as e:
        return None, None, f"Error: {e}"


def on_render_image_panels(
    image_tensor, activations, kernels, model, layer_idx, head_idx
):
    """Image-conditioned panels for the current image, layer, and head."""
    ph = _placeholder("Upload an image first.")
    if image_tensor is None or activations is None or kernels is None:
        return ph, ph, ph
    layer_idx = int(layer_idx)
    head_idx = int(head_idx)
    layers = [layer_idx]
    heads = [head_idx]

    fig_act = plot_activation_maps(activations, layers)
    fig_kern = plot_kernel_on_image(image_tensor, kernels, layers, heads, model)

    k0 = kernels[0]["masked_kernel"]
    fig_head = plot_activation_on_image(
        image_tensor,
        activations,
        layers,
        heads,
        num_heads=k0.shape[0],
        head_dim=k0.shape[1],
    )
    return fig_to_image(fig_act), fig_to_image(fig_kern), fig_to_image(fig_head)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks app."""
    with gr.Blocks(
        title="Hyena Kernel Visualizer",
        theme=gr.themes.Soft(),
    ) as app:
        # ---- State ----
        model_state = gr.State(None)
        kernels_state = gr.State(None)
        activations_state = gr.State(None)
        image_tensor_state = gr.State(None)

        gr.Markdown(
            "# Hyena Kernel Visualizer\n\n"
            "Explore the learned implicit kernels of a Multi-Head Hyena model. "
            "Each Hyena layer replaces attention with a global convolution whose kernel is generated "
            "by a SIREN network and modulated by a learned Gaussian mask. "
            "Use the tabs below to go from a bird's-eye overview to per-head, per-channel detail."
        )

        with gr.Row():
            # ---- Sidebar ----
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Model")
                run_path = gr.Textbox(label="W&B run path (entity/project/run_id)")
                alias = gr.Dropdown(["best", "latest"], value="best", label="Alias")
                device = gr.Dropdown(["cuda", "cpu"], value="cuda", label="Device")
                load_btn = gr.Button("Load Model", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, value="No model loaded.")

                gr.Markdown("### Controls")
                layer_slider = gr.Slider(
                    minimum=0, maximum=11, step=1, value=0,
                    label="Layer",
                    info="Select which Hyena layer to inspect.",
                )
                gr.Markdown(
                    "📍 *Affects: Receptive Field, Kernel Structure, Image Analysis*",
                )
                head_slider = gr.Slider(
                    minimum=0, maximum=5, step=1, value=0,
                    label="Head",
                    info="Select which attention head to inspect.",
                )
                gr.Markdown(
                    "📍 *Affects: Kernel Structure, Image Analysis*",
                )
                all_heads_checkbox = gr.Checkbox(
                    label="Compare all heads",
                    value=False,
                    info="Show all heads side-by-side in Kernel Structure tab.",
                )

                gr.Markdown("### Image (optional)")
                gr.Markdown(
                    "*Upload an image to see how the kernels process real inputs "
                    "in the Image Analysis tab.*"
                )
                image_input = gr.Image(type="filepath", label="Upload image")
                img_status = gr.Textbox(
                    label="Image status", interactive=False, value=""
                )

            # ---- Main area ----
            with gr.Column(scale=3):
                with gr.Tabs():
                    # ==============================================================
                    # Tab 1: Overview
                    # ==============================================================
                    with gr.Tab("Overview"):
                        gr.Markdown(
                            "## How do kernels vary across the model?\n\n"
                            "This grid shows the **Frobenius norm** of each head's "
                            "head_dim x head_dim mixing matrix at every spatial position "
                            "(K_h x K_w), for all layers and all heads. It gives a quick "
                            "bird's-eye view of how kernel magnitude and shape evolve "
                            "with depth.\n\n"
                            "**What to look for:**\n"
                            "- Shallow layers should have broader, more uniform kernels "
                            "(local feature extraction).\n"
                            "- Deeper layers should show sharper, more structured patterns "
                            "(specialized long-range interactions).\n"
                            "- Different heads at the same layer should show diversity "
                            "(multi-scale behavior).\n\n"
                            "*Rendered once on model load. Not affected by sliders.*"
                        )
                        overview_img = gr.Image(
                            label="Kernel Norm Heatmaps (all layers x all heads)",
                            type="pil",
                        )

                    # ==============================================================
                    # Tab 2: Receptive Field
                    # ==============================================================
                    with gr.Tab("Receptive Field"):
                        gr.Markdown(
                            "## How does each head see the spatial world?\n\n"
                            "These panels show **all heads side by side** for the selected "
                            "layer, so you can compare how different heads partition the "
                            "spatial receptive field at the same depth.\n\n"
                            "*Controlled by the **Layer** slider. Head slider has no effect here.*"
                        )

                        gr.Markdown(
                            "### Raw vs Masked Kernel Norms\n\n"
                            "Left = raw SIREN output, right = after Gaussian mask modulation. "
                            "The mask controls the effective receptive field by suppressing "
                            "distant positions, creating a smooth falloff.\n\n"
                            "**What to look for:** Heads with narrower masks focus on local "
                            "context; wider masks capture global patterns. The mask should "
                            "produce a clean, smooth attenuation of the raw kernel."
                        )
                        raw_masked_img = gr.Image(
                            label="Raw vs Masked (all heads)", type="pil"
                        )

                        gr.Markdown(
                            "### Gaussian Mask Shape\n\n"
                            "The learned Gaussian envelope that modulates each head's kernel. "
                            "Annotated with sigma values indicating the effective receptive field "
                            "size in normalized coordinates (grid spans [-1, 1]).\n\n"
                            "**What to look for:** A diversity of sigma across heads indicates "
                            "multi-scale behavior. Small sigma = tight local attention, "
                            "large sigma = broad global attention."
                        )
                        gaussian_img = gr.Image(
                            label="Gaussian Masks (all heads)", type="pil"
                        )

                    # ==============================================================
                    # Tab 3: Kernel Structure
                    # ==============================================================
                    with gr.Tab("Kernel Structure"):
                        gr.Markdown(
                            "## What does this head compute?\n\n"
                            "Detailed analysis of a single head's kernel structure. "
                            "These panels reveal the internal computation — what spatial "
                            "frequencies are captured, how channels interact, and what "
                            "spatial patterns are learned.\n\n"
                            "*Controlled by both **Layer** and **Head** sliders.*"
                        )

                        gr.Markdown(
                            "### Kernel Clustering: Channel Mixing Modes\n\n"
                            "K-Means clustering (k=4) of PCA-reduced mixing matrices across "
                            "all spatial positions. Each pixel is colored by its cluster ID "
                            "(categorical colormap); brightness is modulated by the Frobenius "
                            "norm so faint positions fade out.\n\n"
                            "**What to look for:** Same color = same channel-mixing mode. "
                            "Different colors across spatial positions = the head applies "
                            "different transformations at different offsets. "
                            "A legend maps each color to a cluster (mixing mode) ID."
                        )
                        pca_img = gr.Image(label="Kernel Clustering", type="pil")

                        gr.Markdown(
                            "### Spectral Analysis (DC Suppressed)\n\n"
                            "2D FFT of the kernel norm map (log magnitude) with the DC "
                            "component suppressed. The DC (zero-frequency) component represents "
                            "the mean energy of the kernel and typically dominates the spectrum, "
                            "making it impossible to see any high-frequency structure. By zeroing "
                            "out the center pixel, the remaining frequency content becomes visible.\n\n"
                            "Axes show normalized spatial frequency in [-0.5, 0.5] cycles/pixel. "
                            "Center = low frequencies (smooth/global), edges = high frequencies "
                            "(sharp/local).\n\n"
                            "**What to look for:** Different heads should specialize in "
                            "different frequency bands. Deeper layers may capture higher "
                            "frequencies."
                        )
                        spectral_img = gr.Image(label="Spectral Analysis", type="pil")

                        gr.Markdown(
                            "### Channel Correlation Matrix\n\n"
                            "Each pixel (i, j) shows the Frobenius norm of the K_h x K_w "
                            "spatial filter between output channel i and input channel j. "
                            "This reveals how much cross-channel mixing the head performs.\n\n"
                            "**What to look for:** Diagonal-dominant = depthwise-like "
                            "(channels stay independent). Dense off-diagonal = heavy "
                            "cross-channel mixing. The d/o ratio (diagonal mean / "
                            "off-diagonal mean) quantifies this."
                        )
                        channel_corr_img = gr.Image(
                            label="Channel Correlation", type="pil"
                        )

                        gr.Markdown(
                            "### Head-Grid Mosaic: Spatial Filters per Channel Pair\n\n"
                            "Raw K_h x K_w spatial filters for specific (output_ch, input_ch) "
                            "pairs, shown as grayscale heatmaps. Top row = channel-mean. "
                            "Channel pairs are selected dynamically by spatial energy: the "
                            "top 2 most energetic diagonal and off-diagonal pairs are shown.\n\n"
                            "**What to look for:** Different heads learning different "
                            "orientations, frequencies, or receptive field shapes. "
                            "This is the most direct view of the raw learned kernel values."
                        )
                        kernel_slices_img = gr.Image(
                            label="Head-Grid Mosaic", type="pil"
                        )

                        gr.Markdown(
                            "### Mixing Strength Map (Top Singular Value)\n\n"
                            "The top singular value σ₁ of each spatial position's head_dim × "
                            "head_dim mixing matrix, displayed as a heatmap. This summarizes "
                            "how actively each spatial offset mixes channels — high σ₁ means "
                            "strong cross-channel interaction.\n\n"
                            "**What to look for:** Concentrated high σ₁ near the center = "
                            "local mixing dominates. Broad high σ₁ = long-range channel "
                            "interactions."
                        )
                        svd_img = gr.Image(
                            label="Mixing Strength (σ₁)", type="pil"
                        )

                    # ==============================================================
                    # Tab 4: Image Analysis
                    # ==============================================================
                    with gr.Tab("Image Analysis"):
                        gr.Markdown(
                            "## How does this head process a real image?\n\n"
                            "These panels show image-conditioned visualizations — how the "
                            "learned kernels interact with actual image content. Upload an "
                            "image in the sidebar to activate.\n\n"
                            "*Controlled by both **Layer** and **Head** sliders.*"
                        )

                        gr.Markdown(
                            "### Activation Maps: Effect of Global Convolution\n\n"
                            "Top = input activation magnitude (before convolution), "
                            "middle = output (after), bottom = **log₂ amplification ratio**. "
                            "We use log₂ instead of a raw ratio because: (1) it centers "
                            "\"no change\" at 0 instead of at 1, making the diverging colormap "
                            "meaningful (blue = suppression, red = amplification); (2) it is "
                            "symmetric — a 2× amplification (+1) and a 2× suppression (−1) "
                            "have equal visual weight; (3) it avoids numerical instability "
                            "from dividing by near-zero pre-conv activations.\n\n"
                            "**What to look for:** Spatially structured patterns where the "
                            "model amplifies salient regions and suppresses background."
                        )
                        activation_img = gr.Image(
                            label="Activation Maps", type="pil"
                        )

                        gr.Markdown(
                            "### Kernel Receptive Field on Image\n\n"
                            "The kernel's Frobenius norm (= effective receptive field "
                            "strength) upsampled to image resolution and overlaid on the "
                            "input. This is translation-invariant — it shows the kernel's "
                            "inherent spatial reach, not what it does for this specific input.\n\n"
                            "**What to look for:** Local heads highlight only the center "
                            "region; global heads cover a wider area."
                        )
                        kernel_overlay_img = gr.Image(
                            label="Kernel on Image", type="pil"
                        )

                        gr.Markdown(
                            "### Per-Head Activation on Image\n\n"
                            "Post-convolution activation magnitude for this head, overlaid "
                            "on the input image. Unlike the kernel overlay above, this is "
                            "input-dependent — it reveals what this head actually 'attends to' "
                            "for this specific image.\n\n"
                            "**What to look for:** Heads specializing in different aspects: "
                            "edges, textures, object parts, or background regions."
                        )
                        activation_head_img = gr.Image(
                            label="Per-Head Activation on Image", type="pil"
                        )

        # ---- Event wiring ----

        # Load model → update states, sliders, then render all tabs
        load_btn.click(
            fn=on_load_model,
            inputs=[run_path, alias, device],
            outputs=[model_state, kernels_state, layer_slider, head_slider, status],
        ).then(
            fn=on_render_overview,
            inputs=[kernels_state],
            outputs=[overview_img],
        ).then(
            fn=on_render_receptive_field,
            inputs=[kernels_state, layer_slider],
            outputs=[raw_masked_img, gaussian_img],
        ).then(
            fn=on_render_kernel_structure,
            inputs=[kernels_state, layer_slider, head_slider, all_heads_checkbox],
            outputs=[pca_img, spectral_img, channel_corr_img, kernel_slices_img, svd_img],
        )

        # --- Receptive Field tab: layer slider only ---
        receptive_field_inputs = [kernels_state, layer_slider]
        receptive_field_outputs = [raw_masked_img, gaussian_img]

        layer_slider.change(
            fn=on_render_receptive_field,
            inputs=receptive_field_inputs,
            outputs=receptive_field_outputs,
        )

        # --- Kernel Structure tab: both sliders + all-heads checkbox ---
        kernel_structure_inputs = [kernels_state, layer_slider, head_slider, all_heads_checkbox]
        kernel_structure_outputs = [
            pca_img,
            spectral_img,
            channel_corr_img,
            kernel_slices_img,
            svd_img,
        ]

        layer_slider.change(
            fn=on_render_kernel_structure,
            inputs=kernel_structure_inputs,
            outputs=kernel_structure_outputs,
        )
        head_slider.change(
            fn=on_render_kernel_structure,
            inputs=kernel_structure_inputs,
            outputs=kernel_structure_outputs,
        )
        all_heads_checkbox.change(
            fn=on_render_kernel_structure,
            inputs=kernel_structure_inputs,
            outputs=kernel_structure_outputs,
        )

        # --- Image Analysis tab: both sliders + image upload ---
        image_panel_inputs = [
            image_tensor_state,
            activations_state,
            kernels_state,
            model_state,
            layer_slider,
            head_slider,
        ]
        image_panel_outputs = [
            activation_img,
            kernel_overlay_img,
            activation_head_img,
        ]

        image_input.change(
            fn=on_image_upload,
            inputs=[image_input, model_state, device],
            outputs=[image_tensor_state, activations_state, img_status],
        ).then(
            fn=on_render_image_panels,
            inputs=image_panel_inputs,
            outputs=image_panel_outputs,
        )

        layer_slider.change(
            fn=on_render_image_panels,
            inputs=image_panel_inputs,
            outputs=image_panel_outputs,
        )
        head_slider.change(
            fn=on_render_image_panels,
            inputs=image_panel_inputs,
            outputs=image_panel_outputs,
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Interactive Hyena Kernel Visualizer")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio tunnel"
    )
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
