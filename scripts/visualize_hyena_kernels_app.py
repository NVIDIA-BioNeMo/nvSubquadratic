"""Interactive Hyena Kernel Visualization App (Gradio).

Provides an interactive UI for exploring learned Hyena kernels from trained
ViT-5 + Multi-Head Hyena models. Loads a model from W&B and renders a
complete visual report across representative layers and all heads.

Usage:
    PYTHONPATH=. python scripts/visualize_hyena_kernels_app.py --port 7860
    PYTHONPATH=. python scripts/visualize_hyena_kernels_app.py --share  # public tunnel
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import zipfile

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
    _select_detail_layers,
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


def on_load_model(run_path: str, alias: str, device: str, progress=gr.Progress()):
    """Load model + extract kernels."""
    if not run_path or not run_path.strip():
        return (
            None,
            None,
            "⚠️ Please enter a W&B run path (entity/project/run_id).",
        )

    if device == "cpu":
        gr.Info("CPU mode selected — loading and rendering will be slower than on GPU.")

    try:
        progress(0.0, desc="Downloading checkpoint from W&B…")
        model = load_model(None, run_path, alias, device)
    except Exception as e:
        return (
            None,
            None,
            f"❌ Error loading model: {e}",
        )

    progress(0.6, desc="Extracting kernels from all layers…")
    kernels = extract_all_kernels(model)
    num_layers = len(model.blocks)
    num_heads = kernels[0]["masked_kernel"].shape[0]
    detail = _select_detail_layers(num_layers)
    progress(1.0, desc="Done!")
    status = (
        f"✅ Loaded: {num_layers} layers, {num_heads} heads/layer, device={device}\n"
        f"Detail layers: {detail} (first / middle / last)"
    )

    return model, kernels, status


def on_render_overview(kernels):
    """Full kernel norm grid across all layers and heads."""
    if kernels is None:
        return _placeholder()
    num_layers = len(kernels)
    fig = plot_kernel_norm_heatmaps(kernels, list(range(num_layers)), None)
    return fig_to_image(fig)


def on_render_receptive_field(kernels, progress=gr.Progress()):
    """Receptive field panels for representative layers × all heads."""
    if kernels is None:
        ph = _placeholder()
        return ph, ph
    detail_layers = _select_detail_layers(len(kernels))

    progress(0.0, desc="Rendering raw vs masked…")
    fig_rm = plot_raw_vs_masked(kernels, detail_layers, None)
    progress(0.5, desc="Rendering Gaussian masks…")
    fig_gauss = plot_gaussian_masks(kernels, detail_layers, None)
    progress(1.0, desc="Done!")

    return fig_to_image(fig_rm), fig_to_image(fig_gauss)


def on_render_kernel_structure(kernels, progress=gr.Progress()):
    """Kernel structure panels for representative layers × all heads."""
    if kernels is None:
        ph = _placeholder()
        return ph, ph, ph, ph, ph
    detail_layers = _select_detail_layers(len(kernels))

    progress(0.0, desc="Rendering kernel clustering…")
    fig_pca = plot_kernel_pca(kernels, detail_layers, None)
    progress(0.2, desc="Rendering spectral analysis…")
    fig_spec = plot_spectral_analysis(kernels, detail_layers, None)
    progress(0.4, desc="Rendering channel correlation…")
    fig_corr = plot_channel_correlation(kernels, detail_layers, None)
    progress(0.6, desc="Rendering head-grid mosaic…")
    fig_slices = plot_kernel_slices(kernels, detail_layers, None)
    progress(0.8, desc="Rendering mixing strength…")
    fig_svd = plot_mixing_svd_map(kernels, detail_layers, None)
    progress(1.0, desc="Done!")

    return (
        fig_to_image(fig_pca),
        fig_to_image(fig_spec),
        fig_to_image(fig_corr),
        fig_to_image(fig_slices),
        fig_to_image(fig_svd),
    )


def on_image_upload(image_path, model, device):
    """Load and preprocess image, run forward pass with hooks."""
    if model is None:
        gr.Warning("Please load a model first before uploading an image.")
        return None, None, "⚠️ No model loaded."
    if image_path is None:
        return None, None, ""
    try:
        img_tensor = load_and_preprocess_image(image_path)
        activations = forward_with_hooks(model, img_tensor, device)
        return img_tensor, activations, "✅ Forward pass complete."
    except Exception as e:
        return None, None, f"❌ Error: {e}"


def on_render_image_panels(image_tensor, activations, kernels, model):
    """Image-conditioned panels for representative layers × all heads."""
    ph = _placeholder("Upload an image first.")
    if image_tensor is None or activations is None or kernels is None:
        return ph, ph, ph
    detail_layers = _select_detail_layers(len(kernels))

    k0 = kernels[0]["masked_kernel"]
    fig_act = plot_activation_maps(activations, detail_layers)
    fig_kern = plot_kernel_on_image(image_tensor, kernels, detail_layers, None, model)
    fig_head = plot_activation_on_image(
        image_tensor,
        activations,
        detail_layers,
        None,
        num_heads=k0.shape[0],
        head_dim=k0.shape[1],
    )
    return fig_to_image(fig_act), fig_to_image(fig_kern), fig_to_image(fig_head)


def on_export_panels(
    overview, raw_masked, gaussian,
    pca, spectral, corr, slices, svd,
    activation, kernel_overlay, activation_head,
):
    """Bundle all rendered panels into a ZIP for download."""
    panels = {
        "01_overview.png": overview,
        "02_raw_vs_masked.png": raw_masked,
        "03_gaussian_masks.png": gaussian,
        "04_kernel_clustering.png": pca,
        "05_spectral_analysis.png": spectral,
        "06_channel_correlation.png": corr,
        "07_head_grid_mosaic.png": slices,
        "08_mixing_strength.png": svd,
        "09_activation_maps.png": activation,
        "10_kernel_on_image.png": kernel_overlay,
        "11_activation_on_image.png": activation_head,
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img in panels.items():
            if img is None:
                continue
            buf = io.BytesIO()
            if isinstance(img, Image.Image):
                img.save(buf, format="PNG", dpi=(200, 200))
            else:
                continue
            zf.writestr(name, buf.getvalue())
    tmp.close()
    return gr.File(value=tmp.name, visible=True)


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
            "Load a model to generate a complete visual report across representative layers "
            "(first, middle, last) and all heads."
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

                gr.Markdown("### Image (optional)")
                gr.Markdown(
                    "*Upload an image to populate the Image Analysis tab.*"
                )
                image_input = gr.Image(type="filepath", label="Upload image")
                img_status = gr.Textbox(
                    label="Image status", interactive=False, value=""
                )

                gr.Markdown("---")
                export_btn = gr.Button("📥 Export All Panels", variant="secondary")
                export_file = gr.File(label="Download", visible=False)

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
                            "head_dim × head_dim mixing matrix at every spatial position "
                            "(K_h × K_w), for **all layers and all heads**.\n\n"
                            "**What to look for:**\n"
                            "- Shallow layers → broader, more uniform kernels "
                            "(local feature extraction)\n"
                            "- Deeper layers → sharper, more structured patterns "
                            "(specialized long-range interactions)\n"
                            "- Different heads at the same layer → diversity "
                            "(multi-scale behavior)"
                        )
                        overview_img = gr.Image(
                            label="Kernel Norm Heatmaps (all layers × all heads)",
                            type="pil",
                        )

                    # ==============================================================
                    # Tab 2: Receptive Field
                    # ==============================================================
                    with gr.Tab("Receptive Field"):
                        gr.Markdown(
                            "## How does each head see the spatial world?\n\n"
                            "These panels show **all heads side by side** for 3 representative "
                            "layers (first, middle, last), so you can compare how heads partition "
                            "the spatial receptive field at different depths."
                        )

                        gr.Markdown(
                            "### Raw vs Masked Kernel Norms\n\n"
                            "Left = raw SIREN output, right = after Gaussian mask modulation. "
                            "The mask controls the effective receptive field by suppressing "
                            "distant positions.\n\n"
                            "**What to look for:** Narrower masks → local context; "
                            "wider masks → global patterns. The mask should produce a clean, "
                            "smooth attenuation."
                        )
                        raw_masked_img = gr.Image(
                            label="Raw vs Masked (all heads)", type="pil"
                        )

                        gr.Markdown(
                            "### Gaussian Mask Shape\n\n"
                            "The learned Gaussian envelope per head, annotated with σ values "
                            "(grid spans [-1, 1]).\n\n"
                            "**What to look for:** Diversity of σ across heads = multi-scale "
                            "behavior. Small σ = tight local, large σ = broad global."
                        )
                        gaussian_img = gr.Image(
                            label="Gaussian Masks (all heads)", type="pil"
                        )

                    # ==============================================================
                    # Tab 3: Kernel Structure
                    # ==============================================================
                    with gr.Tab("Kernel Structure"):
                        gr.Markdown(
                            "## What does each head compute?\n\n"
                            "Detailed analysis of kernel structure for 3 representative "
                            "layers × all heads. These panels reveal the internal computation — "
                            "spatial frequencies, channel interactions, and learned patterns."
                        )

                        gr.Markdown(
                            "### Kernel Clustering: Channel Mixing Modes\n\n"
                            "K-Means clustering (k=4) of PCA-reduced mixing matrices across "
                            "all spatial positions. Each pixel is colored by cluster ID; "
                            "brightness is modulated by Frobenius norm.\n\n"
                            "**What to look for:** Same color = same channel-mixing mode. "
                            "Different colors across positions = the head applies "
                            "different transformations at different offsets."
                        )
                        pca_img = gr.Image(label="Kernel Clustering", type="pil")

                        gr.Markdown(
                            "### Spectral Analysis (DC Suppressed)\n\n"
                            "2D FFT of the kernel norm map (log magnitude) with DC "
                            "component zeroed out. Axes show normalized spatial frequency "
                            "in [-0.5, 0.5] cycles/pixel.\n\n"
                            "**What to look for:** Center = low freq (smooth/global), "
                            "edges = high freq (sharp/local). Different heads should "
                            "specialize in different frequency bands."
                        )
                        spectral_img = gr.Image(label="Spectral Analysis", type="pil")

                        gr.Markdown(
                            "### Channel Correlation Matrix\n\n"
                            "Each pixel (i, j) = Frobenius norm of the K_h × K_w "
                            "spatial filter between output channel i and input channel j.\n\n"
                            "**What to look for:** Diagonal-dominant = depthwise-like "
                            "(channels independent). Dense off-diagonal = heavy "
                            "cross-channel mixing. d/o ratio quantifies this."
                        )
                        channel_corr_img = gr.Image(
                            label="Channel Correlation", type="pil"
                        )

                        gr.Markdown(
                            "### Head-Grid Mosaic: Spatial Filters per Channel Pair\n\n"
                            "Raw K_h × K_w spatial filters for specific (out_ch, in_ch) "
                            "pairs as grayscale heatmaps. Top row = channel-mean. "
                            "Pairs selected by spatial energy (top diagonal + off-diagonal).\n\n"
                            "**What to look for:** Different heads learning different "
                            "orientations, frequencies, or receptive field shapes."
                        )
                        kernel_slices_img = gr.Image(
                            label="Head-Grid Mosaic", type="pil"
                        )

                        gr.Markdown(
                            "### Mixing Strength Map (Top Singular Value)\n\n"
                            "σ₁ of each spatial position's head_dim × head_dim mixing "
                            "matrix. High σ₁ = strong cross-channel interaction.\n\n"
                            "**What to look for:** Concentrated σ₁ near center = "
                            "local mixing. Broad σ₁ = long-range channel interactions."
                        )
                        svd_img = gr.Image(
                            label="Mixing Strength (σ₁)", type="pil"
                        )

                    # ==============================================================
                    # Tab 4: Image Analysis
                    # ==============================================================
                    with gr.Tab("Image Analysis"):
                        gr.Markdown(
                            "## How do kernels process a real image?\n\n"
                            "Image-conditioned visualizations for 3 representative "
                            "layers × all heads. Upload an image in the sidebar to "
                            "populate these panels."
                        )

                        gr.Markdown(
                            "### Activation Maps: Effect of Global Convolution\n\n"
                            "Top = input magnitude (before conv), middle = output (after), "
                            "bottom = **log₂ amplification ratio** (blue = suppression, "
                            "red = amplification, 0 = no change).\n\n"
                            "**What to look for:** Spatially structured amplification "
                            "where the model boosts salient regions."
                        )
                        activation_img = gr.Image(
                            label="Activation Maps", type="pil"
                        )

                        gr.Markdown(
                            "### Kernel Receptive Field on Image\n\n"
                            "Kernel Frobenius norm upsampled to image resolution and "
                            "overlaid. Translation-invariant — shows the kernel's "
                            "inherent spatial reach.\n\n"
                            "**What to look for:** Local heads → center only; "
                            "global heads → wider area."
                        )
                        kernel_overlay_img = gr.Image(
                            label="Kernel on Image", type="pil"
                        )

                        gr.Markdown(
                            "### Per-Head Activation on Image\n\n"
                            "Post-conv activation magnitude overlaid on the input. "
                            "Unlike the kernel overlay, this is **input-dependent**.\n\n"
                            "**What to look for:** Heads specializing in edges, "
                            "textures, object parts, or background."
                        )
                        activation_head_img = gr.Image(
                            label="Per-Head Activation on Image", type="pil"
                        )

                    # ==============================================================
                    # Tab 5: Guide
                    # ==============================================================
                    with gr.Tab("📖 Guide"):
                        gr.Markdown(
                            "## Hyena Kernel Visualizer — Quick Reference\n\n"
                            "### What is a Hyena layer?\n\n"
                            "A Hyena layer replaces standard self-attention with a **gated global convolution**. "
                            "The pipeline for each layer is:\n\n"
                            "1. **SIREN network** generates a continuous kernel from spatial coordinates\n"
                            "2. **Learned Gaussian mask** modulates the kernel to control the effective receptive field\n"
                            "3. **Global convolution** (via FFT) applies the masked kernel to the input feature map\n"
                            "4. **Multiplicative gating** (Q ⊙ σ(K) and h ⊙ σ(V)) controls information flow\n\n"
                            "### Kernel dimensions\n\n"
                            "Each Hyena layer produces a kernel tensor of shape:\n\n"
                            "`[num_heads, head_dim, head_dim, K_h, K_w]`\n\n"
                            "| Dimension | Meaning |\n"
                            "|---|---|\n"
                            "| `num_heads` | Number of independent heads (like multi-head attention) |\n"
                            "| `head_dim × head_dim` | Channel-mixing matrix at each spatial position |\n"
                            "| `K_h × K_w` | Spatial extent of the kernel (14×14 for 224px images with patch size 16) |\n\n"
                            "### Tab guide\n\n"
                            "| Tab | What it shows | Key insight |\n"
                            "|---|---|---|\n"
                            "| **Overview** | Frobenius norm heatmaps for all layers × heads | How kernel magnitude evolves with depth |\n"
                            "| **Receptive Field** | Raw vs masked kernels + Gaussian envelopes | How each head's spatial reach is shaped |\n"
                            "| **Kernel Structure** | Clustering, FFT, correlations, SVD | What each head computes internally |\n"
                            "| **Image Analysis** | Activation maps + overlays on real images | How kernels interact with actual content |\n\n"
                            "All panels show **3 representative layers** (first, middle, last) × **all heads**. "
                            "No manual slider interaction needed.\n\n"
                            "### Glossary\n\n"
                            "| Term | Definition |\n"
                            "|---|---|\n"
                            "| **Frobenius norm** | √(Σᵢⱼ aᵢⱼ²) — overall magnitude of a matrix |\n"
                            "| **Mixing matrix** | The head_dim × head_dim matrix K[:,:,y,x] showing how channels interact at position (y,x) |\n"
                            "| **σ₁ (top singular value)** | Largest singular value of the mixing matrix — summarizes mixing strength |\n"
                            "| **DC component** | Zero-frequency (mean) of the FFT — suppressed in spectral plots to reveal structure |\n"
                            "| **d/o ratio** | Diagonal mean / off-diagonal mean of channel correlation — measures depthwise vs cross-channel behavior |\n"
                            "| **SIREN** | Sinusoidal Representation Network — generates the continuous kernel from coordinates |\n"
                            "| **K-Means modes** | Clusters of similar mixing matrices across spatial positions — reveals functional diversity |\n"
                        )

        # ---- Event wiring ----

        # Load model → render all kernel-based tabs
        load_btn.click(
            fn=on_load_model,
            inputs=[run_path, alias, device],
            outputs=[model_state, kernels_state, status],
        ).then(
            fn=on_render_overview,
            inputs=[kernels_state],
            outputs=[overview_img],
        ).then(
            fn=on_render_receptive_field,
            inputs=[kernels_state],
            outputs=[raw_masked_img, gaussian_img],
        ).then(
            fn=on_render_kernel_structure,
            inputs=[kernels_state],
            outputs=[pca_img, spectral_img, channel_corr_img, kernel_slices_img, svd_img],
        )

        # Image upload → forward pass → render image panels
        image_input.change(
            fn=on_image_upload,
            inputs=[image_input, model_state, device],
            outputs=[image_tensor_state, activations_state, img_status],
        ).then(
            fn=on_render_image_panels,
            inputs=[image_tensor_state, activations_state, kernels_state, model_state],
            outputs=[activation_img, kernel_overlay_img, activation_head_img],
        )

        # Export
        export_inputs = [
            overview_img, raw_masked_img, gaussian_img,
            pca_img, spectral_img, channel_corr_img, kernel_slices_img, svd_img,
            activation_img, kernel_overlay_img, activation_head_img,
        ]
        export_btn.click(
            fn=on_export_panels,
            inputs=export_inputs,
            outputs=[export_file],
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
