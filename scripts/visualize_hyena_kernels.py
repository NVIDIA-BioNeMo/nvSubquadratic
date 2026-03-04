"""Visualize learned Hyena kernels from a trained ViT-5 + Multi-Head Hyena model.

Downloads a checkpoint from W&B, loads the model, extracts implicit SIREN kernels
from each layer, and produces a set of diagnostic visualizations.

Usage:
    python scripts/visualize_hyena_kernels.py \
        --run-path "entity/project/run_id" \
        --alias best \
        --image-path /path/to/imagenet/val/image.JPEG \
        --output-dir ./kernel_viz_output \
        --layers 0,5,11 \
        --heads 0,1,2 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Redirect caches to a volume with sufficient storage (must be set before imports).
_CACHE_ROOT = "/ivi/zfs/s0/original_homes/dknigge/.cache"
os.environ.setdefault("MPLCONFIGDIR", f"{_CACHE_ROOT}/matplotlib")
os.environ.setdefault("WANDB_CACHE_DIR", f"{_CACHE_ROOT}/wandb")
os.environ.setdefault("WANDB_DATA_DIR", f"{_CACHE_ROOT}/wandb")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------


_ARTIFACTS_ROOT = Path(f"{_CACHE_ROOT}/nvsubquadratic_artifacts")


def _download_checkpoint(run_path: str, alias: str) -> str:
    """Download checkpoint artifact from W&B to the alternate storage volume.

    Same logic as experiments.utils.checkpointing.download_checkpoint but
    downloads to ``_ARTIFACTS_ROOT`` instead of the CWD-relative ``.artifacts/``.
    """
    import wandb

    from experiments.utils.checkpointing import _select_artifact_with_alias

    api = wandb.Api()
    run = api.run(run_path)

    artifacts = list(run.logged_artifacts())
    if not artifacts:
        raise ValueError(f"No artifacts found for run '{run_path}'.")

    artifact = _select_artifact_with_alias(artifacts, alias=alias)
    if artifact is None:
        raise ValueError(
            f"No artifact with alias '{alias}' found for run '{run_path}'. "
            f"Available: {[a.name + ':' + ','.join(a.aliases) for a in artifacts]}"
        )

    run_id = run.id
    target_root = _ARTIFACTS_ROOT / run_id / alias
    target_root.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(artifact.download(root=str(target_root)))
    ckpt_files = sorted(artifact_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpt_files:
        raise ValueError(f"No .ckpt files found in artifact '{artifact.name}:{alias}' at {artifact_dir}.")
    return str(ckpt_files[0])


def _fetch_net_config_from_wandb(run_path: str) -> dict:
    """Retrieve the network config dict from a W&B run's logged config.

    The training script stores the full ``ExperimentConfig`` (serialized via
    ``config_to_dict``) in the W&B run config.  ``LazyConfig`` objects are
    stored as plain dicts with a ``__target__`` key, which ``instantiate()``
    already handles.

    Returns:
        The ``net`` sub-dict suitable for ``instantiate()``.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    run_config = run.config

    if "net" not in run_config:
        raise KeyError(
            f"W&B run '{run_path}' does not have a 'net' key in its config. "
            f"Available top-level keys: {list(run_config.keys())}"
        )
    return run_config["net"]


def load_model(
    config_path: str | None,
    run_path: str,
    alias: str,
    device: str,
) -> torch.nn.Module:
    """Download checkpoint from W&B, instantiate model, and load weights.

    Args:
        config_path: Path to the experiment config .py file.  If ``None``,
            the network config is fetched from the W&B run's logged config.
        run_path: W&B run path ``entity/project/run_id``.
        alias: Checkpoint alias (``best`` or ``latest``).
        device: Target device.

    Returns:
        The loaded network in eval mode.
    """
    from experiments.utils.checkpointing import (
        StripCompiledPrefix,
        load_checkpoint_state_dict,
    )
    from nvsubquadratic.lazy_config import instantiate

    # Download to alternate storage volume
    ckpt_path = _download_checkpoint(run_path, alias=alias)
    print(f"Downloaded checkpoint: {ckpt_path}")

    # State dict
    state_dict = load_checkpoint_state_dict(ckpt_path)
    state_dict = StripCompiledPrefix()(state_dict)
    # Strip Lightning wrapper prefixes.
    # ClassificationWrapper stores the net as self.model, some configs add a .network level.
    state_dict = {k.removeprefix("model.").removeprefix("network."): v for k, v in state_dict.items()}

    # Instantiate model from config
    if config_path:
        from experiments.utils.cli import load_config_from_file

        config = load_config_from_file(config_path)
        net_config = config.net
    else:
        print("No config path provided — fetching network config from W&B run...")
        net_config = _fetch_net_config_from_wandb(run_path)

    network = instantiate(net_config)

    # Load weights
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    network.to(device).eval()
    return network


# ---------------------------------------------------------------------------
# 2. Kernel extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_kernel(global_conv, spatial_dims: tuple[int, int]) -> dict:
    """Extract raw and masked kernels from a CKConvMultiheadND layer.

    Mirrors the forward logic of CKConvMultiheadND (lines 159-176).

    Returns dict with:
        raw_kernel:    [num_heads, head_dim, head_dim, K_h, K_w]
        masked_kernel: [num_heads, head_dim, head_dim, K_h, K_w]
        grid:          [1, K_h_grid, K_w_grid, 2]
        mask_stds:     [data_dim, num_channels] or None
    """
    num_heads = global_conv.num_heads
    head_dim = global_conv.head_dim

    if global_conv.grid_type == "single":
        grid_lens = [(s + 1) // 2 for s in spatial_dims]
    else:
        grid_lens = list(spatial_dims)

    conv_kernel_flat, grid = global_conv.kernel(grid_lens)
    K_h, K_w = conv_kernel_flat.shape[1], conv_kernel_flat.shape[2]

    def _reshape(flat):
        return (
            flat[0]
            .view(K_h, K_w, num_heads, head_dim, head_dim)
            .permute(2, 3, 4, 0, 1)
            .contiguous()
        )

    raw_kernel = _reshape(conv_kernel_flat).cpu().float()

    # Apply mask
    mask_stds = None
    if not isinstance(global_conv.mask, torch.nn.Identity):
        conv_kernel_flat = global_conv.mask(grid=grid, x=conv_kernel_flat)
        if hasattr(global_conv.mask, "_compute_std"):
            mask_stds = global_conv.mask._compute_std().detach().cpu().float()

    masked_kernel = _reshape(conv_kernel_flat).cpu().float()

    return {
        "raw_kernel": raw_kernel,
        "masked_kernel": masked_kernel,
        "grid": grid.cpu().float(),
        "mask_stds": mask_stds,
    }


def extract_all_kernels(model: torch.nn.Module) -> list[dict]:
    """Extract kernels from every Hyena layer.

    Access path: model.blocks[i].sequence_mixer.inner_mixer.mixer.global_conv
    """
    kernels = []
    for i, block in enumerate(model.blocks):
        global_conv = block.sequence_mixer.inner_mixer.mixer.global_conv
        grid_w = block.sequence_mixer.grid_w
        l_cache = global_conv.kernel.positional_embedding.L_cache
        spatial_dims = (l_cache, grid_w)

        kdata = extract_kernel(global_conv, spatial_dims)
        kdata["layer_idx"] = i
        kdata["spatial_dims"] = spatial_dims
        kernels.append(kdata)
    return kernels


# ---------------------------------------------------------------------------
# 3. Image loading
# ---------------------------------------------------------------------------


def load_and_preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess a single ImageNet image.

    Returns [1, H, W, 3] channels-last tensor (model input format).
    """
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor.permute(0, 2, 3, 1)  # [1, H, W, 3]


# ---------------------------------------------------------------------------
# 4. Forward pass with hooks
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward_with_hooks(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Forward an image and capture pre/post global-conv activations per layer."""
    activations: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []

    for i, block in enumerate(model.blocks):
        gc = block.sequence_mixer.inner_mixer.mixer.global_conv

        def _make_hook(layer_i):
            def _hook(module, inp, out):
                activations[layer_i] = {
                    "pre_conv": inp[0].detach().cpu().float(),
                    "post_conv": out.detach().cpu().float(),
                }

            return _hook

        hooks.append(gc.register_forward_hook(_make_hook(i)))

    # Temporarily disable the quack RMSNorm CUDA kernel — it has strict stride-alignment
    # requirements that the Hyena reshape path can violate during isolated inference.
    # The pure-PyTorch fallback is fine for a single-image visualization forward pass.
    import nvsubquadratic.modules.rms_norm as _rms_mod

    _original_quack = _rms_mod._quack_rmsnorm
    _rms_mod._quack_rmsnorm = None
    try:
        model({"input": image_tensor.to(device), "condition": None})
    finally:
        _rms_mod._quack_rmsnorm = _original_quack

    for h in hooks:
        h.remove()

    return activations


# ---------------------------------------------------------------------------
# 5. Visualization helpers
# ---------------------------------------------------------------------------

# Consistent styling
_CMAP_KERNEL = "inferno"
_CMAP_DIVERGE = "RdBu_r"
_CMAP_SPECTRAL = "viridis"


def _select_detail_layers(num_blocks: int, selected: list[int] | None = None) -> list[int]:
    """Pick first, middle, last layer indices from the selected set."""
    if selected is None:
        selected = list(range(num_blocks))
    if len(selected) <= 3:
        return selected
    return [selected[0], selected[len(selected) // 2], selected[-1]]


# ---------------------------------------------------------------------------
# Panel 1: Kernel norm heatmaps (all layers x heads)
# ---------------------------------------------------------------------------


def plot_kernel_norm_heatmaps(
    all_kernels: list[dict],
    selected_layers: list[int],
    selected_heads: list[int] | None,
) -> plt.Figure:
    """Frobenius norm of the head_dim x head_dim mixing matrix at each spatial position."""
    kernels = [all_kernels[i] for i in selected_layers]
    num_heads = kernels[0]["masked_kernel"].shape[0]
    heads = selected_heads if selected_heads is not None else list(range(num_heads))
    n_layers = len(kernels)
    n_heads = len(heads)

    # Precompute norm maps and global range (99th percentile clipping)
    norm_maps = []
    all_values = []
    for kdata in kernels:
        k = kdata["masked_kernel"]  # [num_heads, head_dim, head_dim, K_h, K_w]
        norms = k.norm(dim=(1, 2))  # [num_heads, K_h, K_w]
        norm_maps.append(norms)
        for h in heads:
            all_values.append(norms[h].flatten())
    all_values = torch.cat(all_values)  # [total_pixels]
    global_vmin = all_values.min().item()
    global_vmax = torch.quantile(all_values, 0.99).item()  # 99th percentile

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.5 * n_heads, 2.2 * n_layers), squeeze=False)
    for row, (kdata, norms) in enumerate(zip(kernels, norm_maps)):
        for col, h in enumerate(heads):
            ax = axes[row, col]
            im = ax.imshow(
                norms[h].numpy(),
                cmap=_CMAP_KERNEL,
                vmin=global_vmin,
                vmax=global_vmax,
                origin="lower",
                aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Head {h}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Layer {kdata['layer_idx']}", fontsize=9)

    fig.suptitle("Kernel Frobenius Norm per Spatial Position", fontsize=13, y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.6, label="||K||_F", pad=0.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Overview of kernel magnitude across all layers and heads. Each cell shows the Frobenius norm\n"
        "of the head_dim×head_dim mixing matrix at each spatial position. Expect shallow layers to have broader,\n"
        "more uniform kernels (local features) and deeper layers to show sharper, more structured patterns.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 2: Raw vs masked comparison
# ---------------------------------------------------------------------------


def plot_raw_vs_masked(
    all_kernels: list[dict],
    detail_layers: list[int],
    selected_heads: list[int] | None,
) -> plt.Figure:
    """Side-by-side norm heatmaps before/after Gaussian mask."""
    kernels = [all_kernels[i] for i in detail_layers]
    num_heads = kernels[0]["masked_kernel"].shape[0]
    heads = selected_heads if selected_heads is not None else list(range(num_heads))
    n_layers = len(kernels)
    n_heads = len(heads)

    fig, axes = plt.subplots(n_layers, 2 * n_heads, figsize=(2.2 * 2 * n_heads, 2.2 * n_layers), squeeze=False)

    for row, kdata in enumerate(kernels):
        raw_norms = kdata["raw_kernel"].norm(dim=(1, 2))  # [num_heads, K_h, K_w]
        masked_norms = kdata["masked_kernel"].norm(dim=(1, 2))  # [num_heads, K_h, K_w]
        # Use 99th percentile clipping to prevent outlier wash-out
        combined = torch.cat([raw_norms.flatten(), masked_norms.flatten()])  # [2*num_heads*K_h*K_w]
        vmax = torch.quantile(combined, 0.99).item()

        for ci, h in enumerate(heads):
            ax_raw = axes[row, 2 * ci]
            ax_raw.imshow(raw_norms[h].numpy(), cmap=_CMAP_KERNEL, vmin=0, vmax=vmax, origin="lower", aspect="auto")
            ax_raw.set_xticks([])
            ax_raw.set_yticks([])
            if row == 0:
                ax_raw.set_title(f"H{h} raw", fontsize=8)
            if ci == 0:
                ax_raw.set_ylabel(f"Layer {kdata['layer_idx']}", fontsize=9)

            ax_m = axes[row, 2 * ci + 1]
            ax_m.imshow(masked_norms[h].numpy(), cmap=_CMAP_KERNEL, vmin=0, vmax=vmax, origin="lower", aspect="auto")
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            if row == 0:
                ax_m.set_title(f"H{h} masked", fontsize=8)

    fig.suptitle("Raw vs Masked Kernel Norms", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Show the effect of the learned Gaussian modulation mask. Left=raw SIREN output, right=after mask.\n"
        "The mask controls the effective receptive field. Expect the mask to suppress distant positions,\n"
        "creating a smooth falloff. Heads with narrower masks focus on local context; wider masks capture global patterns.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 3: Gaussian mask receptive fields
# ---------------------------------------------------------------------------


def plot_gaussian_masks(
    all_kernels: list[dict],
    detail_layers: list[int],
    selected_heads: list[int] | None,
) -> plt.Figure:
    """Reconstruct and visualize the learned Gaussian envelope per head."""
    kernels = [all_kernels[i] for i in detail_layers]
    num_heads_total = kernels[0]["masked_kernel"].shape[0]
    head_dim = kernels[0]["masked_kernel"].shape[1]
    heads = selected_heads if selected_heads is not None else list(range(num_heads_total))
    n_layers = len(kernels)
    n_heads = len(heads)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.5 * n_heads, 2.2 * n_layers), squeeze=False)

    for row, kdata in enumerate(kernels):
        mask_stds = kdata["mask_stds"]  # [data_dim, num_channels] or None
        grid = kdata["grid"]  # [1, K_h, K_w, 2]

        if mask_stds is None:
            for ci in range(n_heads):
                axes[row, ci].text(0.5, 0.5, "No mask", ha="center", va="center", transform=axes[row, ci].transAxes)
                axes[row, ci].set_xticks([])
                axes[row, ci].set_yticks([])
            continue

        # Average stds over the head_dim^2 channels belonging to each head
        # mask_stds: [2, num_heads * head_dim^2]
        stds_per_head = mask_stds.view(2, num_heads_total, head_dim * head_dim).mean(dim=2)  # [2, num_heads]

        for ci, h in enumerate(heads):
            ax = axes[row, ci]
            sigma_h = stds_per_head[0, h].item()
            sigma_w = stds_per_head[1, h].item()

            # Reconstruct Gaussian on grid
            g = grid[0]  # [K_h, K_w, 2]
            gauss = torch.exp(-0.5 * (g[..., 0] ** 2 / max(sigma_h, 1e-6) ** 2 + g[..., 1] ** 2 / max(sigma_w, 1e-6) ** 2))

            im = ax.imshow(gauss.numpy(), cmap=_CMAP_KERNEL, vmin=0, vmax=1, origin="lower", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"H{h} σ=({sigma_h:.2f},{sigma_w:.2f})", fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"Layer {kdata['layer_idx']}", fontsize=9)

    fig.suptitle("Gaussian Mask Receptive Fields (per-head avg)", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Visualize the learned Gaussian envelope that modulates each head's kernel. σ values indicate\n"
        "the effective receptive field size in normalized coordinates (grid spans [-1,1]). Small σ = tight local\n"
        "attention; large σ = broad global attention. Expect a diversity of σ across heads (multi-scale behavior).",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 4: Spectral analysis
# ---------------------------------------------------------------------------


def plot_spectral_analysis(
    all_kernels: list[dict],
    detail_layers: list[int],
    selected_heads: list[int] | None,
) -> plt.Figure:
    """2D FFT log-magnitude spectrum of kernel norm maps."""
    kernels = [all_kernels[i] for i in detail_layers]
    num_heads = kernels[0]["masked_kernel"].shape[0]
    heads = selected_heads if selected_heads is not None else list(range(num_heads))
    n_layers = len(kernels)
    n_heads = len(heads)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.5 * n_heads, 2.2 * n_layers), squeeze=False)

    for row, kdata in enumerate(kernels):
        norms = kdata["masked_kernel"].norm(dim=(1, 2))  # [num_heads, K_h, K_w]
        for ci, h in enumerate(heads):
            ax = axes[row, ci]
            norm_map = norms[h]  # [K_h, K_w]
            Kh, Kw = norm_map.shape
            fft_mag = torch.fft.fftshift(torch.fft.fft2(norm_map)).abs()  # [K_h, K_w]

            # Suppress DC component (center pixel) so high-frequency details are visible
            dc_h, dc_w = Kh // 2, Kw // 2
            fft_mag[dc_h, dc_w] = 0.0

            log_mag = torch.log1p(fft_mag).numpy()  # [K_h, K_w]

            # Plot with relative frequency axes [-0.5, 0.5] cycles/pixel
            extent = [-0.5, 0.5, -0.5, 0.5]  # [left, right, bottom, top]
            ax.imshow(log_mag, cmap=_CMAP_SPECTRAL, origin="lower", aspect="auto", extent=extent)
            ax.set_xticks([-0.5, 0, 0.5])
            ax.set_yticks([-0.5, 0, 0.5])
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"Head {h}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"Layer {kdata['layer_idx']}", fontsize=9)

    fig.suptitle("Spectral Analysis (log |FFT| of kernel norms, DC suppressed)", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Reveal which spatial frequencies each head captures via 2D FFT of the kernel norm map.\n"
        "DC component (center) is suppressed to expose high-frequency structure. Axes show normalized\n"
        "frequency [-0.5, 0.5] cycles/pixel. Different heads should specialize in different frequency bands.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 5: Mixing matrices (dense spatial sampling)
# ---------------------------------------------------------------------------


def plot_mixing_matrices(
    all_kernels: list[dict],
    detail_layers: list[int],
    selected_heads: list[int] | None,
    spatial_step: int = 4,
) -> list[plt.Figure]:
    """Visualize head_dim x head_dim mixing matrices densely across the spatial grid.

    Produces one figure per selected layer. Within each figure:
        rows = sampled spatial positions (every spatial_step-th position)
        cols = selected heads
    """
    kernels = [all_kernels[i] for i in detail_layers]
    num_heads_total = kernels[0]["masked_kernel"].shape[0]
    heads = selected_heads if selected_heads is not None else list(range(num_heads_total))
    n_heads = len(heads)

    figs = []
    for kdata in kernels:
        layer_idx = kdata["layer_idx"]
        kernel = kdata["masked_kernel"]  # [num_heads, D, D, Kh, Kw]
        Kh, Kw = kernel.shape[3], kernel.shape[4]

        # Dense spatial sampling
        h_positions = list(range(0, Kh, spatial_step))
        w_positions = list(range(0, Kw, spatial_step))
        positions = [(y, x) for y in h_positions for x in w_positions]
        n_pos = len(positions)

        fig, axes = plt.subplots(n_pos, n_heads, figsize=(1.8 * n_heads, 1.5 * n_pos), squeeze=False)

        # Compute global color range for this layer
        vmax = 0.0
        for y, x in positions:
            for h in heads:
                vmax = max(vmax, kernel[h, :, :, y, x].abs().max().item())

        for pi, (y, x) in enumerate(positions):
            for ci, h in enumerate(heads):
                ax = axes[pi, ci]
                mat = kernel[h, :, :, y, x].numpy()
                ax.imshow(mat, cmap=_CMAP_DIVERGE, vmin=-vmax, vmax=vmax, aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if pi == 0:
                    ax.set_title(f"H{h}", fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f"({y},{x})", fontsize=7)

        fig.suptitle(f"Layer {layer_idx}: Mixing Matrices (64x64) at Spatial Positions", fontsize=11, y=1.02)
        fig.text(
            0.5,
            -0.01,
            "Goal: Show the 64×64 inter-channel mixing matrix at each spatial offset. Each small heatmap shows\n"
            "how the head_dim channels interact at that spatial position. Expect near-center positions to have\n"
            "stronger structure (these dominate after masking) and distant positions to be weaker / noisier.",
            ha="center",
            va="top",
            fontsize=7,
            style="italic",
            color="0.4",
        )
        fig.tight_layout()
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Panel 6: Activation maps (image-conditioned)
# ---------------------------------------------------------------------------


def plot_activation_maps(
    activations: dict[int, dict[str, torch.Tensor]],
    detail_layers: list[int],
) -> plt.Figure:
    """Pre/post conv activation magnitude + log₂ amplification ratio."""
    from matplotlib.colors import TwoSlopeNorm

    layers = [i for i in detail_layers if i in activations]
    n_layers = len(layers)

    fig, axes = plt.subplots(3, n_layers, figsize=(3.5 * n_layers, 8), squeeze=False)
    row_labels = ["Pre-conv |act|", "Post-conv |act|", "log₂(Post/Pre)"]

    for col, li in enumerate(layers):
        pre = activations[li]["pre_conv"]  # [B, C, H, W] or [B, H, W, C]
        post = activations[li]["post_conv"]

        # Handle both BHL and BLH layouts
        if pre.dim() == 4 and pre.shape[1] > pre.shape[-1]:
            # BHL: [B, C, H, W] — channel dim is large
            pre_mag = pre[0].abs().mean(dim=0).numpy()  # [H, W]
            post_mag = post[0].abs().mean(dim=0).numpy()
        else:
            # BLH: [B, H, W, C]
            pre_mag = pre[0].abs().mean(dim=-1).numpy()  # [H, W]
            post_mag = post[0].abs().mean(dim=-1).numpy()  # [H, W]

        # log₂ ratio: centered at 0 (no change), +ve = amplification, -ve = suppression
        eps = 1e-8
        log_ratio = np.log2(np.clip(post_mag, eps, None) / np.clip(pre_mag, eps, None))  # [H, W]

        for row, data in enumerate([pre_mag, post_mag, log_ratio]):
            ax = axes[row, col]
            if row < 2:
                # Magnitude maps: sequential colormap
                im = ax.imshow(data, cmap=_CMAP_SPECTRAL, origin="lower", aspect="auto")
            else:
                # Log-ratio: diverging colormap centered at 0
                abs_max = max(abs(data.min()), abs(data.max()), eps)
                norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
                im = ax.imshow(data, cmap=_CMAP_DIVERGE, norm=norm, origin="lower", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == 0:
                ax.set_title(f"Layer {li}", fontsize=10)

    fig.suptitle("Activation Maps: Effect of Global Convolution", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Show how the global convolution transforms actual image features. Top=input activation magnitude,\n"
        "middle=output, bottom=log₂ amplification ratio (0=no change, +ve=amplification, −ve=suppression).\n"
        "Expect spatially structured patterns where the model amplifies salient regions and suppresses background.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 7: Kernel overlay on image (per head / layer)
# ---------------------------------------------------------------------------


def _denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and convert [1, H, W, 3] channels-last to [H, W, 3] uint8."""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = tensor[0].cpu().float() * std + mean  # [H, W, 3]
    return img.clamp(0, 1).numpy()


def plot_kernel_on_image(
    image_tensor: torch.Tensor,
    all_kernels: list[dict],
    detail_layers: list[int],
    selected_heads: list[int] | None,
    model: torch.nn.Module,
) -> plt.Figure:
    """Overlay kernel norm heatmaps on the input image per head and layer.

    For each head/layer, the Frobenius norm of the kernel (= effective receptive
    field from the center patch) is upsampled to image resolution and shown as a
    semi-transparent heatmap on top of the original image.
    """
    from torch.nn.functional import interpolate

    kernels = [all_kernels[i] for i in detail_layers]
    num_heads_total = kernels[0]["masked_kernel"].shape[0]
    heads = selected_heads if selected_heads is not None else list(range(num_heads_total))
    n_layers = len(kernels)
    n_heads = len(heads)

    # De-normalize image for display
    img_rgb = _denormalize_image(image_tensor)  # [H, W, 3]
    img_h, img_w = img_rgb.shape[:2]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.8 * n_heads, 2.8 * n_layers), squeeze=False)

    for row, kdata in enumerate(kernels):
        norms = kdata["masked_kernel"].norm(dim=(1, 2))  # [num_heads, K_h, K_w]

        for ci, h in enumerate(heads):
            ax = axes[row, ci]

            # Upsample kernel norm map to image resolution
            norm_map = norms[h].unsqueeze(0).unsqueeze(0)  # [1, 1, K_h, K_w]
            norm_up = interpolate(norm_map, size=(img_h, img_w), mode="bilinear", align_corners=False)
            norm_up = norm_up[0, 0].numpy()

            # Normalize to [0, 1] for overlay alpha
            norm_max = norm_up.max()
            if norm_max > 0:
                norm_up = norm_up / norm_max

            # Show image with kernel overlay
            ax.imshow(img_rgb)
            ax.imshow(norm_up, cmap=_CMAP_KERNEL, alpha=0.75, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Head {h}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"Layer {kdata['layer_idx']}", fontsize=9)

    fig.suptitle("Kernel Receptive Field Overlaid on Image", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Show where each head 'looks' relative to the center of the image. The heatmap is the kernel's\n"
        "Frobenius norm (receptive field strength) overlaid on the input. Bright regions receive high weight\n"
        "during convolution. Expect local heads to highlight only the center, global heads to cover more area.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 8: Per-head activation magnitude overlaid on image
# ---------------------------------------------------------------------------


def plot_activation_on_image(
    image_tensor: torch.Tensor,
    activations: dict[int, dict[str, torch.Tensor]],
    detail_layers: list[int],
    selected_heads: list[int] | None,
    num_heads: int,
    head_dim: int,
) -> plt.Figure:
    """Overlay per-head post-conv activation magnitude on the input image.

    For each head/layer, the channel-mean absolute activation after the global
    convolution is upsampled to image resolution and shown as a semi-transparent
    heatmap on top of the original image. This reveals which spatial regions each
    head amplifies for this specific input.
    """
    from torch.nn.functional import interpolate

    layers = [i for i in detail_layers if i in activations]
    heads = selected_heads if selected_heads is not None else list(range(num_heads))
    n_layers = len(layers)
    n_heads = len(heads)

    img_rgb = _denormalize_image(image_tensor)  # [H, W, 3]
    img_h, img_w = img_rgb.shape[:2]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.8 * n_heads, 2.8 * n_layers), squeeze=False)

    for row, li in enumerate(layers):
        post = activations[li]["post_conv"]  # [B, C, H, W] (BHL from CKConvMultiheadND)

        # Split into heads: [B, num_heads, head_dim, H, W]
        B = post.shape[0]
        spatial = post.shape[2:]  # (H, W)
        post_heads = post.view(B, num_heads, head_dim, *spatial)

        for ci, h in enumerate(heads):
            ax = axes[row, ci]

            # Mean absolute activation over head_dim channels → [H_patch, W_patch]
            act_mag = post_heads[0, h].abs().mean(dim=0)  # [H_patch, W_patch]

            # Upsample to image resolution
            act_up = interpolate(
                act_mag.unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

            # Normalize to [0, 1]
            act_max = act_up.max()
            if act_max > 0:
                act_up = act_up / act_max

            ax.imshow(img_rgb)
            ax.imshow(act_up, cmap=_CMAP_KERNEL, alpha=0.75, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Head {h}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"Layer {li}", fontsize=9)

    fig.suptitle("Per-Head Post-Conv Activation on Image", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Show which image regions each head activates after the global convolution. Unlike Panel 7\n"
        "(translation-invariant kernel shape), this is input-dependent — it reveals what each head actually\n"
        "'attends to' for this image. Expect heads to specialize: edges, textures, object parts, background.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 9: PCA kernel structure — "where meets what"
# ---------------------------------------------------------------------------


def plot_kernel_pca(
    all_kernels: list[dict],
    detail_layers: list[int],
    selected_heads: list[int] | None,
    n_components: int = 3,
    n_clusters: int = 4,
) -> plt.Figure:
    """K-Means cluster map of mixing-matrix modes across spatial positions.

    For each head/layer the kernel has shape [head_dim, head_dim, K_h, K_w].
    We flatten each spatial position's mixing matrix into a vector of length
    head_dim², run PCA for dimensionality reduction, then K-Means to find
    dominant mixing modes. Positions are colored by cluster ID (categorical
    colormap) and brightness is modulated by the Frobenius norm.

    Same color  → same channel-mixing mode.
    Different color → different mode (= heads doing different *things*).
    Brightness  → magnitude (= *how strongly* that mode is expressed).
    """
    kernels = [all_kernels[i] for i in detail_layers]
    num_heads_total = kernels[0]["masked_kernel"].shape[0]
    heads = selected_heads if selected_heads is not None else list(range(num_heads_total))
    n_layers = len(kernels)
    n_heads = len(heads)

    # Use tab10 categorical colormap for cluster IDs
    cluster_cmap = plt.cm.get_cmap("tab10", n_clusters)

    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(2.8 * n_heads, 2.8 * n_layers), squeeze=False
    )

    for row, kdata in enumerate(kernels):
        kernel = kdata["masked_kernel"]  # [num_heads, head_dim, head_dim, K_h, K_w]

        for ci, h in enumerate(heads):
            ax = axes[row, ci]
            K = kernel[h]  # [head_dim, head_dim, K_h, K_w]
            hd, _, Kh, Kw = K.shape
            N = Kh * Kw

            # Flatten mixing matrices: [K_h*K_w, head_dim²]
            flat = K.reshape(hd * hd, N).T  # [N, D]

            # PCA for dimensionality reduction before clustering
            flat_centered = flat - flat.mean(dim=0, keepdim=True)  # [N, D]
            U, S, _Vh = torch.linalg.svd(flat_centered, full_matrices=False)
            scores = U[:, :n_components] * S[:n_components]  # [N, n_components]

            # K-Means clustering (pure PyTorch, no sklearn dependency)
            # Initialize centroids via K-Means++ style: first random, rest max-distance
            scores_np = scores.numpy()  # [N, n_components]
            centroids = np.empty((n_clusters, n_components), dtype=np.float32)
            centroids[0] = scores_np[N // 2]  # Start from center spatial position
            for ki in range(1, n_clusters):
                dists = np.min(
                    np.sum((scores_np[:, None, :] - centroids[None, :ki, :]) ** 2, axis=2),
                    axis=1,
                )  # [N]
                centroids[ki] = scores_np[np.argmax(dists)]

            # Run 20 iterations of Lloyd's algorithm
            for _ in range(20):
                dists = np.sum(
                    (scores_np[:, None, :] - centroids[None, :, :]) ** 2, axis=2
                )  # [N, n_clusters]
                labels = np.argmin(dists, axis=1)  # [N]
                for ki in range(n_clusters):
                    mask = labels == ki
                    if mask.any():
                        centroids[ki] = scores_np[mask].mean(axis=0)

            # Map cluster IDs to colors: [K_h, K_w, 4(RGBA)]
            label_map = labels.reshape(Kh, Kw)
            rgba = cluster_cmap(label_map)  # [K_h, K_w, 4]

            # Modulate brightness by Frobenius norm (so low-magnitude positions fade)
            norms = K.norm(dim=(0, 1))  # [K_h, K_w]
            nmax = norms.max()
            if nmax > 0:
                brightness = (norms / nmax).numpy()  # [K_h, K_w]
            else:
                brightness = np.ones((Kh, Kw))
            rgba[:, :, :3] *= brightness[:, :, None]  # modulate RGB by brightness

            ax.imshow(rgba, origin="lower", aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Head {h}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"Layer {kdata['layer_idx']}", fontsize=9)

    # Add a categorical legend for cluster IDs
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=cluster_cmap(i), label=f"Mode {i}") for i in range(n_clusters)]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=7, title="Cluster", title_fontsize=8)

    fig.suptitle("Kernel Clustering: Channel Mixing Modes × Spatial Position", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Reveal *what* each head computes at each spatial offset via K-Means clustering of mixing\n"
        "matrices (PCA-reduced). Each color = a distinct mixing mode; brightness = Frobenius norm magnitude.\n"
        "Same color = same mode; different colors across heads = genuine functional diversity.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 10: Head-Grid Mosaic (spatial filter patterns per channel pair)
# ---------------------------------------------------------------------------


def plot_kernel_slices(
    all_kernels: list[dict],
    detail_layers: list[int],
    selected_heads: list[int] | None,
    n_pairs: int = 6,
) -> plt.Figure:
    """Head-Grid Mosaic: visualize K_h x K_w spatial filter for specific channel pairs.

    For each head/layer, picks a few (output_ch, input_ch) pairs and plots
    the raw K[h, out_ch, in_ch, :, :] spatial filter as a grayscale heatmap.
    This shows the actual learned spatial patterns — are heads learning
    different orientations, frequencies, or spatial structures?

    Pairs shown: first diagonal (ch0,ch0), a few off-diagonal pairs, and the
    channel-mean. Each is independently normalized to [0, 1].
    """
    kernels = [all_kernels[i] for i in detail_layers]
    num_heads_total = kernels[0]["masked_kernel"].shape[0]
    heads = selected_heads if selected_heads is not None else list(range(num_heads_total))
    n_layers = len(kernels)
    n_heads = len(heads)

    fig, axes = plt.subplots(
        n_layers * n_pairs,
        n_heads,
        figsize=(2.2 * n_heads, 1.6 * n_layers * n_pairs),
        squeeze=False,
    )

    for li, kdata in enumerate(kernels):
        kernel = kdata["masked_kernel"]  # [num_heads, head_dim, head_dim, K_h, K_w]
        hd = kernel.shape[1]

        # Dynamically select channel pairs based on spatial energy
        # Compute per-channel-pair energy: ||K[h, i, j, :, :]||_F averaged over heads
        # kernel: [num_heads, head_dim, head_dim, K_h, K_w]
        pair_energy = kernel.norm(dim=(3, 4)).mean(dim=0)  # [head_dim, head_dim]

        pairs: list[tuple[str, tuple[int, int] | None]] = [("mean", None)]

        # Top diagonal pairs by energy
        diag_energies = pair_energy.diagonal()  # [head_dim]
        n_diag = min(2, n_pairs - 1)
        top_diag_idx = torch.argsort(diag_energies, descending=True)[:n_diag]
        for i in top_diag_idx.tolist():
            pairs.append((f"({i},{i})", (i, i)))

        # Top off-diagonal pairs by energy
        off_diag_mask = ~torch.eye(hd, dtype=torch.bool)  # [head_dim, head_dim]
        off_diag_energies = pair_energy.clone()
        off_diag_energies[~off_diag_mask] = -1  # mask out diagonal
        n_off = min(n_pairs - len(pairs), 3)
        flat_idx = torch.argsort(off_diag_energies.flatten(), descending=True)[:n_off]
        for fi in flat_idx.tolist():
            r, c = fi // hd, fi % hd
            pairs.append((f"({r},{c})", (r, c)))

        for ci, h in enumerate(heads):
            K = kernel[h]  # [head_dim, head_dim, K_h, K_w]

            for pi, (label, pair) in enumerate(pairs):
                ax = axes[li * n_pairs + pi, ci]

                if pair is None:
                    # Channel-mean spatial filter
                    s = K.mean(dim=(0, 1))  # [K_h, K_w]
                else:
                    s = K[pair[0], pair[1]]  # [K_h, K_w]

                # Normalize to [0, 1]
                smin, smax = s.min(), s.max()
                if smax > smin:
                    s_norm = ((s - smin) / (smax - smin)).numpy()
                else:
                    s_norm = torch.zeros_like(s).numpy()

                ax.imshow(s_norm, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])

                if ci == 0:
                    ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
                if pi == 0 and li == 0:
                    ax.set_title(f"Head {h}", fontsize=9)
                if pi == 0 and ci == 0:
                    ax.annotate(
                        f"Layer {kdata['layer_idx']}",
                        xy=(-0.3, 0.5),
                        xycoords="axes fraction",
                        fontsize=9,
                        fontweight="bold",
                        ha="right",
                        va="center",
                    )

    fig.suptitle("Head-Grid Mosaic: Spatial Filters per Channel Pair", fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Show the raw K_h×K_w spatial filter for specific (out_ch, in_ch) pairs. Top row = channel-mean.\n"
        "Diagonal pairs (i,i) show self-channel spatial structure; off-diagonal (i,j) show cross-channel patterns.\n"
        "Look for: different heads learning different orientations, frequencies, or receptive field shapes.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 11: Channel Correlation Matrix (intra-head connectivity)
# ---------------------------------------------------------------------------


def plot_channel_correlation(
    all_kernels: list[dict],
    detail_layers: list[int],
    selected_heads: list[int] | None,
) -> plt.Figure:
    """Channel Correlation Matrix: how much cross-channel mixing occurs per head.

    For each head/layer, reduces the kernel [head_dim, head_dim, K_h, K_w] to a
    [head_dim, head_dim] matrix by taking the Frobenius norm over the spatial dims
    (K_h, K_w). The resulting heatmap shows which (output_ch, input_ch) pairs have
    strong spatial filters and which are near-zero.

    Diagonal-dominant → depthwise-like (channels don't mix much).
    Dense/off-diagonal → heavy cross-channel mixing.
    """
    kernels = [all_kernels[i] for i in detail_layers]
    num_heads_total = kernels[0]["masked_kernel"].shape[0]
    heads = selected_heads if selected_heads is not None else list(range(num_heads_total))
    n_layers = len(kernels)
    n_heads = len(heads)

    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(3.0 * n_heads, 3.0 * n_layers), squeeze=False
    )

    for row, kdata in enumerate(kernels):
        kernel = kdata["masked_kernel"]  # [num_heads, head_dim, head_dim, K_h, K_w]

        for ci, h in enumerate(heads):
            ax = axes[row, ci]
            K = kernel[h]  # [head_dim, head_dim, K_h, K_w]

            # Frobenius norm over spatial dims → [head_dim, head_dim]
            corr = K.norm(dim=(2, 3))  # [head_dim, head_dim]

            # Normalize to [0, 1] for consistent colormap
            cmax = corr.max()
            if cmax > 0:
                corr_norm = (corr / cmax).numpy()
            else:
                corr_norm = corr.numpy()

            im = ax.imshow(corr_norm, cmap=_CMAP_KERNEL, vmin=0, vmax=1, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Head {h}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"Layer {kdata['layer_idx']}", fontsize=9)

            # Annotate diagonal vs off-diagonal ratio
            diag_mean = corr.diagonal().mean().item()
            off_diag_mask = ~torch.eye(corr.shape[0], dtype=torch.bool)
            off_diag_mean = corr[off_diag_mask].mean().item()
            ratio = diag_mean / off_diag_mean if off_diag_mean > 0 else float("inf")
            ax.text(
                0.98, 0.02, f"d/o={ratio:.1f}",
                transform=ax.transAxes, fontsize=6,
                ha="right", va="bottom", color="white",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )

    fig.suptitle("Channel Correlation Matrix (Spatial Frobenius Norm)", fontsize=13, y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.6, label="||K[i,j,:,:]||_F (normalized)", pad=0.02)
    fig.text(
        0.5,
        -0.01,
        "Goal: Show intra-head channel connectivity. Each pixel (i,j) = Frobenius norm of the K_h×K_w filter\n"
        "between output channel i and input channel j. Diagonal-dominant = depthwise-like (channels stay separate);\n"
        "dense off-diagonal = heavy cross-channel mixing. d/o ratio = diagonal mean / off-diagonal mean.",
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="0.4",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize learned Hyena kernels from a trained ViT-5 + Multi-Head Hyena model."
    )
    parser.add_argument("--run-path", type=str, required=True, help="W&B run path: 'entity/project/run_id'")
    parser.add_argument(
        "--alias",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Checkpoint alias to download (default: best)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to an ImageNet image for activation visualization (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./kernel_viz_output",
        help="Directory to save output figures (default: ./kernel_viz_output)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer indices or 'all' (default: all)",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="all",
        help="Comma-separated head indices or 'all' (default: all)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures (default: 150)")
    parser.add_argument(
        "--spatial-step",
        type=int,
        default=4,
        help="Step size for spatial sampling in mixing matrix panel (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print("Loading model...")
    model = load_model(None, args.run_path, args.alias, args.device)
    num_blocks = len(model.blocks)

    # Parse selections
    selected_layers = list(range(num_blocks)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    selected_heads = None if args.heads == "all" else [int(x) for x in args.heads.split(",")]

    # 2. Extract kernels
    print("Extracting kernels from all layers...")
    all_kernels = extract_all_kernels(model)
    k0 = all_kernels[0]["masked_kernel"]
    print(f"  Kernel shape per layer: {list(k0.shape)} (num_heads, head_dim, head_dim, K_h, K_w)")

    # Detail layers for panels that don't show all layers
    detail_layers = _select_detail_layers(num_blocks, selected_layers)

    # 3. Generate visualizations
    figures: list[tuple[str, plt.Figure]] = []

    print("Generating Panel 1: Kernel norm heatmaps...")
    fig1 = plot_kernel_norm_heatmaps(all_kernels, selected_layers, selected_heads)
    figures.append(("01_kernel_norms", fig1))

    print("Generating Panel 2: Raw vs masked comparison...")
    fig2 = plot_raw_vs_masked(all_kernels, detail_layers, selected_heads)
    figures.append(("02_raw_vs_masked", fig2))

    print("Generating Panel 3: Gaussian mask receptive fields...")
    fig3 = plot_gaussian_masks(all_kernels, detail_layers, selected_heads)
    figures.append(("03_gaussian_masks", fig3))

    print("Generating Panel 4: Spectral analysis...")
    fig4 = plot_spectral_analysis(all_kernels, detail_layers, selected_heads)
    figures.append(("04_spectral_analysis", fig4))

    print("Generating Panel 5: Mixing matrices...")
    mixing_figs = plot_mixing_matrices(all_kernels, detail_layers, selected_heads, spatial_step=args.spatial_step)
    for i, mfig in enumerate(mixing_figs):
        layer_idx = all_kernels[detail_layers[i]]["layer_idx"]
        figures.append((f"05_mixing_matrices_layer{layer_idx}", mfig))

    print("Generating Panel 9: Kernel PCA (where × what)...")
    fig9 = plot_kernel_pca(all_kernels, detail_layers, selected_heads)
    figures.append(("09_kernel_pca", fig9))

    print("Generating Panel 10: Head-grid mosaic...")
    fig10 = plot_kernel_slices(all_kernels, detail_layers, selected_heads)
    figures.append(("10_head_grid_mosaic", fig10))

    print("Generating Panel 11: Channel correlation matrix...")
    fig11 = plot_channel_correlation(all_kernels, detail_layers, selected_heads)
    figures.append(("11_channel_correlation", fig11))

    # 4. Optional: image-conditioned panels
    if args.image_path is not None:
        print(f"Forwarding image: {args.image_path}")
        image_tensor = load_and_preprocess_image(args.image_path)
        activations = forward_with_hooks(model, image_tensor, args.device)

        print("Generating Panel 6: Activation maps...")
        fig6 = plot_activation_maps(activations, detail_layers)
        figures.append(("06_activation_maps", fig6))

        print("Generating Panel 7: Kernel overlay on image...")
        fig7 = plot_kernel_on_image(image_tensor, all_kernels, detail_layers, selected_heads, model)
        figures.append(("07_kernel_on_image", fig7))

        print("Generating Panel 8: Per-head activation on image...")
        k0 = all_kernels[0]["masked_kernel"]
        fig8 = plot_activation_on_image(
            image_tensor, activations, detail_layers, selected_heads,
            num_heads=k0.shape[0], head_dim=k0.shape[1],
        )
        figures.append(("08_activation_on_image", fig8))

    # 5. Save
    for name, fig in figures:
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved {path}")

    pdf_path = output_dir / "hyena_kernel_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    print(f"  Saved PDF report: {pdf_path}")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
