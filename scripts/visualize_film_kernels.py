#!/usr/bin/env python
"""Visualize input-dependent kernels produced by FiLM-Hyena.

Downloads a checkpoint from W&B, loads the FiLM-Hyena model (EMA weights),
runs a few ImageNet validation images through it with hooks on every
SIRENKernelND, and produces three figure panels:

  A. Static (unconditioned) SIREN kernels — no FiLM, per-block normalization.
  B. Conditioned kernels per image — all 12 blocks, input images on top.
  C. Difference maps (conditioned − static) — what FiLM adds, per image/block.

Usage (inside an srun interactive job):
    srun --gres=gpu:1 ... bash -c '
      source .env && conda activate nv-subq &&
      PYTHONPATH=. python scripts/visualize_film_kernels.py'
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.checkpointing import (  # noqa: E402
    align_compiled_keys,
    download_checkpoint,
    load_checkpoint_state_dict,
)
from nvsubquadratic.lazy_config import instantiate  # noqa: E402
from nvsubquadratic.modules import rms_norm as _rms_norm_module  # noqa: E402
from nvsubquadratic.modules.kernels_nd import SIRENKernelND  # noqa: E402


def _rmsnorm_forward_safe(self, x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch RMSNorm forward (avoids quack stride-alignment errors)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.eps)
    return (self.weight * x).to(input_dtype)


_rms_norm_module.RMSNorm.forward = _rmsnorm_forward_safe

# ── Config ───────────────────────────────────────────────────────────────────
WANDB_RUN_PATH = "implicit-long-convs/nvsubquadratic/peeaqdkq"
CHECKPOINT_ALIAS = "best"
CONFIG_MODULE = "examples.vit5_imagenet.v3.vit5_small_pretrain_hyena_cls_row_gated_film_ema"
IMAGENET_VAL_DIR = "/shared/data/image_datasets/imagenet_folder/val"
NUM_IMAGES = 8
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "film_kernel_viz"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
KERNEL_SEQ_LENS = (15, 14)  # CLS-row grid: 15 rows x 14 cols


def load_config():
    """Import and return the FiLM-Hyena ExperimentConfig."""
    mod = importlib.import_module(CONFIG_MODULE)
    return mod.get_config()


def build_and_load_model(config):
    """Build the network and load EMA weights from the W&B checkpoint."""
    network = instantiate(config.net)
    network.eval()

    ckpt_path = download_checkpoint(run_path=WANDB_RUN_PATH, alias=CHECKPOINT_ALIAS)
    raw_sd = load_checkpoint_state_dict(ckpt_path)
    print(f"[info] Using EMA weights from 'state_dict' ({len(raw_sd)} keys).")

    net_prefix = "network."
    compile_prefix = "_orig_mod."
    sd = {}
    for k, v in raw_sd.items():
        key = k
        if key.startswith(net_prefix):
            key = key[len(net_prefix) :]
        if key.startswith(compile_prefix):
            key = key[len(compile_prefix) :]
        sd[key] = v

    sd = align_compiled_keys(sd, set(network.state_dict().keys()))
    network.load_state_dict(sd, strict=True)
    print("[info] Model loaded successfully.")
    return network


def get_val_transform():
    """Validation transform matching the DALI pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_diverse_images(val_dir: str, n: int, transform):
    """Load n images from evenly-spaced classes in the validation set."""
    val_path = Path(val_dir)
    class_dirs = sorted([d for d in val_path.iterdir() if d.is_dir()])
    step = max(1, len(class_dirs) // n)
    selected_dirs = class_dirs[::step][:n]

    images, labels, paths = [], [], []
    for cls_dir in selected_dirs:
        img_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not img_files:
            continue
        img_path = img_files[0]
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        images.append(tensor)
        labels.append(cls_dir.name)
        paths.append(str(img_path))

    print(f"[info] Loaded {len(images)} images from classes: {labels}")
    return images, labels, paths


# ── Kernel extraction ────────────────────────────────────────────────────────


def get_siren_modules(network):
    """Return an ordered list of (name, SIRENKernelND) from the network."""
    return [(name, mod) for name, mod in network.named_modules() if isinstance(mod, SIRENKernelND)]


def extract_static_kernels(network, device="cpu"):
    """Call each SIRENKernelND directly with conditioning=None (no FiLM).

    Returns:
        dict[block_idx -> kernel tensor [1, H, W, out_dim]]
    """
    siren_modules = get_siren_modules(network)
    static_kernels = {}
    for idx, (name, mod) in enumerate(siren_modules):
        mod = mod.to(device)
        with torch.no_grad():
            kernel, _grid = mod(KERNEL_SEQ_LENS, conditioning=None)
        static_kernels[idx] = kernel.detach().cpu()
        print(f"[static] Block {idx}: kernel shape {tuple(kernel.shape)}")
    return static_kernels


class KernelCaptureHook:
    """Forward hook that captures SIRENKernelND outputs."""

    def __init__(self):
        self.kernels: dict[int, torch.Tensor] = {}
        self._handles = []

    def register(self, network):
        """Register hooks on all SIRENKernelND modules."""
        siren_modules = get_siren_modules(network)
        for idx, (name, mod) in enumerate(siren_modules):
            handle = mod.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)
            print(f"[hook] Registered on block {idx}: {name}")
        print(f"[hook] Total hooks: {len(self._handles)}")

    def _make_hook(self, block_idx):
        def hook_fn(module, input, output):
            kernel, grid = output
            self.kernels[block_idx] = kernel.detach().cpu()

        return hook_fn

    def clear(self):
        self.kernels.clear()

    def remove_all(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def run_inference(network, images, device="cpu"):
    """Run each image through the model, capturing kernels per block per image."""
    hook = KernelCaptureHook()
    hook.register(network)
    network = network.to(device)
    print(f"[infer] Running on device: {device}")

    all_kernels = {}
    logits_list = []

    for img_idx, img_tensor in enumerate(images):
        hook.clear()
        x = img_tensor.unsqueeze(0).to(device)
        x = x.permute(0, 2, 3, 1)  # [1, H, W, C] channels-last

        with torch.no_grad():
            out = network({"input": x})

        logits_list.append(out["logits"].cpu())
        for block_idx, kernel in hook.kernels.items():
            all_kernels[(block_idx, img_idx)] = kernel

        pred = out["logits"].argmax(dim=-1).item()
        print(f"[infer] Image {img_idx}: predicted class {pred}")

    hook.remove_all()
    return all_kernels, logits_list


# ── Visualization helpers ────────────────────────────────────────────────────


def kernel_to_heatmap(kernel: torch.Tensor) -> np.ndarray:
    """Convert kernel [1, H, W, out_dim] to channel-mean heatmap [H, W]."""
    return kernel[0].mean(dim=-1).numpy()


def denorm_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Denormalize a [C, H, W] tensor to a [H, W, 3] uint8 numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_panel_a(static_kernels, output_dir):
    """Panel A: unconditioned SIREN kernels (no FiLM), per-block normalization."""
    num_blocks = len(static_kernels)
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle("Static SIREN Kernels (no FiLM conditioning, channel mean)", fontsize=18, y=0.98)

    for block_idx in range(num_blocks):
        r, c = divmod(block_idx, cols)
        ax = axes[r][c]
        hm = kernel_to_heatmap(static_kernels[block_idx])
        abs_max = max(abs(hm.min()), abs(hm.max()))
        abs_max = max(abs_max, 1e-8)
        im = ax.imshow(hm, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
        ax.set_title(f"Block {block_idx}  (max={abs_max:.4f})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "panel_a_static_kernels.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] Panel A -> {out_path}")


def plot_panel_b(all_kernels, images, labels, output_dir):
    """Panel B: conditioned kernels per image. Top row = input images, then 12 block rows."""
    n_images = len(images)
    num_blocks = max(b for (b, _) in all_kernels.keys()) + 1
    n_rows = 1 + num_blocks  # image row + kernel rows

    fig, axes = plt.subplots(n_rows, n_images, figsize=(3 * n_images, 2.5 * n_rows))
    fig.suptitle("FiLM-Conditioned Kernels per Image (channel mean, per-block normalization)", fontsize=16, y=1.01)

    # Top row: input images
    for col in range(n_images):
        ax = axes[0][col]
        ax.imshow(denorm_image(images[col]))
        ax.set_title(labels[col], fontsize=9)
        ax.axis("off")

    # Kernel rows
    for block_idx in range(num_blocks):
        row = block_idx + 1
        row_heatmaps = []
        for img_idx in range(n_images):
            key = (block_idx, img_idx)
            if key in all_kernels:
                row_heatmaps.append(kernel_to_heatmap(all_kernels[key]))
            else:
                row_heatmaps.append(None)

        valid = [h for h in row_heatmaps if h is not None]
        abs_max = max((max(abs(h.min()), abs(h.max())) for h in valid), default=1.0)
        abs_max = max(abs_max, 1e-8)

        for col, hm in enumerate(row_heatmaps):
            ax = axes[row][col]
            if hm is not None:
                ax.imshow(hm, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
            if col == 0:
                ax.set_ylabel(f"Block {block_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out_path = output_dir / "panel_b_conditioned_kernels.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] Panel B -> {out_path}")


def plot_panel_c(all_kernels, static_kernels, images, labels, output_dir):
    """Panel C: difference (conditioned - static) per image/block. Top row = images."""
    n_images = len(images)
    num_blocks = len(static_kernels)
    n_rows = 1 + num_blocks

    fig, axes = plt.subplots(n_rows, n_images, figsize=(3 * n_images, 2.5 * n_rows))
    fig.suptitle(
        "FiLM Effect: Conditioned − Static Kernel (channel mean, per-block normalization)", fontsize=16, y=1.01
    )

    # Top row: input images
    for col in range(n_images):
        ax = axes[0][col]
        ax.imshow(denorm_image(images[col]))
        ax.set_title(labels[col], fontsize=9)
        ax.axis("off")

    # Difference rows
    for block_idx in range(num_blocks):
        row = block_idx + 1
        static_hm = kernel_to_heatmap(static_kernels[block_idx])

        diff_heatmaps = []
        for img_idx in range(n_images):
            key = (block_idx, img_idx)
            if key in all_kernels:
                diff_heatmaps.append(kernel_to_heatmap(all_kernels[key]) - static_hm)
            else:
                diff_heatmaps.append(None)

        valid = [d for d in diff_heatmaps if d is not None]
        abs_max = max((max(abs(d.min()), abs(d.max())) for d in valid), default=1.0)
        abs_max = max(abs_max, 1e-8)

        for col, diff in enumerate(diff_heatmaps):
            ax = axes[row][col]
            if diff is not None:
                ax.imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
            if col == 0:
                ax.set_ylabel(f"Block {block_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out_path = output_dir / "panel_c_film_effect.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] Panel C -> {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FiLM-Hyena Kernel Visualization")
    print("=" * 60)

    print("\n[1/5] Loading config and model...")
    config = load_config()
    network = build_and_load_model(config)

    print("\n[2/5] Loading validation images...")
    transform = get_val_transform()
    images, labels, paths = load_diverse_images(IMAGENET_VAL_DIR, NUM_IMAGES, transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[3/5] Extracting static (unconditioned) kernels...")
    static_kernels = extract_static_kernels(network, device=device)

    print("\n[4/5] Running inference with kernel hooks...")
    all_kernels, logits = run_inference(network, images, device=device)

    print(f"\n[5/5] Generating visualizations ({len(all_kernels)} kernel snapshots)...")
    plot_panel_a(static_kernels, OUTPUT_DIR)
    plot_panel_b(all_kernels, images, labels, OUTPUT_DIR)
    plot_panel_c(all_kernels, static_kernels, images, labels, OUTPUT_DIR)

    print(f"\nDone! All figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
