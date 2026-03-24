#!/usr/bin/env python
"""Export FiLM-Hyena kernel data for interactive visualization.

Downloads a checkpoint from W&B (or loads a local .ckpt), loads the FiLM-Hyena
model (EMA weights), runs validation images through it, and saves all kernel
data to a .npz file.

Usage (inside srun):
    # From W&B (default: original FiLM model):
    srun --gres=gpu:1 ... bash -c '
      source .env && conda activate nv-subq &&
      PYTHONPATH=. python scripts/export_film_kernels.py'

    # Local checkpoint with custom config:
    srun --gres=gpu:1 ... bash -c '
      source .env && conda activate nv-subq &&
      PYTHONPATH=. python scripts/export_film_kernels.py \
        --local-ckpt runs/.../checkpoints/last.ckpt \
        --config examples.vit5_imagenet.v3.vit5_small_pretrain_hyena_cls_row_gated_film_posemb_ema \
        --output outputs/film_kernel_viz/kernel_data_posemb.npz \
        --use-ema'
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
from nvsubquadratic.modules.film import RegisterPooling  # noqa: E402
from nvsubquadratic.modules.kernels_nd import SIRENKernelND  # noqa: E402


def _rmsnorm_forward_safe(self, x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch RMSNorm forward (avoids quack stride-alignment errors)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.eps)
    return (self.weight * x).to(input_dtype)


_rms_norm_module.RMSNorm.forward = _rmsnorm_forward_safe

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_WANDB_RUN_PATH = "implicit-long-convs/nvsubquadratic/peeaqdkq"
DEFAULT_CHECKPOINT_ALIAS = "best"
DEFAULT_CONFIG_MODULE = "examples.vit5_imagenet.v3.vit5_small_pretrain_hyena_cls_row_gated_film_ema"
DEFAULT_OUTPUT = "outputs/film_kernel_viz/kernel_data.npz"

IMAGENET_VAL_DIR = "/shared/data/image_datasets/imagenet_folder/val"
NUM_IMAGES = 20

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
KERNEL_SEQ_LENS = (15, 14)


def parse_args():
    parser = argparse.ArgumentParser(description="Export FiLM-Hyena kernels")
    parser.add_argument("--wandb-run", default=DEFAULT_WANDB_RUN_PATH, help="W&B run path (entity/project/run_id)")
    parser.add_argument("--alias", default=DEFAULT_CHECKPOINT_ALIAS, help="W&B artifact alias (best/latest)")
    parser.add_argument("--local-ckpt", default=None, help="Path to a local .ckpt file (skips W&B download)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_MODULE, help="Python config module path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .npz path (relative to project root)")
    parser.add_argument("--use-ema", action="store_true", help="Load ema_network weights instead of network weights")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    return parser.parse_args()


def load_config(config_module: str):
    mod = importlib.import_module(config_module)
    return mod.get_config()


def build_and_load_model(config, args):
    network = instantiate(config.net)
    network.eval()

    if args.local_ckpt:
        ckpt_path = args.local_ckpt
        print(f"[info] Using local checkpoint: {ckpt_path}")
    else:
        ckpt_path = download_checkpoint(run_path=args.wandb_run, alias=args.alias)
    raw_sd = load_checkpoint_state_dict(ckpt_path)
    print(f"[info] Loaded state_dict ({len(raw_sd)} keys).")

    net_prefix = "ema_network." if args.use_ema else "network."
    compile_prefix = "_orig_mod."
    sd = {}
    for k, v in raw_sd.items():
        if not k.startswith(net_prefix):
            continue
        key = k[len(net_prefix) :]
        if key.startswith(compile_prefix):
            key = key[len(compile_prefix) :]
        sd[key] = v

    if not sd:
        raise ValueError(
            f"No keys found with prefix '{net_prefix}'. Available prefixes: {{k.split('.')[0] for k in raw_sd.keys()}}"
        )
    print(f"[info] Extracted {len(sd)} keys with prefix '{net_prefix}'.")

    sd = align_compiled_keys(sd, set(network.state_dict().keys()))
    network.load_state_dict(sd, strict=True)
    print("[info] Model loaded successfully.")
    return network


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_diverse_images(val_dir: str, n: int, transform):
    val_path = Path(val_dir)
    class_dirs = sorted([d for d in val_path.iterdir() if d.is_dir()])
    step = max(1, len(class_dirs) // n)
    selected_dirs = class_dirs[::step][:n]

    images, labels, raw_images = [], [], []
    for cls_dir in selected_dirs:
        img_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not img_files:
            continue
        img = Image.open(img_files[0]).convert("RGB")
        images.append(transform(img))
        labels.append(cls_dir.name)
        # Save a thumbnail for display
        thumb = img.resize((112, 112), Image.BICUBIC)
        raw_images.append(np.array(thumb))

    print(f"[info] Loaded {len(images)} images from classes: {labels}")
    return images, labels, raw_images


def get_siren_modules(network):
    return [(name, mod) for name, mod in network.named_modules() if isinstance(mod, SIRENKernelND)]


def get_register_pooling_modules(network):
    return [(name, mod) for name, mod in network.named_modules() if isinstance(mod, RegisterPooling)]


def extract_static_kernels(network, device):
    siren_modules = get_siren_modules(network)
    static = {}
    for idx, (name, mod) in enumerate(siren_modules):
        mod = mod.to(device)
        with torch.no_grad():
            kernel, _ = mod(KERNEL_SEQ_LENS, conditioning=None)
        static[idx] = kernel.detach().cpu().numpy()
        print(f"[static] Block {idx}: {kernel.shape}")
    return static


def extract_register_weights(network):
    pooling_modules = get_register_pooling_modules(network)
    weights = {}
    for idx, (name, mod) in enumerate(pooling_modules):
        logits = mod.logits.detach()
        w = F.softmax(logits, dim=0).numpy()
        weights[idx] = w
        print(f"[regpool] Block {idx}: {w.shape}, max_reg={w.argmax()}, max_w={w.max():.3f}")
    return weights


class KernelCaptureHook:
    def __init__(self):
        self.kernels: dict[int, np.ndarray] = {}
        self._handles = []

    def register(self, network):
        siren_modules = get_siren_modules(network)
        for idx, (name, mod) in enumerate(siren_modules):
            self._handles.append(mod.register_forward_hook(self._make_hook(idx)))
        print(f"[hook] Registered {len(self._handles)} hooks.")

    def _make_hook(self, block_idx):
        def hook_fn(module, input, output):
            kernel, _ = output
            self.kernels[block_idx] = kernel.detach().cpu().numpy()

        return hook_fn

    def clear(self):
        self.kernels.clear()

    def remove_all(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def run_inference(network, images, device):
    hook = KernelCaptureHook()
    hook.register(network)
    network = network.to(device)
    print(f"[infer] Running on device: {device}")

    # conditioned[block_idx] will be a list of arrays, one per image
    conditioned = {i: [] for i in range(12)}
    predictions = []

    for img_idx, img_tensor in enumerate(images):
        hook.clear()
        x = img_tensor.unsqueeze(0).to(device).permute(0, 2, 3, 1)

        with torch.no_grad():
            out = network({"input": x})

        pred = out["logits"].argmax(dim=-1).item()
        predictions.append(pred)

        for block_idx, kernel_np in hook.kernels.items():
            conditioned[block_idx].append(kernel_np)

        print(f"[infer] Image {img_idx}: predicted class {pred}")

    hook.remove_all()

    # Stack: conditioned[block_idx] -> [n_images, H, W, out_dim]
    for block_idx in conditioned:
        conditioned[block_idx] = np.concatenate(conditioned[block_idx], axis=0)

    return conditioned, predictions


def main():
    args = parse_args()
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FiLM-Hyena Kernel Data Export")
    print(f"  config:  {args.config}")
    print(f"  ckpt:    {args.local_ckpt or f'{args.wandb_run} ({args.alias})'}")
    print(f"  ema:     {args.use_ema}")
    print(f"  output:  {output_path}")
    print("=" * 60)

    print("\n[1/5] Loading config and model...")
    config = load_config(args.config)
    network = build_and_load_model(config, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[2/5] Loading validation images...")
    transform = get_val_transform()
    images, labels, thumbnails = load_diverse_images(IMAGENET_VAL_DIR, args.num_images, transform)

    print("\n[3/5] Extracting static kernels...")
    static_kernels = extract_static_kernels(network, device)

    print("\n[4/5] Extracting register pooling weights...")
    reg_weights = extract_register_weights(network)

    print("\n[5/5] Running inference (conditioned kernels)...")
    conditioned_kernels, predictions = run_inference(network, images, device)

    # Pack into npz
    data = {
        "labels": np.array(labels),
        "predictions": np.array(predictions),
        "thumbnails": np.stack(thumbnails),  # [n_images, 112, 112, 3]
    }
    for block_idx in range(12):
        data[f"static_{block_idx}"] = static_kernels[block_idx]  # [1, H, W, C]
        data[f"cond_{block_idx}"] = conditioned_kernels[block_idx]  # [n_images, H, W, C]
        data[f"regw_{block_idx}"] = reg_weights[block_idx]  # [13]

    np.savez_compressed(str(output_path), **data)
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved to {output_path} ({file_size_mb:.1f} MB)")
    print(f"Contents: {len(data)} arrays, {len(images)} images, 12 blocks")


if __name__ == "__main__":
    main()
