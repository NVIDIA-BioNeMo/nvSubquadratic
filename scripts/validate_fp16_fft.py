#!/usr/bin/env python
"""Validate a Hyena-GAP checkpoint on ImageNet with f32 vs fp16 FFT.

Runs the full ImageNet-1k validation set through the model twice:
once with standard f32 FFT and once with fp16 FFT (power-of-2 padding
+ ortho normalization).  Reports top-1 accuracy for each.

Usage:
    PYTHONPATH=. python scripts/validate_fp16_fft.py \
        --checkpoint <path/to/checkpoint.ckpt>
"""

import argparse
import importlib
import time
from collections import OrderedDict

import torch
import torchvision.transforms as T
import torchvision.datasets as datasets

from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.ops.fftconv import fftconv2d_bhl, fftconv2d_bhl_w_reshape
from nvsubquadratic.ops.fftconv_fp16 import fftconv2d_fp16_bhl, fftconv2d_fp16_bhl_w_reshape


def load_config():
    spec = importlib.util.spec_from_file_location(
        "config",
        "examples/vit5_imagenet/v2/vit5_small_pretrain_hyena_gap_apex.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def build_network(cfg):
    from nvsubquadratic.lazy_config import instantiate
    return instantiate(cfg.net)


def load_checkpoint(net, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        key = k
        for prefix in ("network._orig_mod.", "network.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        new_sd[key] = v
    net.load_state_dict(new_sd, strict=True)
    return net


def set_fp16_fft(net, enabled: bool):
    count = 0
    for module in net.modules():
        if isinstance(module, CKConvND):
            module.use_fp16_fft = enabled
            if enabled:
                module.fftconv_fn = fftconv2d_fp16_bhl_w_reshape
                module.fftconv_fn_bhl_input = fftconv2d_fp16_bhl
            else:
                module.fftconv_fn = fftconv2d_bhl_w_reshape
                module.fftconv_fn_bhl_input = fftconv2d_bhl
            count += 1
    return count


def get_val_loader(data_dir, batch_size=256, num_workers=8):
    """Standard ImageNet validation loader (center-crop 224)."""
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def validate(net, loader, device="cuda", max_batches=None):
    """Run validation and return top-1 accuracy."""
    net.eval()
    correct = 0
    total = 0

    start = time.perf_counter()
    for i, (images, targets) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        # Model expects channels-last [B, H, W, C]
        images = images.to(device).permute(0, 2, 3, 1)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = net({"input": images, "condition": targets})

        logits = out["logits"]
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(loader) if not max_batches else max_batches}] "
                  f"acc={correct/total:.4f} ({correct}/{total})")

    elapsed = time.perf_counter() - start
    accuracy = correct / total
    return accuracy, total, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-dir", type=str, default="/shared/data/image_datasets/imagenet_folder/val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit number of val batches (for quick testing)")
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val dir: {args.val_dir}")
    print()

    # Build model
    print("Loading config and building network...")
    cfg = load_config()
    net = build_network(cfg)
    print("Loading checkpoint...")
    net = load_checkpoint(net, args.checkpoint)
    net = net.to(device).eval()

    # Create val loader
    print("Creating validation loader...")
    loader = get_val_loader(args.val_dir, args.batch_size, args.num_workers)
    n_batches = len(loader) if not args.max_batches else min(args.max_batches, len(loader))
    print(f"  {len(loader.dataset)} images, {n_batches} batches of {args.batch_size}")

    # ─── F32 FFT ─────────────────────────────────────────────────────────
    print("\n=== Validation with F32 FFT ===")
    set_fp16_fft(net, enabled=False)
    acc_f32, total_f32, time_f32 = validate(net, loader, device, args.max_batches)
    print(f"\n  Top-1 accuracy: {acc_f32:.4f} ({acc_f32*100:.2f}%)")
    print(f"  Time: {time_f32:.1f}s  ({total_f32/time_f32:.1f} img/s)")

    # ─── FP16 FFT ────────────────────────────────────────────────────────
    print("\n=== Validation with FP16 FFT ===")
    n = set_fp16_fft(net, enabled=True)
    print(f"  Enabled fp16 FFT on {n} CKConvND modules")
    acc_fp16, total_fp16, time_fp16 = validate(net, loader, device, args.max_batches)
    print(f"\n  Top-1 accuracy: {acc_fp16:.4f} ({acc_fp16*100:.2f}%)")
    print(f"  Time: {time_fp16:.1f}s  ({total_fp16/time_fp16:.1f} img/s)")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  F32 FFT accuracy:  {acc_f32*100:.2f}%")
    print(f"  FP16 FFT accuracy: {acc_fp16*100:.2f}%")
    print(f"  Difference:        {(acc_fp16 - acc_f32)*100:+.2f}%")
    print(f"  Speed:             {time_f32/time_fp16:.3f}x")


if __name__ == "__main__":
    main()
