#!/usr/bin/env python
"""Compute per-channel mean and standard deviation for an ImageNet dataset copy.

The script replicates the ImageNet diffusion preprocessing (resize -> crop -> optional
downsample -> tensor conversion) but skips normalization so we can derive the statistics
needed for `transforms.Normalize`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def _build_transform(
    *,
    image_size: int,
    final_image_size: int,
    center_crop: bool,
) -> transforms.Compose:
    resize_op = transforms.Resize(image_size + 32)
    crop_op = transforms.CenterCrop(image_size) if center_crop else transforms.Resize(image_size)
    downsample_ops: list[transforms.Transform] = []
    if final_image_size != image_size:
        downsample_ops.append(transforms.Resize(final_image_size, interpolation=InterpolationMode.BICUBIC))
    return transforms.Compose([resize_op, crop_op, *downsample_ops, transforms.ToTensor()])


def compute_stats(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    cache_dir: Path,
    hf_token: Optional[str],
    image_size: int,
    final_image_size: int,
    center_crop: bool,
    max_images: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    ds = load_dataset(
        path=dataset_name,
        name=dataset_config,
        split=split,
        streaming=False,
        cache_dir=str(cache_dir),
        token=hf_token,
    )
    transform = _build_transform(
        image_size=image_size,
        final_image_size=final_image_size,
        center_crop=center_crop,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0.0

    for idx, example in enumerate(tqdm(ds, desc=f"Processing {split}")):
        image = example["image"].convert("RGB")
        tensor = transform(image)
        c, h, w = tensor.shape
        pixels = h * w
        flat = tensor.view(c, -1)
        channel_sum += flat.sum(dim=1).double()
        channel_sum_sq += (flat**2).sum(dim=1).double()
        total_pixels += pixels

        if max_images is not None and (idx + 1) >= max_images:
            break

    if total_pixels == 0:
        raise RuntimeError("No images processed; check dataset path and filters.")

    mean = channel_sum / total_pixels
    variance = channel_sum_sq / total_pixels - mean**2
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    return mean.float(), std.float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ImageNet per-channel mean/std.")
    parser.add_argument("--dataset-name", default="imagenet-1k")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--final-image-size", type=int, default=256)
    parser.add_argument("--center-crop", action="store_true", default=True)
    parser.add_argument("--no-center-crop", dest="center_crop", action="store_false")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of samples (for smoke tests).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mean, std = compute_stats(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir.expanduser(),
        hf_token=args.hf_token,
        image_size=args.image_size,
        final_image_size=args.final_image_size,
        center_crop=args.center_crop,
        max_images=args.max_images,
    )
    print("Per-channel mean:", mean.tolist())
    print("Per-channel std: ", std.tolist())


if __name__ == "__main__":
    main()
