"""Head-to-head benchmark: ImageFolder vs DALI single-GPU throughput.

Compares pure data-loading throughput (img/s) and per-batch latency for:
  1. ImageFolder + torchvision transforms (CPU decode + CPU augment)
  2. DALI (GPU decode via nvJPEG + GPU augment)

Both pipelines mirror the reference training recipe from
``examples/vit5_imagenet/vit5_small_pretrain_apex.py``:
  Resize(image_size + 32, BICUBIC) -> RandomCrop(image_size)
  -> RandomHorizontalFlip -> [ColorJitter + ThreeAugment] -> Normalize

Usage:
    PYTHONPATH=. python scripts/benchmark_dali_vs_folder.py \\
        --imagefolder-dir /local_scratch/$USER/imagenet_folder \\
        --batch-size 256 --num-workers 14 --num-batches 200

    # With full training augmentations:
    PYTHONPATH=. python scripts/benchmark_dali_vs_folder.py \\
        --imagefolder-dir /local_scratch/$USER/imagenet_folder \\
        --batch-size 256 --num-workers 14 --num-batches 200 \\
        --three-augment --color-jitter 0.3
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from experiments.datamodules.imagenet import ThreeAugment

# DALI imports (optional)
try:
    from experiments.datamodules.imagenet_dali import (
        DALI_AVAILABLE,  # False when nvidia-dali is not installed
        DALIIteratorWrapper,
        _build_train_pipeline,
        _IMAGENET_MEAN,
        _IMAGENET_STD,
    )
except ImportError:
    DALI_AVAILABLE = False


def benchmark_loader(loader, name: str, num_batches: int, device) -> dict:
    """Time ``num_batches`` iterations and report throughput statistics."""
    print(f"\n{'─' * 60}")
    print(f" Benchmarking: {name}")
    print(f"{'─' * 60}")

    # Warm-up: 5 batches
    iterator = iter(loader)
    for _ in range(min(5, num_batches)):
        try:
            _ = next(iterator)
        except StopIteration:
            break
    torch.cuda.synchronize()

    # Timed run
    latencies = []
    samples_total = 0
    iterator = iter(loader)
    t0 = time.perf_counter()

    for i in range(num_batches):
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            print(f"  [!] Dataset exhausted after {i} batches.")
            break

        # Parse batch
        if isinstance(batch, (tuple, list)):
            images, labels = batch
        elif isinstance(batch, dict):
            images = batch.get("images", batch.get("input"))
            labels = batch.get("labels", batch.get("label"))
        else:
            images = batch

        # Ensure tensor is on GPU for fair comparison
        if isinstance(images, torch.Tensor) and not images.is_cuda:
            images = images.to(device, non_blocking=True)
        torch.cuda.synchronize()

        batch_time = time.perf_counter() - batch_start
        latencies.append(batch_time)
        bs = images.shape[0] if isinstance(images, torch.Tensor) else 0
        samples_total += bs

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i + 1}/{num_batches}] {samples_total / elapsed:.1f} img/s  (last batch {batch_time * 1e3:.1f} ms)")

    total_time = time.perf_counter() - t0
    img_per_sec = samples_total / total_time if total_time > 0 else 0

    import statistics

    mean_lat = statistics.mean(latencies) * 1e3
    p50_lat = statistics.median(latencies) * 1e3
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))] * 1e3 if latencies else 0

    print(f"\n  Results for {name}:")
    print(f"    Throughput:   {img_per_sec:>8.1f} img/s")
    print(f"    Mean latency: {mean_lat:>8.2f} ms/batch")
    print(f"    P50 latency:  {p50_lat:>8.2f} ms/batch")
    print(f"    P95 latency:  {p95_lat:>8.2f} ms/batch")
    print(f"    Total images: {samples_total}")
    print(f"    Total time:   {total_time:.2f}s")

    return {
        "name": name,
        "img_per_sec": img_per_sec,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "total_images": samples_total,
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="DALI vs ImageFolder throughput benchmark")
    parser.add_argument("--imagefolder-dir", type=str, required=True,
                        help="Path to ImageFolder root (train/ & val/ subdirs)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=14)
    parser.add_argument("--num-batches", type=int, default=200)
    # image_size / final_image_size match the training config conventions:
    #   image_size        = intermediate crop size (pipeline uses Resize(image_size+32) -> Crop(image_size))
    #   final_image_size  = model input size (optional extra resize after crop)
    parser.add_argument("--image-size", type=int, default=224,
                        help="Intermediate crop size. Pipeline: Resize(image_size+32) -> RandomCrop(image_size)")
    parser.add_argument("--final-image-size", type=int, default=224,
                        help="Final model input size. Extra resize applied when != image_size.")
    parser.add_argument("--three-augment", action="store_true",
                        help="Add ColorJitter + ThreeAugment (Grayscale/Solarize/GaussianBlur) — matches training")
    parser.add_argument("--color-jitter", type=float, default=0.3,
                        help="ColorJitter strength (used only with --three-augment)")
    parser.add_argument("--skip-imagefolder", action="store_true",
                        help="Skip the ImageFolder/torchvision baseline (use when results are already known)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augment_label = "ThreeAugment" if args.three_augment else "basic"
    print(f"Device:          {device}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Workers/threads: {args.num_workers}")
    print(f"Batches:         {args.num_batches}")
    print(f"image_size:      {args.image_size}  (Resize {args.image_size + 32} -> Crop {args.image_size})")
    print(f"final_image_size:{args.final_image_size}")
    print(f"Augmentations:   {augment_label}")

    results = []

    train_dir = os.path.join(args.imagefolder_dir, "train")

    # ──────────────────────────────────────────────────────────────────────
    # 1. ImageFolder + torchvision  (mirrors imagenet.py _build_transform)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("1. ImageFolder + torchvision transforms")
    print(f"{'=' * 60}")
    if args.skip_imagefolder:
        print("  Skipped (--skip-imagefolder).")
    else:
        ops = [
            transforms.Resize(args.image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
        ]
        if args.three_augment:
            ops.append(transforms.ColorJitter(
                brightness=args.color_jitter,
                contrast=args.color_jitter,
                saturation=args.color_jitter,
            ))
            ops.append(ThreeAugment())
        if args.image_size != args.final_image_size:
            ops.append(transforms.Resize(args.final_image_size, interpolation=transforms.InterpolationMode.BICUBIC))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = transforms.Compose(ops)

        try:
            dataset = ImageFolder(train_dir, transform=transform)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=args.num_workers > 0,
            )
            results.append(benchmark_loader(loader, "ImageFolder", args.num_batches, device))
            del loader, dataset
        except Exception as e:
            print(f"  ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # 2. DALI (mirrors imagenet_dali.py _training_pipeline)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("2. DALI (GPU decode + GPU augment)")
    print(f"{'=' * 60}")

    if not DALI_AVAILABLE:
        print("  DALI not available — skipping.")
    else:
        try:
            pipeline = _build_train_pipeline(
                image_dir=train_dir,
                batch_size=args.batch_size,
                num_threads=args.num_workers,
                device_id=0,
                image_size=args.image_size,
                final_image_size=args.final_image_size,
                shard_id=0,
                num_shards=1,
                seed=42,
                use_three_augment=args.three_augment,
                color_jitter=args.color_jitter if args.three_augment else 0.0,
                prefetch_queue_depth=2,
                mean=_IMAGENET_MEAN,
                std=_IMAGENET_STD,
            )
            num_train = sum(len(files) for _, _, files in os.walk(train_dir))
            dali_loader = DALIIteratorWrapper(
                pipeline,
                size=num_train,
                batch_size=args.batch_size,
                auto_reset=True,
                drop_last=True,
            )
            results.append(benchmark_loader(dali_loader, "DALI", args.num_batches, device))
            del dali_loader, pipeline
        except Exception as e:
            print(f"  ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'*' * 60}")
    print("  FINAL RESULTS")
    print(f"{'*' * 60}")
    print(f"  {'Backend':<20s} {'img/s':>10s} {'mean lat':>10s} {'P95 lat':>10s}")
    print(f"  {'─' * 50}")
    for r in results:
        print(
            f"  {r['name']:<20s} {r['img_per_sec']:>10.1f} "
            f"{r['mean_latency_ms']:>8.2f}ms {r['p95_latency_ms']:>8.2f}ms"
        )

    if len(results) == 2:
        speedup = results[1]["img_per_sec"] / results[0]["img_per_sec"]
        print(f"\n  Speedup (DALI / ImageFolder): {speedup:.2f}x")


if __name__ == "__main__":
    main()
