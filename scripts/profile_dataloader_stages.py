"""Profile the dataloading pipeline stage-by-stage to identify bottlenecks.

Breaks down the ImageNet dataloading pipeline into 6 independent stages:
  1. Raw disk I/O (no decode)
  2. I/O + JPEG decode
  3. I/O + decode + full augmentation
  4. Per-transform breakdown
  5. Full DataLoader sweep (varying num_workers and prefetch_factor)
  6. CPU-to-GPU transfer

Usage (interactive SLURM session with 1 GPU):
    PYTHONPATH=. python scripts/profile_dataloader_stages.py [--data-dir /path/to/imagenet_folder]
"""

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from experiments.datamodules.imagenet import (
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    AugmentConfig,
    ImageNetDataModule,
    MixupConfig,
    ThreeAugment,
)


# ── Defaults ─────────────────────────────────────────────────────────────────

NUM_IMAGES = 500       # images per single-image stage
NUM_BATCHES = 100      # batches for DataLoader / transfer stages
BATCH_SIZE = 256
IMAGE_SIZE = 224
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

class TimedTransform:
    """Wraps a transform to accumulate wall-clock time."""

    def __init__(self, transform, name=None):
        self.transform = transform
        self.name = name or type(transform).__name__
        self.total_time = 0.0
        self.count = 0

    def __call__(self, img):
        t0 = time.perf_counter()
        result = self.transform(img)
        self.total_time += time.perf_counter() - t0
        self.count += 1
        return result

    @property
    def avg_ms(self):
        return (self.total_time / max(self.count, 1)) * 1000


def _collect_jpeg_paths(data_dir: str, n: int) -> list[str]:
    """Collect n random JPEG paths from the training set.

    Standard ImageNet uses .JPEG extension; try all common variants.
    """
    all_files = []
    for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.JPG", "*.png", "*.PNG"):
        pattern = os.path.join(data_dir, "train", "*", ext)
        all_files.extend(glob.glob(pattern))
        if all_files:
            break  # stop at the first extension that finds matches
    if not all_files:
        raise FileNotFoundError(f"No image files found in {data_dir}/train/*/")
    random.seed(SEED)
    return random.sample(all_files, min(n, len(all_files)))


def _build_train_transforms() -> list:
    """Build the same transform list used by ImageNetDataModule for training."""
    ops = [
        transforms.Resize(IMAGE_SIZE + 32, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ThreeAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
    ]
    return ops


def _print_header(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _print_result(label: str, total_s: float, count: int, unit: str = "images"):
    per_item_ms = total_s / max(count, 1) * 1000
    throughput = count / max(total_s, 1e-9)
    print(f"  {label}")
    print(f"    Total: {total_s:.2f}s for {count} {unit}")
    print(f"    Per {unit[:-1] if unit.endswith('s') else unit}: {per_item_ms:.2f}ms")
    print(f"    Throughput: {throughput:.0f} {unit}/sec")


# ── Stage 1: Raw disk I/O ───────────────────────────────────────────────────

def stage_raw_io(files: list[str]) -> dict:
    _print_header("STAGE 1: Raw disk I/O (read bytes, no decode)")

    total_bytes = 0
    t0 = time.perf_counter()
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        data = os.read(fd, os.path.getsize(path))
        total_bytes += len(data)
        os.close(fd)
    elapsed = time.perf_counter() - t0

    mb = total_bytes / (1024 * 1024)
    bandwidth = mb / max(elapsed, 1e-9)
    _print_result("Raw file read", elapsed, len(files))
    print(f"    Total read: {mb:.1f} MB")
    print(f"    Bandwidth: {bandwidth:.0f} MB/s")

    return {"stage": "raw_io", "elapsed_s": elapsed, "num_files": len(files),
            "total_mb": mb, "bandwidth_mbs": bandwidth}


# ── Stage 2: I/O + JPEG decode ──────────────────────────────────────────────

def stage_decode(files: list[str]) -> dict:
    _print_header("STAGE 2: I/O + JPEG decode (PIL)")

    t0 = time.perf_counter()
    for path in files:
        img = Image.open(path).convert("RGB")
        # Force decode (PIL is lazy)
        img.load()
    elapsed = time.perf_counter() - t0

    _print_result("Decode (PIL)", elapsed, len(files))

    return {"stage": "decode", "elapsed_s": elapsed, "num_files": len(files)}


# ── Stage 3: I/O + decode + full transform ──────────────────────────────────

def stage_decode_and_transform(files: list[str]) -> dict:
    _print_header("STAGE 3: I/O + decode + full augmentation pipeline")

    transform = transforms.Compose(_build_train_transforms())

    t0 = time.perf_counter()
    for path in files:
        img = Image.open(path).convert("RGB")
        _ = transform(img)
    elapsed = time.perf_counter() - t0

    _print_result("Decode + transform", elapsed, len(files))

    return {"stage": "decode_and_transform", "elapsed_s": elapsed, "num_files": len(files)}


# ── Stage 4: Per-transform breakdown ────────────────────────────────────────

def stage_per_transform(files: list[str]) -> dict:
    _print_header("STAGE 4: Per-transform breakdown")

    raw_ops = _build_train_transforms()
    timed_ops = [TimedTransform(op) for op in raw_ops]
    pipeline = transforms.Compose(timed_ops)

    for path in files:
        img = Image.open(path).convert("RGB")
        _ = pipeline(img)

    # Compute totals
    total_ms = sum(op.total_time for op in timed_ops) * 1000
    results = []

    print(f"\n  {'Transform':<35} {'Time/img (ms)':>14} {'% of pipeline':>14}")
    print(f"  {'─' * 35} {'─' * 14} {'─' * 14}")
    for op in timed_ops:
        pct = op.total_time * 1000 / max(total_ms, 1e-9) * 100
        print(f"  {op.name:<35} {op.avg_ms:>13.2f} {pct:>13.1f}%")
        results.append({"name": op.name, "avg_ms": op.avg_ms, "pct": pct})

    avg_total = total_ms / max(len(files), 1)
    print(f"  {'─' * 35} {'─' * 14}")
    print(f"  {'TOTAL':<35} {avg_total:>13.2f}")

    return {"stage": "per_transform", "num_files": len(files),
            "total_pipeline_avg_ms": avg_total, "transforms": results}


# ── Stage 5: Full DataLoader sweep ──────────────────────────────────────────

def stage_dataloader_sweep(data_dir: str) -> dict:
    _print_header("STAGE 5: Full DataLoader sweep (num_workers x prefetch_factor)")

    worker_counts = [1, 4, 8, 13, 16]
    prefetch_factors = [2, 4, 8]
    results = []

    print(f"\n  {'Workers':>8} {'Prefetch':>9} {'ms/batch':>10} {'img/sec':>10}")
    print(f"  {'─' * 8} {'─' * 9} {'─' * 10} {'─' * 10}")

    best_throughput = 0
    best_config = ""

    for nw in worker_counts:
        for pf in prefetch_factors:
            try:
                dm = ImageNetDataModule(
                    data_dir=os.environ.get("IMAGENET_PATH", data_dir),
                    imagefolder_dir=data_dir,
                    prefetch_factor=pf,
                    batch_size=BATCH_SIZE,
                    num_workers=nw,
                    pin_memory=True,
                    seed=SEED,
                    image_size=IMAGE_SIZE,
                    final_image_size=IMAGE_SIZE,
                    center_crop=True,
                    num_classes=1000,
                    drop_labels=False,
                    hf_dataset_name="ILSVRC/imagenet-1k",
                    task="classification",
                    augment_cfg=AugmentConfig(use_three_augment=True, color_jitter=0.3),
                )
                dm.prepare_data()
                dm.setup("fit")
                loader = dm.train_dataloader()
                it = iter(loader)

                # Warmup
                for _ in range(3):
                    next(it)

                t0 = time.perf_counter()
                for _ in range(NUM_BATCHES):
                    next(it)
                elapsed = time.perf_counter() - t0

                ms_per_batch = elapsed / NUM_BATCHES * 1000
                throughput = BATCH_SIZE * NUM_BATCHES / elapsed

                print(f"  {nw:>8} {pf:>9} {ms_per_batch:>9.1f} {throughput:>9.0f}")
                results.append({"num_workers": nw, "prefetch_factor": pf,
                                "ms_per_batch": ms_per_batch, "throughput": throughput})

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = f"workers={nw}, prefetch={pf}"

                del loader, it, dm
            except Exception as e:
                print(f"  {nw:>8} {pf:>9} {'ERROR':>10} {str(e)[:30]}")

    print(f"\n  >> Best: {best_config} ({best_throughput:.0f} img/sec)")

    return {"stage": "dataloader_sweep", "results": results,
            "best_config": best_config, "best_throughput": best_throughput}


# ── Stage 6: CPU-to-GPU transfer ────────────────────────────────────────────

def stage_gpu_transfer() -> dict:
    _print_header("STAGE 6: CPU-to-GPU transfer")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping.")
        return {"stage": "gpu_transfer", "skipped": True}

    device = torch.device("cuda")
    results = []

    # Test with pin_memory=True and False
    for pinned in [False, True]:
        # Create a batch-sized tensor on CPU
        tensor = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        if pinned:
            tensor = tensor.pin_memory()

        # Warmup
        for _ in range(5):
            _ = tensor.to(device, non_blocking=True)
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(NUM_BATCHES):
            _ = tensor.to(device, non_blocking=True)
            torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        per_batch_ms = total_ms / NUM_BATCHES
        mb_per_batch = tensor.nelement() * tensor.element_size() / (1024 * 1024)
        bandwidth = mb_per_batch * NUM_BATCHES / (total_ms / 1000) / 1024  # GB/s

        label = "pinned" if pinned else "paged"
        print(f"  {label:>8}: {per_batch_ms:.2f}ms/batch, "
              f"{mb_per_batch:.1f} MB/batch, {bandwidth:.1f} GB/s")

        results.append({"pinned": pinned, "ms_per_batch": per_batch_ms,
                        "mb_per_batch": mb_per_batch, "bandwidth_gbs": bandwidth})

    return {"stage": "gpu_transfer", "results": results}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global NUM_IMAGES, NUM_BATCHES, BATCH_SIZE

    parser = argparse.ArgumentParser(description="Profile ImageNet dataloading stages")
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("IMAGENET_FOLDER_PATH",
                                               "/shared/data/image_datasets/imagenet_folder"),
                        help="Path to ImageFolder-format ImageNet directory")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES,
                        help="Number of images for single-image stages (1-4)")
    parser.add_argument("--num-batches", type=int, default=NUM_BATCHES,
                        help="Number of batches for DataLoader/transfer stages (5-6)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--json", type=str, default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()
    NUM_IMAGES = args.num_images
    NUM_BATCHES = args.num_batches
    BATCH_SIZE = args.batch_size

    data_dir = args.data_dir

    print("ImageNet Dataloading Stage Profiler")
    print("─" * 64)
    print(f"  Data dir:    {data_dir}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  Num images:  {NUM_IMAGES} (single-image stages)")
    print(f"  Num batches: {NUM_BATCHES} (DataLoader/transfer stages)")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name()}")
    print(f"  CPUs:        {os.cpu_count()}")

    files = _collect_jpeg_paths(data_dir, NUM_IMAGES)
    print(f"  Sampled {len(files)} JPEG files from {data_dir}/train/")

    all_results = []

    # Stages 1-4: single-image profiling
    all_results.append(stage_raw_io(files))
    all_results.append(stage_decode(files))
    all_results.append(stage_decode_and_transform(files))
    all_results.append(stage_per_transform(files))

    # Stage 5: DataLoader sweep
    all_results.append(stage_dataloader_sweep(data_dir))

    # Stage 6: GPU transfer
    all_results.append(stage_gpu_transfer())

    # ── Summary ──────────────────────────────────────────────────────────
    _print_header("SUMMARY: Single-image pipeline breakdown")

    io_ms = all_results[0]["elapsed_s"] / max(len(files), 1) * 1000
    decode_ms = all_results[1]["elapsed_s"] / max(len(files), 1) * 1000
    full_ms = all_results[2]["elapsed_s"] / max(len(files), 1) * 1000
    decode_only = decode_ms - io_ms
    transform_only = full_ms - decode_ms

    total = full_ms
    print(f"  {'Component':<25} {'ms/image':>10} {'%':>8}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 8}")
    print(f"  {'Disk I/O':<25} {io_ms:>9.2f} {io_ms / total * 100:>7.1f}%")
    print(f"  {'JPEG decode':<25} {decode_only:>9.2f} {decode_only / total * 100:>7.1f}%")
    print(f"  {'Augmentation':<25} {transform_only:>9.2f} {transform_only / total * 100:>7.1f}%")
    print(f"  {'─' * 25} {'─' * 10}")
    print(f"  {'Total':<25} {total:>9.2f}")
    print()

    if decode_only + transform_only > io_ms:
        cpu_pct = (decode_only + transform_only) / total * 100
        print(f"  >> CPU processing (decode + augment) is {cpu_pct:.0f}% of per-image time.")
        print(f"  >> GPU-accelerated decode/augment (DALI) would likely help.")
    else:
        print(f"  >> Disk I/O is the dominant cost ({io_ms / total * 100:.0f}%).")
        print(f"  >> Consider faster storage or prefetching, not GPU decode.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {args.json}")


if __name__ == "__main__":
    main()
