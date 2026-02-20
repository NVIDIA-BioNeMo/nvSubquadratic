#!/usr/bin/env python3
"""Convert HuggingFace Arrow ImageNet cache to WebDataset TAR shards.

This script reads the locally cached HuggingFace ILSVRC/imagenet-1k dataset
(already downloaded as Arrow files) and re-packages it into WebDataset TAR
shards for efficient sequential I/O.

Usage:
    # From the repo root with the nvsubq conda environment activated:
    conda activate nvsubq
    python scripts/convert_imagenet_to_webdataset.py \
        --src data/imagenet \
        --dst data/imagenet-wds \
        --splits train validation

    # Or run just one split:
    python scripts/convert_imagenet_to_webdataset.py \
        --src data/imagenet \
        --dst data/imagenet-wds \
        --splits train

Requirements:
    pip install webdataset
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset

# WebDataset import — fail early with a helpful message
try:
    import webdataset as wds
except ImportError:
    print("ERROR: webdataset is not installed. Run: pip install webdataset")
    sys.exit(1)


def convert_split(
    src_dir: str,
    dst_dir: str,
    split: str,
    max_shard_size: int = 500_000_000,  # ~500 MB per shard
    dataset_name: str = "ILSVRC/imagenet-1k",
) -> None:
    """Convert one split of the HuggingFace ImageNet dataset to WebDataset.

    Args:
        src_dir: Path to the HuggingFace cache directory (e.g. data/imagenet).
        dst_dir: Output directory for the WebDataset shards.
        split: Dataset split name (e.g. "train", "validation").
        max_shard_size: Maximum shard file size in bytes.
        dataset_name: HuggingFace dataset name.
    """
    print(f"\n{'='*70}")
    print(f"Converting split: {split}")
    print(f"{'='*70}")

    # Load the cached dataset (no download — uses local Arrow files)
    print(f"Loading HuggingFace dataset from cache: {src_dir}")
    ds = load_dataset(
        dataset_name,
        split=split,
        cache_dir=src_dir,
    )
    num_samples = len(ds)
    print(f"Loaded {num_samples:,} samples")

    # Set up output directory
    shard_dir = Path(dst_dir) / split
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Shard pattern: imagenet-train-000000.tar, imagenet-train-000001.tar, ...
    shard_pattern = str(shard_dir / f"imagenet-{split}-%06d.tar")
    print(f"Writing shards to: {shard_pattern}")

    # Write shards
    t_start = time.time()
    written = 0

    with wds.ShardWriter(shard_pattern, maxsize=max_shard_size) as sink:
        for idx in range(num_samples):
            example = ds[idx]
            image = example["image"]
            label = example["label"]

            # Encode PIL image to JPEG bytes
            buf = io.BytesIO()
            # Convert to RGB if needed (some images may be grayscale/RGBA)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buf, format="JPEG", quality=95)
            jpeg_bytes = buf.getvalue()

            # Write sample: key + extensions determine the field names
            # WebDataset convention: __key__ is the unique sample ID
            sample = {
                "__key__": f"{idx:08d}",
                "jpg": jpeg_bytes,
                "cls": label,
            }
            sink.write(sample)
            written += 1

            if written % 10_000 == 0:
                elapsed = time.time() - t_start
                rate = written / elapsed
                eta = (num_samples - written) / rate if rate > 0 else 0
                print(
                    f"  [{written:>8,}/{num_samples:,}] "
                    f"{rate:.0f} samples/s, "
                    f"ETA: {eta/60:.1f} min"
                )

    elapsed = time.time() - t_start
    num_shards = len(list(shard_dir.glob("*.tar")))
    total_size = sum(f.stat().st_size for f in shard_dir.glob("*.tar"))

    print(f"\nDone! Split: {split}")
    print(f"  Samples:   {written:,}")
    print(f"  Shards:    {num_shards}")
    print(f"  Total size: {total_size / 1e9:.1f} GB")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Rate:      {written/elapsed:.0f} samples/s")

    # Write a metadata file for easy reference
    meta = {
        "split": split,
        "num_samples": written,
        "num_shards": num_shards,
        "total_size_bytes": total_size,
        "shard_pattern": f"imagenet-{split}-{{000000..{num_shards-1:06d}}}.tar",
        "format": {"image": "jpg", "label": "cls"},
    }
    meta_path = shard_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata:  {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace ImageNet Arrow cache to WebDataset TAR shards."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/imagenet",
        help="Path to HuggingFace cache directory (default: data/imagenet)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/imagenet-wds",
        help="Output directory for WebDataset shards (default: data/imagenet-wds)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Splits to convert (default: train validation)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=500_000_000,
        help="Max shard size in bytes (default: 500MB)",
    )
    args = parser.parse_args()

    print("ImageNet Arrow → WebDataset Converter")
    print(f"  Source:     {args.src}")
    print(f"  Output:     {args.dst}")
    print(f"  Splits:     {args.splits}")
    print(f"  Shard size: {args.max_shard_size / 1e6:.0f} MB")

    for split in args.splits:
        convert_split(
            src_dir=args.src,
            dst_dir=args.dst,
            split=split,
            max_shard_size=args.max_shard_size,
        )

    print(f"\n{'='*70}")
    print("All splits converted successfully!")
    print(f"WebDataset shards are in: {args.dst}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
