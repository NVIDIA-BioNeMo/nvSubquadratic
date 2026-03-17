"""Extract ImageNet from HuggingFace Arrow format directly to a tar archive.

Writes raw JPEG bytes directly into a single tar file (no re-encoding, no
intermediate files on disk).  The tar entries mirror the ImageFolder layout
used by extract_imagenet_to_folder.py so that extraction produces a ready-to-
use ImageFolder dataset:

    imagenet_imagefolder/train/0000/000000.jpg
    imagenet_imagefolder/train/0999/004231.jpg
    imagenet_imagefolder/val/0000/000000.jpg

Writing one large sequential file avoids the GPFS small-file metadata overhead
that makes writing 1.28M individual JPEGs extremely slow.

Usage:
    PYTHONPATH=. python scripts/extract_imagenet_to_tar.py
"""

import io
import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path


HF_DATASET = os.environ.get("IMAGENET_HF_DATASET", "imagenet-1k")
HF_CACHE = os.environ.get("IMAGENET_PATH", "/scratch-shared/dknigge/hf_cache")
OUTPUT_TAR = os.environ.get("IMAGENET_OUTPUT_TAR", "/scratch-shared/dknigge/imagenet_imagefolder.tar")
TAR_ROOT = "imagenet_imagefolder"


def _guess_ext(raw_bytes: bytes) -> str:
    if raw_bytes[:2] == b"\xff\xd8":
        return ".jpg"
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def extract_split_to_tar(split: str, tf: tarfile.TarFile) -> None:
    import datasets

    print(f"\n{'=' * 60}")
    print(f"Extracting split: {split}")
    print(f"{'=' * 60}")

    hf_token = os.environ.get("HF_TOKEN")
    ds = datasets.load_dataset(
        HF_DATASET,
        split=split,
        streaming=False,
        cache_dir=HF_CACHE,
        token=hf_token,
    )
    ds_raw = ds.cast_column("image", datasets.Image(decode=False))

    folder_name = "train" if split == "train" else "val"
    class_counters = defaultdict(int)
    total = len(ds_raw)
    t0 = time.time()

    for i in range(total):
        row = ds_raw[i]
        label = row["label"]
        img_bytes = row["image"]["bytes"]

        ext = _guess_ext(img_bytes)
        filename = f"{class_counters[label]:06d}{ext}"
        tar_path = f"{TAR_ROOT}/{folder_name}/{label:04d}/{filename}"

        info = tarfile.TarInfo(name=tar_path)
        info.size = len(img_bytes)
        tf.addfile(info, io.BytesIO(img_bytes))

        class_counters[label] += 1

        if (i + 1) % 50_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  [{split}] {i + 1:>8d}/{total}  ({rate:.0f} img/s, ETA {eta / 60:.1f} min)", flush=True)

    elapsed = time.time() - t0
    print(
        f"  [{split}] Done: {total} images, {len(class_counters)} classes "
        f"in {elapsed:.0f}s ({total / elapsed:.0f} img/s)"
    )


def main():
    tmp_path = OUTPUT_TAR + ".tmp"
    out_path = Path(OUTPUT_TAR)

    if out_path.exists():
        print(f"Output already exists: {out_path}. Nothing to do.")
        return

    print(f"Output tar: {OUTPUT_TAR}")
    print(f"HF cache:   {HF_CACHE}")

    with tarfile.open(tmp_path, "w|") as tf:
        extract_split_to_tar("train", tf)
        extract_split_to_tar("validation", tf)

    Path(tmp_path).rename(out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"\nDone. Tar written to {out_path} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
