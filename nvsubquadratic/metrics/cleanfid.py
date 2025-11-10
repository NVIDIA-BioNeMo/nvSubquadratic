"""Thin wrapper around CleanFID helpers."""

from __future__ import annotations

from pathlib import Path

from cleanfid import fid


def compute_folder_fid(
    sample_dir: str | Path,
    *,
    dataset_name: str,
    dataset_resolution: int,
    dataset_split: str = "train",
) -> float:
    """Compute FID between a folder of samples and CleanFID reference statistics."""
    sample_path = Path(sample_dir).expanduser().resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_path}")

    score = fid.compute_fid(
        fdir1=sample_path.as_posix(),
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        dataset_split=dataset_split,
    )
    return float(score)

