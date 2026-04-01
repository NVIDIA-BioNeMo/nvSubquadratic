"""Validate that HDF5 caching produces bit-identical results to upstream.

Loads a few samples from the Gray-Scott validation split with and without
the _enable_h5_caching patch and asserts exact equality on all tensor fields.

Run with:
    conda run -n nv-subq python -m pytest tests/test_h5_caching.py -v
"""

import os

import numpy as np
import pytest
import torch
from the_well.data.datasets import WellDataset

from experiments.datamodules.pde.well import _enable_h5_caching


WELL_BASE = "/shared/data/image_datasets/the_well/datasets"
DATASET = "gray_scott_reaction_diffusion"
SPLIT = "valid"
SAMPLE_INDICES = [0, 1, 42, 100]

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(WELL_BASE, DATASET)),
    reason=f"No HDF5 files found in path {os.path.join(WELL_BASE, DATASET)} — dataset not available",
)


def _make_dataset():
    return WellDataset(
        well_base_path=WELL_BASE,
        well_dataset_name=DATASET,
        well_split_name=SPLIT,
        use_normalization=False,
        n_steps_input=4,
        n_steps_output=1,
        max_rollout_steps=32,
    )


def test_cached_matches_upstream():
    """Samples from cached and uncached datasets must be bit-identical."""
    ds_ref = _make_dataset()
    ds_cached = _make_dataset()
    _enable_h5_caching(ds_cached)

    for idx in SAMPLE_INDICES:
        sample_ref = ds_ref[idx]
        sample_cached = ds_cached[idx]

        assert sample_ref.keys() == sample_cached.keys(), (
            f"Key mismatch at index {idx}: {sample_ref.keys()} vs {sample_cached.keys()}"
        )

        for key in sample_ref:
            ref_val = sample_ref[key]
            cached_val = sample_cached[key]

            if isinstance(ref_val, torch.Tensor):
                assert torch.equal(ref_val, cached_val), (
                    f"Tensor mismatch at index {idx}, key '{key}': "
                    f"max diff = {(ref_val - cached_val).abs().max().item()}"
                )
            elif isinstance(ref_val, np.ndarray):
                np.testing.assert_array_equal(ref_val, cached_val)
            else:
                assert ref_val == cached_val, f"Value mismatch at index {idx}, key '{key}': {ref_val} vs {cached_val}"


def test_cached_repeated_access():
    """Accessing the same index twice must return identical results (handle reuse)."""
    ds = _make_dataset()
    _enable_h5_caching(ds)

    idx = 42
    sample_a = ds[idx]
    sample_b = ds[idx]

    for key in sample_a:
        if isinstance(sample_a[key], torch.Tensor):
            assert torch.equal(sample_a[key], sample_b[key]), f"Mismatch on repeated access, key '{key}'"


def test_cached_different_files():
    """Access samples from different HDF5 files (different file_idx values)."""
    ds = _make_dataset()
    _enable_h5_caching(ds)

    n = len(ds)
    indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for idx in indices:
        sample = ds[idx]
        assert "input_fields" in sample, f"Missing input_fields at index {idx}"
        assert sample["input_fields"].shape[0] == 4, f"Wrong n_steps_input at index {idx}"
