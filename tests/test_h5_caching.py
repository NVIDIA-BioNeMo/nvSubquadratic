# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate that HDF5 caching produces bit-identical results to upstream.

Loads a few samples from the Gray-Scott validation split with and without
the _enable_h5_caching patch and asserts exact equality on all tensor fields.

Run with:
    conda run -n nv-subq python -m pytest tests/test_h5_caching.py -v
"""

import pathlib

import numpy as np
import pytest
import torch
from the_well.data.datasets import WellDataset

from experiments.datamodules.pde.well import _enable_h5_caching


WELL_BASE = "/shared/data/image_datasets/the_well/datasets"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(WELL_BASE).exists(),
    reason=f"WELL dataset not available at {WELL_BASE}",
)
DATASET = "gray_scott_reaction_diffusion"
SPLIT = "valid"
SAMPLE_INDICES = [0, 1, 42, 100]


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
