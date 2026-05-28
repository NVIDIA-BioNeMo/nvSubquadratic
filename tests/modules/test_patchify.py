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

# TODO: Add license header here

"""Tests for the Patchify and Unpatchify modules."""

import pytest
import torch

from nvsubquadratic.modules.patchify import Patchify, Unpatchify


@pytest.mark.parametrize(
    "data_dim,spatial_shape",
    [
        (1, (64,)),
        (2, (64, 64)),
        (3, (16, 16, 16)),
    ],
)
def test_patchify_unpatchify_roundtrip(device, data_dim: int, spatial_shape: tuple) -> None:
    """Test that patchify -> unpatchify preserves shape."""
    B, hidden_dim, embedding_dim, patch_size = 2, 3, 32, 8

    x = torch.randn(B, *spatial_shape, hidden_dim, device=device)

    patchify_layer = Patchify(
        in_features=hidden_dim,
        out_features=embedding_dim,
        data_dim=data_dim,
        patch_size=patch_size,
    ).to(device)

    unpatchify_layer = Unpatchify(
        in_features=embedding_dim,
        out_features=hidden_dim,
        data_dim=data_dim,
        patch_size=patch_size,
    ).to(device)

    y = patchify_layer(x)
    x_rec = unpatchify_layer(y)

    assert x_rec.shape == x.shape, f"Expected {x.shape}, got {x_rec.shape}"


@pytest.mark.parametrize(
    "data_dim,spatial_shape",
    [
        (1, (64,)),
        (2, (64, 64)),
        (3, (16, 16, 16)),
    ],
)
def test_patchify_output_shape(device, data_dim: int, spatial_shape: tuple) -> None:
    """Test that patchify produces the expected output shape."""
    B, hidden_dim, embedding_dim, patch_size = 2, 3, 32, 8

    x = torch.randn(B, *spatial_shape, hidden_dim, device=device)

    patchify_layer = Patchify(
        in_features=hidden_dim,
        out_features=embedding_dim,
        data_dim=data_dim,
        patch_size=patch_size,
    ).to(device)

    y = patchify_layer(x)

    # Expected spatial dims: each dimension divided by patch_size
    expected_spatial = tuple(s // patch_size for s in spatial_shape)
    expected_shape = (B, *expected_spatial, embedding_dim)

    assert y.shape == expected_shape, f"Expected {expected_shape}, got {tuple(y.shape)}"


def test_patchify_invalid_data_dim() -> None:
    """Test that invalid data_dim raises ValueError."""
    with pytest.raises(ValueError, match="data_dim must be 1, 2, or 3"):
        Patchify(in_features=3, out_features=32, data_dim=4, patch_size=8)


def test_unpatchify_invalid_data_dim() -> None:
    """Test that invalid data_dim raises ValueError."""
    with pytest.raises(ValueError, match="data_dim must be 1, 2, or 3"):
        Unpatchify(in_features=32, out_features=3, data_dim=4, patch_size=8)
