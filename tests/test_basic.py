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

"""
Basic tests for nvSubquadratic package.

These tests verify that the package can be imported and basic functionality
works.
"""

import pytest


def test_package_import() -> None:
    """Test that the package can be imported successfully."""
    import nvsubquadratic

    assert nvsubquadratic is not None


def test_torch_import() -> None:
    """Test that PyTorch is available (dependency check)."""
    import torch

    assert torch is not None
    assert hasattr(torch, "__version__")


@pytest.mark.skip
def test_subquadratic_ops_import() -> None:
    """Test that subquadratic-ops is available (required dependency check)."""
    try:
        import subquadratic_ops  # type: ignore[import-untyped]

        assert subquadratic_ops is not None
        # Check if it has expected attributes or functions
        # This is a basic check - you might want to add more specific checks
        # based on what subquadratic_ops actually provides
        assert hasattr(subquadratic_ops, "__version__") or hasattr(subquadratic_ops, "__file__")
    except ImportError as e:
        pytest.fail(f"subquadratic-ops is a required dependency but could not be imported: {e}")


def test_megatron_import() -> None:
    """Test that megatron-core is available (required dependency check)."""
    try:
        import megatron.core  # type: ignore[import-untyped]

        assert megatron.core is not None
        # Check if it has expected attributes or functions
        # This is a basic check - you might want to add more specific checks
        # based on what megatron.core actually provides
        assert hasattr(megatron.core, "__version__") or hasattr(megatron.core, "__file__")
    except ImportError as e:
        pytest.fail(f"megatron-core is a required dependency but could not be imported: {e}")


def test_always_passes() -> None:
    """A test that always passes - useful for debugging test infrastructure."""
    assert True
