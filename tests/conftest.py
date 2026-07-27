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

"""Shared pytest fixtures for all test files."""

import pytest
import torch


# ---------------------------------------------------------------------------
# subquadratic-ops version gate
# ---------------------------------------------------------------------------


# The kernels ship as one distribution per CUDA major version. Query every
# known name: pinning only one of them silently reports (0, 0, 0) on a machine
# that has the other installed, which turns the gates below into blanket xfails.
_SUBQ_OPS_DISTRIBUTIONS = ("subquadratic-ops-torch-cu13", "subquadratic-ops-torch-cu12")


def _subq_ops_version() -> tuple[int, ...]:
    """Return the installed subquadratic-ops-torch version as an int tuple.

    Returns ``(0, 0, 0)`` when no known distribution is installed.
    """
    from importlib.metadata import version

    for dist in _SUBQ_OPS_DISTRIBUTIONS:
        try:
            return tuple(int(x) for x in version(dist).split(".")[:3])
        except Exception:
            continue
    return (0, 0, 0)


_SUBQ_OPS_MIN_VERSION = (0, 2, 0)
# fused_fft_conv2d (the native-dtype single-launch 2D kernel) landed in 0.2.2.
_SUBQ_OPS_FUSED_MIN_VERSION = (0, 2, 2)
_subq_installed = _subq_ops_version()

requires_subq_ops_v2 = pytest.mark.xfail(
    _subq_installed < _SUBQ_OPS_MIN_VERSION,
    reason=(
        f"subquadratic_ops_torch >= {'.'.join(str(x) for x in _SUBQ_OPS_MIN_VERSION)} required "
        f"(installed: {'.'.join(str(x) for x in _subq_installed)})"
    ),
    strict=False,
)

# Unlike requires_subq_ops_v2 this skips rather than xfails: the fused kernel is
# a strictly newer addition, so an older-but-working install is "not applicable"
# rather than "expected to fail".
requires_subq_ops_fused = pytest.mark.skipif(
    _subq_installed < _SUBQ_OPS_FUSED_MIN_VERSION,
    reason=(
        f"subquadratic_ops_torch >= {'.'.join(str(x) for x in _SUBQ_OPS_FUSED_MIN_VERSION)} "
        f"required for fused_fft_conv2d (installed: {'.'.join(str(x) for x in _subq_installed)})"
    ),
)


@pytest.fixture
def device():
    """Get CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return torch.device("cpu")


@pytest.fixture(params=["float32", "float16", "bfloat16"])
def dtype_fixture(request):
    """Parametrize tests across different dtypes.

    Returns the torch dtype directly. Tests can check tensor.dtype if needed
    for dtype-specific logic (e.g., setting tolerances).
    """
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return dtype_map[request.param]
