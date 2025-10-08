# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
