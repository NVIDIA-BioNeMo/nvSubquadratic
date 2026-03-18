# TODO: Add license header here

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
