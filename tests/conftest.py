# TODO: Add license header here

"""Shared pytest fixtures for all test files."""

import pytest
import torch


# ---------------------------------------------------------------------------
# subquadratic-ops version gate
# ---------------------------------------------------------------------------


def _subq_ops_version() -> tuple[int, ...]:
    """Return the installed subquadratic-ops-torch-cu12 version as an int tuple."""
    try:
        from importlib.metadata import version

        return tuple(int(x) for x in version("subquadratic-ops-torch-cu12").split(".")[:3])
    except Exception:
        return (0, 0, 0)


_SUBQ_OPS_MIN_VERSION = (0, 2, 0)
_subq_installed = _subq_ops_version()

requires_subq_ops_v2 = pytest.mark.xfail(
    _subq_installed < _SUBQ_OPS_MIN_VERSION,
    reason=(
        f"subquadratic_ops_torch >= {'.'.join(str(x) for x in _SUBQ_OPS_MIN_VERSION)} required "
        f"(installed: {'.'.join(str(x) for x in _subq_installed)})"
    ),
    strict=True,
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
