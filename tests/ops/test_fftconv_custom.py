# TODO: Add license header here


"""Tests for the fftconv_custom wrappers (subq_ops CUDA kernel drop-in replacements).

Validates :mod:`nvsubquadratic.ops.fftconv_custom` against the torch.fft
reference implementation in :mod:`nvsubquadratic.ops.fftconv`.

Covers:
  - Forward correctness: shared kernel, FiLM kernel, BHL and BLH layouts
  - Chunked vs non-chunked consistency
  - Backward correctness: gradient comparison for shared and FiLM kernels
  - Shortcut semantics
  - Dtype handling: bf16 and fp32 inputs

Usage (requires GPU — run inside SLURM):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/ops/test_fftconv_custom.py -v -o addopts=""
"""

import pytest
import torch


# Tolerances for f32 comparison between torch.fft and subq_ops wrapper
ATOL_F32 = 1e-3
RTOL_F32 = 1e-4

# Tolerance for gradient comparison (slightly looser due to different FFT implementations)
ATOL_GRAD = 1e-2
RTOL_GRAD = 1e-3


def _has_subq_ops() -> bool:
    try:
        from subquadratic_ops_torch.fft_conv2d import fft_conv2d  # noqa: F401

        return True
    except ImportError:
        return False


requires_subq_ops = pytest.mark.skipif(
    not _has_subq_ops(),
    reason="subquadratic_ops_torch not installed",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
# TODO(@moradza): remove this skip when subquadratic-ops adds support for B * conv kernels
skip_batch_kernel = pytest.mark.skip(
    reason="B * conv kernels not supported in current subquadratic-ops release; pending next version"
)

pytestmark = [requires_subq_ops, requires_cuda]


@pytest.fixture
def device() -> str:
    return "cuda"


# ---------------------------------------------------------------------------
# Helpers — reference functions from the torch.fft implementation
# ---------------------------------------------------------------------------


def _ref_bhl(x, kernel, shortcut=None):
    from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl

    return fftconv2d_fp32_bhl(x, kernel, shortcut)


def _ref_bhl_w_reshape(x, kernel, shortcut=None):
    from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl_w_reshape

    return fftconv2d_fp32_bhl_w_reshape(x, kernel, shortcut)


def _custom_bhl(x, kernel, shortcut=None):
    from nvsubquadratic.ops.fftconv_custom import fftconv2d_bhl

    return fftconv2d_bhl(x, kernel, shortcut)


def _custom_bhl_w_reshape(x, kernel, shortcut=None):
    from nvsubquadratic.ops.fftconv_custom import fftconv2d_bhl_w_reshape

    return fftconv2d_bhl_w_reshape(x, kernel, shortcut)


def _custom_bhl_chunked(x, kernel, shortcut=None, chunk_size=None):
    from nvsubquadratic.ops.fftconv_custom import fftconv2d_bhl_chunked

    return fftconv2d_bhl_chunked(x, kernel, shortcut, chunk_size)


def _custom_bhl_w_reshape_chunked(x, kernel, shortcut=None, chunk_size=None):
    from nvsubquadratic.ops.fftconv_custom import fftconv2d_bhl_w_reshape_chunked

    return fftconv2d_bhl_w_reshape_chunked(x, kernel, shortcut, chunk_size)


# ---------------------------------------------------------------------------
# Forward correctness — shared kernel
# ---------------------------------------------------------------------------

SHARED_SHAPES = [
    (2, 64, 32, 32, 7, 7),
    (4, 128, 14, 14, 7, 7),
    (2, 256, 28, 28, 13, 13),
    (1, 64, 64, 64, 15, 15),
    (1, 384, 14, 14, 14, 14),
    # grid_type="double": kernel larger than input
    (2, 64, 14, 14, 28, 28),
]


class TestForwardSharedKernel:
    """Forward correctness for shared kernels [1, H, Kx, Ky]."""

    @pytest.mark.parametrize("B, H, X, Y, Kx, Ky", SHARED_SHAPES)
    def test_matches_reference_bhl(self, device, B, H, X, Y, Kx, Ky):
        """Custom wrapper matches torch.fft reference in BHL layout."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, kernel)
        y_custom = _custom_bhl(x, kernel)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (2, 64, 32, 32, 7, 7),
            (1, 384, 14, 14, 14, 14),
        ],
    )
    def test_matches_reference_blh(self, device, B, H, X, Y, Kx, Ky):
        """Custom wrapper matches torch.fft reference in BLH layout."""
        torch.manual_seed(42)
        x = torch.randn(B, X, Y, H, device=device, dtype=torch.float32)
        kernel = torch.randn(1, Kx, Ky, H, device=device, dtype=torch.float32)

        y_ref = _ref_bhl_w_reshape(x, kernel)
        y_custom = _custom_bhl_w_reshape(x, kernel)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)


# ---------------------------------------------------------------------------
# Forward correctness — FiLM (per-sample) kernel
# ---------------------------------------------------------------------------

FILM_SHAPES = [
    (4, 64, 32, 32, 7, 7),
    (2, 128, 14, 14, 7, 7),
    (3, 64, 28, 28, 13, 13),
    # grid_type="double": kernel larger than input
    (2, 64, 14, 14, 28, 28),
]


class TestForwardFiLMKernel:
    """Forward correctness for per-sample FiLM kernels [B, H, Kx, Ky]."""

    @skip_batch_kernel
    @pytest.mark.parametrize("B, H, X, Y, Kx, Ky", FILM_SHAPES)
    def test_matches_reference(self, device, B, H, X, Y, Kx, Ky):
        """Custom wrapper matches torch.fft reference for FiLM kernels."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, kernel)
        y_custom = _custom_bhl(x, kernel)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)


# ---------------------------------------------------------------------------
# Chunked correctness
# ---------------------------------------------------------------------------


class TestChunked:
    """Chunked variants match the non-chunked variants exactly."""

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (2, 64, 32, 32, 7, 7),
            (4, 128, 14, 14, 7, 7),
            (2, 64, 14, 14, 28, 28),
        ],
    )
    @pytest.mark.parametrize("chunk_size", [32, 64, 128])
    def test_chunked_matches_non_chunked_shared(self, device, B, H, X, Y, Kx, Ky, chunk_size):
        """Chunked and non-chunked produce identical output for shared kernels."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y = _custom_bhl(x, kernel)
        y_chunked = _custom_bhl_chunked(x, kernel, chunk_size=chunk_size)

        torch.testing.assert_close(y_chunked, y, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (4, 64, 32, 32, 7, 7),
            (2, 128, 14, 14, 7, 7),
        ],
    )
    @skip_batch_kernel
    def test_chunked_matches_non_chunked_film(self, device, B, H, X, Y, Kx, Ky):
        """Chunked and non-chunked produce identical output for FiLM kernels."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32)

        y = _custom_bhl(x, kernel)
        y_chunked = _custom_bhl_chunked(x, kernel, chunk_size=32)

        torch.testing.assert_close(y_chunked, y, atol=0, rtol=0)

    def test_chunked_blh_matches_non_chunked(self, device):
        """Chunked BLH (w_reshape) matches non-chunked BLH."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 2, 64, 32, 32, 7, 7
        x = torch.randn(B, X, Y, H, device=device, dtype=torch.float32)
        kernel = torch.randn(1, Kx, Ky, H, device=device, dtype=torch.float32)

        y = _custom_bhl_w_reshape(x, kernel)
        y_chunked = _custom_bhl_w_reshape_chunked(x, kernel, chunk_size=32)

        torch.testing.assert_close(y_chunked, y, atol=0, rtol=0)

    def test_chunked_matches_reference(self, device):
        """Chunked wrapper matches torch.fft reference."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 2, 128, 14, 14, 7, 7
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, kernel)
        y_chunked = _custom_bhl_chunked(x, kernel, chunk_size=32)

        torch.testing.assert_close(y_chunked, y_ref, atol=ATOL_F32, rtol=RTOL_F32)


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------


class TestBackward:
    """Gradient comparison between custom wrapper and torch.fft reference."""

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (2, 32, 16, 16, 5, 5),
            (4, 64, 14, 14, 7, 7),
        ],
    )
    def test_backward_shared_kernel(self, device, B, H, X, Y, Kx, Ky):
        """Gradients match reference for shared kernels."""
        torch.manual_seed(42)

        # Reference
        x_ref = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        k_ref = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)
        y_ref = _ref_bhl(x_ref, k_ref)
        y_ref.sum().backward()

        # Custom wrapper
        x_cus = x_ref.detach().clone().requires_grad_(True)
        k_cus = k_ref.detach().clone().requires_grad_(True)
        y_cus = _custom_bhl(x_cus, k_cus)
        y_cus.sum().backward()

        torch.testing.assert_close(x_cus.grad, x_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)
        torch.testing.assert_close(k_cus.grad, k_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (4, 32, 16, 16, 5, 5),
            (2, 64, 14, 14, 7, 7),
        ],
    )
    @skip_batch_kernel
    def test_backward_film_kernel(self, device, B, H, X, Y, Kx, Ky):
        """Gradients match reference for FiLM (per-sample) kernels."""
        torch.manual_seed(42)

        # Reference
        x_ref = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        k_ref = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)
        y_ref = _ref_bhl(x_ref, k_ref)
        y_ref.sum().backward()

        # Custom wrapper
        x_cus = x_ref.detach().clone().requires_grad_(True)
        k_cus = k_ref.detach().clone().requires_grad_(True)
        y_cus = _custom_bhl(x_cus, k_cus)
        y_cus.sum().backward()

        torch.testing.assert_close(x_cus.grad, x_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)
        torch.testing.assert_close(k_cus.grad, k_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)


# ---------------------------------------------------------------------------
# Shortcut
# ---------------------------------------------------------------------------


class TestShortcut:
    """Shortcut (per-channel residual) semantics match the reference."""

    def test_shortcut_shared_kernel(self, device):
        """Shortcut applied correctly for shared kernel."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 2, 64, 32, 32, 7, 7
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, kernel, shortcut)
        y_custom = _custom_bhl(x, kernel, shortcut)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    @skip_batch_kernel
    def test_shortcut_film_kernel(self, device):
        """Shortcut applied correctly for FiLM kernel."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 4, 32, 16, 16, 5, 5
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, kernel, shortcut)
        y_custom = _custom_bhl(x, kernel, shortcut)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    def test_shortcut_chunked(self, device):
        """Shortcut applied correctly in chunked mode."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 2, 64, 32, 32, 7, 7
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, kernel, shortcut)
        y_chunked = _custom_bhl_chunked(x, kernel, shortcut, chunk_size=32)

        torch.testing.assert_close(y_chunked, y_ref, atol=ATOL_F32, rtol=RTOL_F32)


# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------


class TestDtype:
    """Verify the wrapper accepts non-fp32 dtypes and returns matching dtype."""

    def test_bf16_input_returns_bf16(self, device):
        """bf16 input produces bf16 output."""
        x = torch.randn(2, 32, 16, 16, device=device, dtype=torch.bfloat16)
        kernel = torch.randn(1, 32, 5, 5, device=device, dtype=torch.bfloat16)
        y = _custom_bhl(x, kernel)
        assert y.dtype == torch.bfloat16

    def test_fp32_input_returns_fp32(self, device):
        """fp32 input produces fp32 output."""
        x = torch.randn(2, 32, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 32, 5, 5, device=device, dtype=torch.float32)
        y = _custom_bhl(x, kernel)
        assert y.dtype == torch.float32

    def test_bf16_matches_fp32_reference(self, device):
        """bf16 wrapper output is close to fp32 reference (within mixed-precision tolerance)."""
        torch.manual_seed(42)
        x_fp32 = torch.randn(2, 32, 16, 16, device=device, dtype=torch.float32)
        k_fp32 = torch.randn(1, 32, 5, 5, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x_fp32, k_fp32)
        y_bf16 = _custom_bhl(x_fp32.bfloat16(), k_fp32.bfloat16())

        torch.testing.assert_close(y_bf16.float(), y_ref, atol=5e-2, rtol=1e-2)

    def test_bf16_chunked(self, device):
        """Chunked variant also handles bf16 correctly."""
        x = torch.randn(2, 64, 16, 16, device=device, dtype=torch.bfloat16)
        kernel = torch.randn(1, 64, 5, 5, device=device, dtype=torch.bfloat16)
        y = _custom_bhl_chunked(x, kernel, chunk_size=32)
        assert y.dtype == torch.bfloat16
