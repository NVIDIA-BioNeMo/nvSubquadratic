# TODO: Add license header here


"""Tests for subquadratic_ops_torch.fft_conv2d CUDA kernel.

Validates the GPU-accelerated FFT conv2d from the ``subquadratic_ops_torch``
package against our reference ``fftconv2d_fp32_bhl`` implementation.

Covers:
  - Shared kernel ``[1, H, Kx, Ky]`` and ``[H, Kx, Ky]``
  - FiLM (per-sample) batched kernel ``[B, H, Kx, Ky]``
  - Various spatial sizes, kernel sizes, and channel counts
  - Backward pass (autograd gradients for x and kernel)
  - FP16 support detection
  - Per-sample independence for batched kernels

Usage (requires GPU — run inside SLURM):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/ops/test_subq_ops_fft_conv2d.py -v -o addopts=""
"""

import pytest
import torch

from tests.conftest import requires_subq_ops_v2


# Tolerances for f32 comparison between torch.fft and CUDA kernel
ATOL_F32 = 1e-3
RTOL_F32 = 1e-4

# Tolerance for gradient comparison
ATOL_GRAD = 1e-2
RTOL_GRAD = 1e-3


def _has_subq_ops() -> bool:
    """Check if subquadratic_ops_torch is installed and importable."""
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
skip_batch_kernel = requires_subq_ops_v2

pytestmark = [requires_subq_ops, requires_cuda, requires_subq_ops_v2]


@pytest.fixture
def device() -> str:
    return "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_fftconv2d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference fftconv2d_fp32_bhl from nvsubquadratic.ops.fftconv."""
    from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl

    return fftconv2d_fp32_bhl(x, kernel, shortcut)


def _subq_fftconv2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Wrapper for subquadratic_ops_torch.fft_conv2d."""
    from subquadratic_ops_torch.fft_conv2d import fft_conv2d

    return fft_conv2d(x, kernel)


# ---------------------------------------------------------------------------
# Correctness — shared kernel [1, H, Kx, Ky]
# ---------------------------------------------------------------------------


class TestSharedKernel:
    """Tests with a single shared kernel broadcast across the batch."""

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (2, 64, 32, 32, 7, 7),
            (4, 128, 14, 14, 7, 7),
            (2, 256, 28, 28, 13, 13),
            (1, 64, 64, 64, 15, 15),
            (2, 32, 16, 16, 3, 3),
            (1, 384, 14, 14, 14, 14),
        ],
    )
    def test_matches_reference(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """CUDA kernel output matches torch.fft reference for shared kernels."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel_4d = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_fftconv2d_bhl(x, kernel_4d, shortcut=None)
        y_subq = _subq_fftconv2d(x, kernel_4d.squeeze(0))

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL_F32, rtol=RTOL_F32)
        assert y_subq.shape == y_ref.shape

    def test_3d_kernel_shape(self, device: str) -> None:
        """CUDA kernel accepts [H, Kx, Ky] weight (3D, no batch dim)."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 32, 32, device=device, dtype=torch.float32)
        kernel_3d = torch.randn(64, 7, 7, device=device, dtype=torch.float32)

        y = _subq_fftconv2d(x, kernel_3d)
        assert y.shape == (2, 64, 32, 32)

    def test_4d_kernel_shape(self, device: str) -> None:
        """CUDA kernel accepts [1, H, Kx, Ky] weight (4D, batch=1)."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 32, 32, device=device, dtype=torch.float32)
        kernel_4d = torch.randn(1, 64, 7, 7, device=device, dtype=torch.float32)

        y = _subq_fftconv2d(x, kernel_4d)
        assert y.shape == (2, 64, 32, 32)

    def test_3d_and_4d_agree(self, device: str) -> None:
        """3D and 4D kernel shapes produce identical output."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 32, 32, device=device, dtype=torch.float32)
        kernel_3d = torch.randn(64, 7, 7, device=device, dtype=torch.float32)
        kernel_4d = kernel_3d.unsqueeze(0)

        y_3d = _subq_fftconv2d(x, kernel_3d)
        y_4d = _subq_fftconv2d(x, kernel_4d)

        torch.testing.assert_close(y_3d, y_4d, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Correctness — FiLM (per-sample batched) kernel [B, H, Kx, Ky]
# ---------------------------------------------------------------------------


class TestFiLMKernel:
    """Tests with per-sample (FiLM) batched kernels."""

    @skip_batch_kernel
    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (4, 64, 32, 32, 7, 7),
            (2, 128, 14, 14, 7, 7),
            (3, 64, 28, 28, 13, 13),
            (8, 32, 16, 16, 5, 5),
        ],
    )
    def test_matches_reference(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """CUDA kernel matches torch.fft reference for per-sample FiLM kernels."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_fftconv2d_bhl(x, kernel, shortcut=None)
        y_subq = _subq_fftconv2d(x, kernel)

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    @skip_batch_kernel
    def test_per_sample_independence(self, device: str) -> None:
        """Batched FiLM result equals running each sample individually."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 4, 64, 32, 32, 7, 7
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32)

        y_batched = _subq_fftconv2d(x, kernel)

        for i in range(B):
            y_single = _subq_fftconv2d(x[i : i + 1], kernel[i : i + 1])
            torch.testing.assert_close(
                y_batched[i : i + 1],
                y_single,
                atol=0,
                rtol=0,
                msg=f"Sample {i} differs between batched and individual execution",
            )

    @skip_batch_kernel
    def test_different_kernels_give_different_outputs(self, device: str) -> None:
        """Verify per-sample kernels actually produce distinct outputs."""
        torch.manual_seed(42)
        B, H, X, Y = 2, 32, 16, 16
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, 5, 5, device=device, dtype=torch.float32)

        y = _subq_fftconv2d(x, kernel)
        # With different random kernels per sample, outputs should differ
        assert not torch.allclose(y[0], y[1], atol=1e-3)


# ---------------------------------------------------------------------------
# Backward pass (autograd)
# ---------------------------------------------------------------------------


class TestBackward:
    """Verify autograd backward works for both x and kernel."""

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (2, 64, 32, 32, 7, 7),
            (4, 32, 14, 14, 5, 5),
        ],
    )
    def test_backward_shared_kernel(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """Gradients flow through x and shared kernel."""
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)

        y = _subq_fftconv2d(x, kernel)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "x.grad is None"
        assert kernel.grad is not None, "kernel.grad is None"
        assert x.grad.shape == x.shape
        assert kernel.grad.shape == kernel.shape
        assert x.grad.norm().item() > 0, "x.grad is all zeros"
        assert kernel.grad.norm().item() > 0, "kernel.grad is all zeros"

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (4, 64, 32, 32, 7, 7),
            (2, 32, 14, 14, 5, 5),
        ],
    )
    @skip_batch_kernel
    def test_backward_film_kernel(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """Gradients flow through x and per-sample FiLM kernel."""
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)

        y = _subq_fftconv2d(x, kernel)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "x.grad is None"
        assert kernel.grad is not None, "kernel.grad is None"
        assert x.grad.shape == x.shape
        assert kernel.grad.shape == kernel.shape

    def test_backward_matches_reference(self, device: str) -> None:
        """Gradients from CUDA kernel match torch.fft reference gradients."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 2, 32, 16, 16, 5, 5

        # Reference
        x_ref = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        k_ref = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)
        y_ref = _ref_fftconv2d_bhl(x_ref, k_ref, shortcut=None)
        y_ref.sum().backward()

        # CUDA kernel
        x_subq = x_ref.detach().clone().requires_grad_(True)
        k_subq = k_ref.detach().clone().squeeze(0).requires_grad_(True)
        y_subq = _subq_fftconv2d(x_subq, k_subq)
        y_subq.sum().backward()

        torch.testing.assert_close(x_subq.grad, x_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)
        # kernel grads have different shapes: [H, Kx, Ky] vs [1, H, Kx, Ky]
        torch.testing.assert_close(k_subq.grad, k_ref.grad.squeeze(0), atol=ATOL_GRAD, rtol=RTOL_GRAD)

    @pytest.mark.parametrize(
        "B, H, X, Y, Kx, Ky",
        [
            (4, 32, 16, 16, 5, 5),
            (2, 64, 14, 14, 7, 7),
        ],
    )
    @skip_batch_kernel
    def test_backward_matches_reference_film(
        self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int
    ) -> None:
        """Gradients from CUDA kernel match torch.fft reference for FiLM (per-sample) kernels."""
        torch.manual_seed(42)

        # Reference (torch.fft)
        x_ref = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        k_ref = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)
        y_ref = _ref_fftconv2d_bhl(x_ref, k_ref, shortcut=None)
        y_ref.sum().backward()

        # CUDA kernel (subq_ops)
        x_subq = x_ref.detach().clone().requires_grad_(True)
        k_subq = k_ref.detach().clone().requires_grad_(True)
        y_subq = _subq_fftconv2d(x_subq, k_subq)
        y_subq.sum().backward()

        torch.testing.assert_close(x_subq.grad, x_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)
        torch.testing.assert_close(k_subq.grad, k_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)


# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------


class TestDtype:
    """Test dtype support and output dtype behavior."""

    def test_f32_output_dtype(self, device: str) -> None:
        """Float32 input produces float32 output."""
        x = torch.randn(2, 32, 16, 16, device=device, dtype=torch.float32)
        k = torch.randn(32, 5, 5, device=device, dtype=torch.float32)
        y = _subq_fftconv2d(x, k)
        assert y.dtype == torch.float32

    def test_fp16_support(self, device: str) -> None:
        """Check if the CUDA kernel supports fp16 inputs."""
        x = torch.randn(2, 32, 16, 16, device=device, dtype=torch.float16)
        k = torch.randn(32, 5, 5, device=device, dtype=torch.float16)
        try:
            y = _subq_fftconv2d(x, k)
            assert y.dtype == torch.float16
        except (RuntimeError, AttributeError):
            pytest.skip("FP16 not supported in this build of subquadratic_ops_torch")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: large kernels, non-square inputs, batch=1."""

    def test_batch_one(self, device: str) -> None:
        """Single-batch input works correctly."""
        torch.manual_seed(42)
        x = torch.randn(1, 32, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(32, 5, 5, device=device, dtype=torch.float32)

        y_subq = _subq_fftconv2d(x, kernel)
        y_ref = _ref_fftconv2d_bhl(x, kernel.unsqueeze(0), shortcut=None)

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    def test_non_square_input(self, device: str) -> None:
        """Non-square spatial dimensions (e.g. 32x16)."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 32, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 64, 7, 5, device=device, dtype=torch.float32)

        y_ref = _ref_fftconv2d_bhl(x, kernel, shortcut=None)
        y_subq = _subq_fftconv2d(x, kernel.squeeze(0))

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    def test_non_square_kernel(self, device: str) -> None:
        """Non-square kernel (e.g. 3x7)."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 32, 32, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 64, 3, 7, device=device, dtype=torch.float32)

        y_ref = _ref_fftconv2d_bhl(x, kernel, shortcut=None)
        y_subq = _subq_fftconv2d(x, kernel.squeeze(0))

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    def test_kernel_same_size_as_input(self, device: str) -> None:
        """Kernel spatial dims equal to input spatial dims."""
        torch.manual_seed(42)
        x = torch.randn(2, 32, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 32, 16, 16, device=device, dtype=torch.float32)

        y_ref = _ref_fftconv2d_bhl(x, kernel, shortcut=None)
        y_subq = _subq_fftconv2d(x, kernel.squeeze(0))

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    def test_odd_spatial_dims(self, device: str) -> None:
        """Odd spatial dimensions (e.g. 15x17)."""
        torch.manual_seed(42)
        x = torch.randn(2, 32, 15, 17, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 32, 5, 5, device=device, dtype=torch.float32)

        y_ref = _ref_fftconv2d_bhl(x, kernel, shortcut=None)
        y_subq = _subq_fftconv2d(x, kernel.squeeze(0))

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    def test_single_channel(self, device: str) -> None:
        """H=1 (single channel)."""
        torch.manual_seed(42)
        x = torch.randn(2, 1, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 5, 5, device=device, dtype=torch.float32)

        y = _subq_fftconv2d(x, kernel)
        assert y.shape == (2, 1, 16, 16)

    @pytest.mark.parametrize("H", [63, 65, 127, 255])
    def test_non_power_of_two_channels(self, device: str, H: int) -> None:
        """Non-power-of-2 channel counts."""
        torch.manual_seed(42)
        x = torch.randn(2, H, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, 5, 5, device=device, dtype=torch.float32)

        y_ref = _ref_fftconv2d_bhl(x, kernel, shortcut=None)
        y_subq = _subq_fftconv2d(x, kernel.squeeze(0))

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL_F32, rtol=RTOL_F32)
