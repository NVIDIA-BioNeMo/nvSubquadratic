# TODO: Add license header here


"""Tests for FP16 circular FFT convolution operators (1D, 2D, 3D).

These tests verify that fp16 circular FFT convolutions:
1. Produce outputs matching the fp32 circular reference within expected tolerance
2. Preserve correct output shapes for all dimensions
3. Handle both BHL and BLH (w_reshape) layouts
4. Work with and without the per-channel shortcut
5. Reject mismatched shortcut dtypes
6. Reject non-power-of-2 spatial dimensions
7. Preserve caller dtype

All tests require CUDA (cuFFT only supports fp16 FFT).

See tests/README.md for test suites, markers, and SLURM usage.
"""

from __future__ import annotations

import pytest
import torch

from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_fp32_bhl as circular_fftconv1d_f32,
)
from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv2d_fp32_bhl as circular_fftconv2d_f32,
)
from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv3d_fp32_bhl as circular_fftconv3d_f32,
)
from nvsubquadratic.ops.circular_fftconv_fp16 import (
    circular_fftconv1d_fp16_bhl,
    circular_fftconv1d_fp16_bhl_w_reshape,
    circular_fftconv2d_fp16_bhl,
    circular_fftconv2d_fp16_bhl_w_reshape,
    circular_fftconv3d_fp16_bhl,
    circular_fftconv3d_fp16_bhl_w_reshape,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fp16 FFT (cuFFT)")

# fp16 has ~0.1% relative error vs f32 due to reduced precision and ortho normalization.
# K==L (full-size kernel) cases in higher dimensions accumulate more error.
RTOL_FP16 = 0.15
ATOL_FP16 = 0.4
RTOL_FP16_BWD = 0.15
ATOL_FP16_BWD = 0.15


@pytest.fixture
def device() -> str:
    return "cuda"


###############################################################################
# 1D
###############################################################################


class TestCircularFP16Conv1D:
    """Tests for 1D fp16 circular FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,L,K",
        [
            (2, 32, 64, 7),
            (2, 16, 128, 32),
            (1, 64, 256, 64),
            (4, 16, 128, 15),
        ],
    )
    def test_fp16_vs_f32(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """FP16 output matches f32 reference within tolerance."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_f32 = circular_fftconv1d_f32(x, kernel, shortcut)
        y_fp16 = circular_fftconv1d_fp16_bhl(x, kernel, shortcut)

        assert y_fp16.shape == y_f32.shape
        assert y_fp16.dtype == x.dtype
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_no_shortcut(self, device: str) -> None:
        """FP16 1D works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 32, 7, device=device, dtype=torch.float32)

        y_f32 = circular_fftconv1d_f32(x, kernel, None)
        y_fp16 = circular_fftconv1d_fp16_bhl(x, kernel, None)

        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly and matches direct BHL call."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 64, 32, device=device, dtype=torch.float32)  # [B, L, H]
        k_blh = torch.randn(1, 7, 32, device=device, dtype=torch.float32)  # [1, K, H]
        shortcut = torch.randn(32, device=device, dtype=torch.float32)

        y = circular_fftconv1d_fp16_bhl_w_reshape(x_blh, k_blh, shortcut)

        assert y.shape == (2, 64, 32)  # [B, L, H]

    def test_returns_in_caller_dtype(self, device: str) -> None:
        """Output dtype matches x's dtype."""
        torch.manual_seed(42)

        x_f32 = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
        kernel_f32 = torch.randn(1, 32, 7, device=device, dtype=torch.float32)
        y1 = circular_fftconv1d_fp16_bhl(x_f32, kernel_f32, None)
        assert y1.dtype == torch.float32

        x_bf16 = x_f32.to(torch.bfloat16)
        kernel_bf16 = kernel_f32.to(torch.bfloat16)
        y2 = circular_fftconv1d_fp16_bhl(x_bf16, kernel_bf16, None)
        assert y2.dtype == torch.bfloat16

    def test_rejects_mismatched_shortcut_dtype(self, device: str) -> None:
        """Mismatched shortcut dtype raises AssertionError."""
        x = torch.randn(2, 32, 64, device=device, dtype=torch.float16)
        kernel = torch.randn(1, 32, 7, device=device, dtype=torch.float16)
        shortcut_bad = torch.randn(32, device=device, dtype=torch.float32)

        with pytest.raises(AssertionError, match="shortcut.dtype"):
            circular_fftconv1d_fp16_bhl(x, kernel, shortcut_bad)

    def test_rejects_non_power_of_2(self, device: str) -> None:
        """Non-power-of-2 L raises AssertionError."""
        x = torch.randn(2, 16, 100, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 16, 7, device=device, dtype=torch.float32)

        with pytest.raises(AssertionError, match="power of 2"):
            circular_fftconv1d_fp16_bhl(x, kernel, None)

    def test_phase_shift_vs_roll(self, device: str) -> None:
        """Phase-shift and spatial-roll alignment produce identical results."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 128, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 16, 15, device=device, dtype=torch.float32)
        shortcut = torch.randn(16, device=device, dtype=torch.float32)

        y_phase = circular_fftconv1d_fp16_bhl(x, kernel, shortcut, use_phase_shift=True)
        y_roll = circular_fftconv1d_fp16_bhl(x, kernel, shortcut, use_phase_shift=False)

        # Phase ramp goes through complex32, accumulating more rounding error than spatial roll.
        torch.testing.assert_close(y_phase, y_roll, rtol=0.05, atol=0.05)

    def test_batched_kernel(self, device: str) -> None:
        """FP16 1D supports batched kernels [B, H, K]."""
        torch.manual_seed(42)
        B, H, L, K = 2, 32, 64, 7
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, K, device=device, dtype=torch.float32)

        y_f32 = circular_fftconv1d_f32(x, kernel, None)
        y_fp16 = circular_fftconv1d_fp16_bhl(x, kernel, None)

        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    @pytest.mark.parametrize(
        "B,H,L,K",
        [(2, 16, 64, 15), (1, 32, 128, 7)],
    )
    def test_backward_vs_f32(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """FP16 circular 1D gradients match fp32 circular reference gradients."""
        torch.manual_seed(42)
        x1 = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)

        y1 = circular_fftconv1d_fp16_bhl(x1, k1)
        grad_output = torch.randn_like(y1)
        y1.backward(grad_output)

        x2 = x1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        y2 = circular_fftconv1d_f32(x2, k2)
        y2.backward(grad_output)

        torch.testing.assert_close(x1.grad, x2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)
        torch.testing.assert_close(k1.grad, k2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)


###############################################################################
# 2D
###############################################################################


class TestCircularFP16Conv2D:
    """Tests for 2D fp16 circular FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky",
        [
            (2, 16, 32, 32, 7, 7),
            (2, 32, 64, 64, 15, 15),
            (1, 8, 16, 16, 5, 5),
            (4, 8, 32, 32, 32, 32),
        ],
    )
    def test_fp16_vs_f32(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """FP16 2D output matches f32 reference within tolerance."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_f32 = circular_fftconv2d_f32(x, kernel, shortcut)
        y_fp16 = circular_fftconv2d_fp16_bhl(x, kernel, shortcut)

        assert y_fp16.shape == y_f32.shape
        assert y_fp16.dtype == x.dtype
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_no_shortcut(self, device: str) -> None:
        """FP16 2D works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 32, 32, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 16, 7, 7, device=device, dtype=torch.float32)

        y_f32 = circular_fftconv2d_f32(x, kernel, None)
        y_fp16 = circular_fftconv2d_fp16_bhl(x, kernel, None)

        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 32, 32, 16, device=device, dtype=torch.float32)  # [B, X, Y, H]
        k_blh = torch.randn(1, 7, 7, 16, device=device, dtype=torch.float32)
        shortcut = torch.randn(16, device=device, dtype=torch.float32)

        y = circular_fftconv2d_fp16_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y.shape == (2, 32, 32, 16)

    def test_phase_shift_vs_roll(self, device: str) -> None:
        """Phase-shift and spatial-roll alignment produce identical results."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32, 32, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 8, 7, 7, device=device, dtype=torch.float32)
        shortcut = torch.randn(8, device=device, dtype=torch.float32)

        y_phase = circular_fftconv2d_fp16_bhl(x, kernel, shortcut, use_phase_shift=True)
        y_roll = circular_fftconv2d_fp16_bhl(x, kernel, shortcut, use_phase_shift=False)

        # Phase ramp goes through complex32, accumulating more rounding error than spatial roll.
        torch.testing.assert_close(y_phase, y_roll, rtol=0.05, atol=0.05)

    def test_rejects_non_power_of_2(self, device: str) -> None:
        """Non-power-of-2 spatial dims raise AssertionError."""
        x = torch.randn(2, 8, 14, 14, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 8, 7, 7, device=device, dtype=torch.float32)

        with pytest.raises(AssertionError, match="power"):
            circular_fftconv2d_fp16_bhl(x, kernel, None)

    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky",
        [(2, 8, 16, 16, 5, 5), (1, 16, 32, 32, 7, 7)],
    )
    def test_backward_vs_f32(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """FP16 circular 2D gradients match fp32 circular reference gradients."""
        torch.manual_seed(42)
        x1 = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)

        y1 = circular_fftconv2d_fp16_bhl(x1, k1)
        grad_output = torch.randn_like(y1)
        y1.backward(grad_output)

        x2 = x1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        y2 = circular_fftconv2d_f32(x2, k2)
        y2.backward(grad_output)

        torch.testing.assert_close(x1.grad, x2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)
        torch.testing.assert_close(k1.grad, k2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)


###############################################################################
# 3D
###############################################################################


class TestCircularFP16Conv3D:
    """Tests for 3D fp16 circular FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz",
        [
            (2, 4, 16, 16, 16, 5, 5, 5),
            (1, 8, 8, 8, 8, 3, 3, 3),
            (2, 4, 16, 16, 16, 16, 16, 16),
        ],
    )
    def test_fp16_vs_f32(self, device: str, B: int, H: int, X: int, Y: int, Z: int, Kx: int, Ky: int, Kz: int) -> None:
        """FP16 3D output matches f32 reference within tolerance."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_f32 = circular_fftconv3d_f32(x, kernel, shortcut)
        y_fp16 = circular_fftconv3d_fp16_bhl(x, kernel, shortcut)

        assert y_fp16.shape == y_f32.shape
        assert y_fp16.dtype == x.dtype
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_no_shortcut(self, device: str) -> None:
        """FP16 3D works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 4, 3, 3, 3, device=device, dtype=torch.float32)

        y_f32 = circular_fftconv3d_f32(x, kernel, None)
        y_fp16 = circular_fftconv3d_fp16_bhl(x, kernel, None)

        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 16, 16, 16, 4, device=device, dtype=torch.float32)
        k_blh = torch.randn(1, 3, 3, 3, 4, device=device, dtype=torch.float32)
        shortcut = torch.randn(4, device=device, dtype=torch.float32)

        y = circular_fftconv3d_fp16_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y.shape == (2, 16, 16, 16, 4)

    def test_phase_shift_vs_roll(self, device: str) -> None:
        """Phase-shift and spatial-roll alignment produce identical results."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 4, 5, 5, 5, device=device, dtype=torch.float32)
        shortcut = torch.randn(4, device=device, dtype=torch.float32)

        y_phase = circular_fftconv3d_fp16_bhl(x, kernel, shortcut, use_phase_shift=True)
        y_roll = circular_fftconv3d_fp16_bhl(x, kernel, shortcut, use_phase_shift=False)

        # Phase ramp goes through complex32, accumulating more rounding error than spatial roll.
        torch.testing.assert_close(y_phase, y_roll, rtol=0.1, atol=0.1)

    def test_rejects_non_power_of_2(self, device: str) -> None:
        """Non-power-of-2 spatial dims raise AssertionError."""
        x = torch.randn(2, 4, 12, 12, 12, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 4, 3, 3, 3, device=device, dtype=torch.float32)

        with pytest.raises(AssertionError, match="power"):
            circular_fftconv3d_fp16_bhl(x, kernel, None)

    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz",
        [(1, 4, 8, 8, 8, 3, 3, 3), (2, 4, 16, 16, 16, 5, 5, 5)],
    )
    def test_backward_vs_f32(
        self, device: str, B: int, H: int, X: int, Y: int, Z: int, Kx: int, Ky: int, Kz: int
    ) -> None:
        """FP16 circular 3D gradients match fp32 circular reference gradients."""
        torch.manual_seed(42)
        x1 = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32, requires_grad=True)

        y1 = circular_fftconv3d_fp16_bhl(x1, k1)
        grad_output = torch.randn_like(y1)
        y1.backward(grad_output)

        x2 = x1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        y2 = circular_fftconv3d_f32(x2, k2)
        y2.backward(grad_output)

        torch.testing.assert_close(x1.grad, x2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)
        torch.testing.assert_close(k1.grad, k2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
