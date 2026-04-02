# TODO: Add license header here


"""Tests for circular (periodic) FFT convolution operators (1D, 2D, 3D).

These tests verify that circular FFT convolutions:
1. Produce outputs matching a reference circular-padded spatial conv within tolerance
2. Preserve correct output shapes for all dimensions
3. Handle both BHL and BLH (w_reshape) layouts
4. Work with and without the per-channel shortcut
5. Reject mismatched shortcut dtypes
6. Preserve caller dtype (fp32, bf16, fp16 inputs → same dtype output)
7. Give identical results for phase-shift vs spatial-roll alignment

All tests require CUDA (cuFFT for GPU FFT).

See tests/README.md for test suites, markers, and SLURM usage.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_fp32_bhl,
    circular_fftconv1d_fp32_bhl_w_reshape,
    circular_fftconv2d_fp32_bhl,
    circular_fftconv2d_fp32_bhl_w_reshape,
    circular_fftconv3d_fp32_bhl,
    circular_fftconv3d_fp32_bhl_w_reshape,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FFT tests")

# fp32 circular conv should match spatial reference to high precision.
# K==L (full-size kernel) cases accumulate more floating-point error than small-K cases.
RTOL = 5e-4
ATOL = 5e-4
RTOL_BWD = 1e-3
ATOL_BWD = 1e-3


@pytest.fixture
def device() -> str:
    return "cuda"


# ---------------------------------------------------------------------------
#  Reference helpers — spatial depthwise conv with circular padding
# ---------------------------------------------------------------------------


def _ref_circular_conv1d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Reference 1D circular convolution via F.conv1d with explicit circular padding."""
    _, H, _L = x.shape
    _, _, K = kernel.shape
    k_for_conv = kernel.squeeze(0).unsqueeze(1)  # [H, 1, K]
    k_flipped = torch.flip(k_for_conv, dims=[-1])
    pad_left = K // 2
    pad_right = K - 1 - K // 2
    padded = F.pad(x, (pad_left, pad_right), mode="circular")
    return F.conv1d(padded, k_flipped, groups=H, padding=0)


def _ref_circular_conv2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Reference 2D circular convolution via F.conv2d with explicit circular padding."""
    _, H, _X, _Y = x.shape
    _, _, K_x, K_y = kernel.shape
    k_for_conv = kernel.squeeze(0).unsqueeze(1)  # [H, 1, Kx, Ky]
    k_flipped = torch.flip(k_for_conv, dims=[-1, -2])
    pw_l, pw_r = K_y // 2, K_y - 1 - K_y // 2
    ph_t, ph_b = K_x // 2, K_x - 1 - K_x // 2
    padded = F.pad(x, (pw_l, pw_r, ph_t, ph_b), mode="circular")
    return F.conv2d(padded, k_flipped, groups=H, padding=0)


def _ref_circular_conv3d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Reference 3D circular convolution via F.conv3d with explicit circular padding."""
    _, H, _X, _Y, _Z = x.shape
    _, _, Kx, Ky, Kz = kernel.shape
    k_for_conv = kernel.squeeze(0).unsqueeze(1)  # [H, 1, Kx, Ky, Kz]
    k_flipped = torch.flip(k_for_conv, dims=[-1, -2, -3])
    pz_l, pz_r = Kz // 2, Kz - 1 - Kz // 2
    py_t, py_b = Ky // 2, Ky - 1 - Ky // 2
    px_f, px_k = Kx // 2, Kx - 1 - Kx // 2
    padded = F.pad(x, (pz_l, pz_r, py_t, py_b, px_f, px_k), mode="circular")
    return F.conv3d(padded, k_flipped, groups=H, padding=0)


###############################################################################
# 1D
###############################################################################


class TestCircularFFTConv1D:
    """Tests for 1D circular FFT convolution (fp32)."""

    @pytest.mark.parametrize(
        "B,H,L,K",
        [
            (2, 16, 64, 7),
            (2, 32, 128, 32),
            (1, 8, 256, 256),
            (4, 16, 64, 15),
        ],
    )
    def test_vs_spatial_reference(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """Circular FFT conv matches spatial circular-padded F.conv1d."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)

        y_ref = _ref_circular_conv1d(x, kernel)
        y_fft = circular_fftconv1d_fp32_bhl(x, kernel)

        assert y_fft.shape == y_ref.shape
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL, atol=ATOL)

    def test_with_shortcut(self, device: str) -> None:
        """Shortcut residual is correctly added."""
        torch.manual_seed(42)
        B, H, L, K = 2, 16, 64, 7
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_no_sc = circular_fftconv1d_fp32_bhl(x, kernel, None)
        y_sc = circular_fftconv1d_fp32_bhl(x, kernel, shortcut)

        expected = y_no_sc + rearrange(shortcut, "h -> 1 h 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=1e-5, atol=1e-5)

    def test_no_shortcut(self, device: str) -> None:
        """Works without shortcut (no crash)."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 16, 7, device=device, dtype=torch.float32)

        y = circular_fftconv1d_fp32_bhl(x, kernel, None)
        assert y.shape == (2, 16, 64)
        assert y.dtype == torch.float32

    def test_rejects_mismatched_shortcut_dtype(self, device: str) -> None:
        """Mismatched shortcut dtype raises AssertionError."""
        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
        kernel = torch.randn(1, 16, 7, device=device, dtype=torch.bfloat16)
        shortcut_bad = torch.randn(16, device=device, dtype=torch.float32)

        with pytest.raises(AssertionError, match="shortcut.dtype"):
            circular_fftconv1d_fp32_bhl(x, kernel, shortcut_bad)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_preserves_caller_dtype(self, device: str, dtype: torch.dtype) -> None:
        """Output dtype matches input dtype."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64, device=device, dtype=dtype)
        kernel = torch.randn(1, 16, 7, device=device, dtype=dtype)
        shortcut = torch.randn(16, device=device, dtype=dtype)

        y = circular_fftconv1d_fp32_bhl(x, kernel, shortcut)
        assert y.dtype == dtype

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly and matches direct BHL call."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 64, 16, device=device, dtype=torch.float32)  # [B, L, H]
        k_blh = torch.randn(1, 7, 16, device=device, dtype=torch.float32)  # [1, K, H]
        shortcut = torch.randn(16, device=device, dtype=torch.float32)

        y_wrap = circular_fftconv1d_fp32_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y_wrap.shape == (2, 64, 16)  # [B, L, H]

        # Compare with manual reshape + direct call
        x_bhl = rearrange(x_blh, "b l h -> b h l")
        k_bhl = rearrange(k_blh, "b k h -> b h k")
        y_direct = circular_fftconv1d_fp32_bhl(x_bhl, k_bhl, shortcut)
        y_direct_blh = rearrange(y_direct, "b h l -> b l h")

        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    def test_phase_shift_vs_roll(self, device: str) -> None:
        """Phase-shift and spatial-roll alignment produce identical results."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 128, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 16, 15, device=device, dtype=torch.float32)
        shortcut = torch.randn(16, device=device, dtype=torch.float32)

        y_phase = circular_fftconv1d_fp32_bhl(x, kernel, shortcut, use_phase_shift=True)
        y_roll = circular_fftconv1d_fp32_bhl(x, kernel, shortcut, use_phase_shift=False)

        torch.testing.assert_close(y_phase, y_roll, rtol=1e-5, atol=1e-5)

    def test_batched_kernel(self, device: str) -> None:
        """Supports batched kernels [B, H, K]."""
        torch.manual_seed(42)
        B, H, L, K = 2, 16, 64, 7
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, K, device=device, dtype=torch.float32)

        y = circular_fftconv1d_fp32_bhl(x, kernel, None)
        assert y.shape == (B, H, L)

    @pytest.mark.parametrize(
        "B,H,L,K",
        [(2, 8, 64, 15), (1, 16, 128, 7)],
    )
    def test_backward_vs_reference(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """Gradients w.r.t. x and kernel match spatial circular reference."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)

        y = circular_fftconv1d_fp32_bhl(x, kernel)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_circular_conv1d(x_ref, k_ref)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


###############################################################################
# 2D
###############################################################################


class TestCircularFFTConv2D:
    """Tests for 2D circular FFT convolution (fp32)."""

    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky",
        [
            (2, 8, 32, 32, 7, 7),
            (2, 16, 64, 64, 15, 15),
            (1, 8, 32, 32, 32, 32),
            (4, 8, 16, 16, 5, 5),
        ],
    )
    def test_vs_spatial_reference(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """Circular FFT conv matches spatial circular-padded F.conv2d."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_circular_conv2d(x, kernel)
        y_fft = circular_fftconv2d_fp32_bhl(x, kernel)

        assert y_fft.shape == y_ref.shape
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL, atol=ATOL)

    def test_with_shortcut(self, device: str) -> None:
        """Shortcut residual is correctly added."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 2, 8, 32, 32, 7, 7
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_no_sc = circular_fftconv2d_fp32_bhl(x, kernel, None)
        y_sc = circular_fftconv2d_fp32_bhl(x, kernel, shortcut)

        expected = y_no_sc + rearrange(shortcut, "h -> 1 h 1 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=1e-5, atol=1e-5)

    def test_no_shortcut(self, device: str) -> None:
        """Works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32, 32, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 8, 7, 7, device=device, dtype=torch.float32)

        y = circular_fftconv2d_fp32_bhl(x, kernel, None)
        assert y.shape == (2, 8, 32, 32)
        assert y.dtype == torch.float32

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 32, 32, 8, device=device, dtype=torch.float32)  # [B, X, Y, H]
        k_blh = torch.randn(1, 7, 7, 8, device=device, dtype=torch.float32)
        shortcut = torch.randn(8, device=device, dtype=torch.float32)

        y_wrap = circular_fftconv2d_fp32_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y_wrap.shape == (2, 32, 32, 8)

        x_bhl = rearrange(x_blh, "b x y h -> b h x y")
        k_bhl = rearrange(k_blh, "b kx ky h -> b h kx ky")
        y_direct = circular_fftconv2d_fp32_bhl(x_bhl, k_bhl, shortcut)
        y_direct_blh = rearrange(y_direct, "b h x y -> b x y h")

        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    def test_phase_shift_vs_roll(self, device: str) -> None:
        """Phase-shift and spatial-roll alignment produce identical results."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32, 32, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 8, 7, 7, device=device, dtype=torch.float32)

        y_phase = circular_fftconv2d_fp32_bhl(x, kernel, None, use_phase_shift=True)
        y_roll = circular_fftconv2d_fp32_bhl(x, kernel, None, use_phase_shift=False)

        torch.testing.assert_close(y_phase, y_roll, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky",
        [(2, 4, 16, 16, 5, 5), (1, 8, 32, 32, 7, 7)],
    )
    def test_backward_vs_reference(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """Gradients w.r.t. x and kernel match spatial circular reference."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)

        y = circular_fftconv2d_fp32_bhl(x, kernel)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_circular_conv2d(x_ref, k_ref)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


###############################################################################
# 3D
###############################################################################


class TestCircularFFTConv3D:
    """Tests for 3D circular FFT convolution (fp32)."""

    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz",
        [
            (2, 4, 16, 16, 16, 5, 5, 5),
            (1, 8, 16, 16, 16, 3, 3, 3),
            (2, 4, 16, 16, 16, 16, 16, 16),
        ],
    )
    def test_vs_spatial_reference(
        self, device: str, B: int, H: int, X: int, Y: int, Z: int, Kx: int, Ky: int, Kz: int
    ) -> None:
        """Circular FFT conv matches spatial circular-padded F.conv3d."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32)

        y_ref = _ref_circular_conv3d(x, kernel)
        y_fft = circular_fftconv3d_fp32_bhl(x, kernel)

        assert y_fft.shape == y_ref.shape
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL, atol=ATOL)

    def test_with_shortcut(self, device: str) -> None:
        """Shortcut residual is correctly added."""
        torch.manual_seed(42)
        B, H, X, Y, Z = 2, 4, 16, 16, 16
        Kx, Ky, Kz = 3, 3, 3
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_no_sc = circular_fftconv3d_fp32_bhl(x, kernel, None)
        y_sc = circular_fftconv3d_fp32_bhl(x, kernel, shortcut)

        expected = y_no_sc + rearrange(shortcut, "h -> 1 h 1 1 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=1e-5, atol=1e-5)

    def test_no_shortcut(self, device: str) -> None:
        """Works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 4, 3, 3, 3, device=device, dtype=torch.float32)

        y = circular_fftconv3d_fp32_bhl(x, kernel, None)
        assert y.shape == (2, 4, 16, 16, 16)
        assert y.dtype == torch.float32

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 16, 16, 16, 4, device=device, dtype=torch.float32)
        k_blh = torch.randn(1, 3, 3, 3, 4, device=device, dtype=torch.float32)
        shortcut = torch.randn(4, device=device, dtype=torch.float32)

        y_wrap = circular_fftconv3d_fp32_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y_wrap.shape == (2, 16, 16, 16, 4)

        x_bhl = rearrange(x_blh, "b x y z h -> b h x y z")
        k_bhl = rearrange(k_blh, "b kx ky kz h -> b h kx ky kz")
        y_direct = circular_fftconv3d_fp32_bhl(x_bhl, k_bhl, shortcut)
        y_direct_blh = rearrange(y_direct, "b h x y z -> b x y z h")

        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    def test_phase_shift_vs_roll(self, device: str) -> None:
        """Phase-shift and spatial-roll alignment produce identical results."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 4, 5, 5, 5, device=device, dtype=torch.float32)

        y_phase = circular_fftconv3d_fp32_bhl(x, kernel, None, use_phase_shift=True)
        y_roll = circular_fftconv3d_fp32_bhl(x, kernel, None, use_phase_shift=False)

        torch.testing.assert_close(y_phase, y_roll, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz",
        [(1, 4, 8, 8, 8, 3, 3, 3), (2, 4, 16, 16, 16, 5, 5, 5)],
    )
    def test_backward_vs_reference(
        self, device: str, B: int, H: int, X: int, Y: int, Z: int, Kx: int, Ky: int, Kz: int
    ) -> None:
        """Gradients w.r.t. x and kernel match spatial circular reference."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32, requires_grad=True)

        y = circular_fftconv3d_fp32_bhl(x, kernel)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_circular_conv3d(x_ref, k_ref)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
