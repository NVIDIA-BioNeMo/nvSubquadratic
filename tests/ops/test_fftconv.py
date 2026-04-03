# TODO: Add license header here


"""Tests for standard (zero-padded) FFT convolution operators (1D, 2D, 3D).

These tests verify that FFT convolutions:
1. Forward output matches spatial ``F.conv{1,2,3}d`` with 'same' padding
2. Causal 1D forward output matches left-padded spatial convolution
3. Backward gradients (w.r.t. x and kernel) match the reference spatial conv
4. Handle both BHL and BLH (w_reshape) layouts
5. Shortcut residual is correctly added
6. Caller dtype is preserved

All tests require CUDA.

See tests/README.md for test suites, markers, and SLURM usage.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.ops.fftconv import (
    causal_fftconv1d_fp32_bhl,
    causal_fftconv1d_fp32_bhl_w_reshape,
    fftconv1d_fp32_bhl,
    fftconv1d_fp32_bhl_w_reshape,
    fftconv2d_fp32_bhl,
    fftconv2d_fp32_bhl_w_reshape,
    fftconv3d_fp32_bhl,
    fftconv3d_fp32_bhl_w_reshape,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FFT tests")

RTOL = 1e-4
ATOL = 1e-4
RTOL_BWD = 1e-3
ATOL_BWD = 1e-3


@pytest.fixture
def device() -> str:
    return "cuda"


# ---------------------------------------------------------------------------
#  Reference helpers — spatial depthwise conv (BHL layout)
# ---------------------------------------------------------------------------


def _ref_conv1d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Reference 1D 'same' convolution via F.conv1d with flipped kernel."""
    _, H, _L = x.shape
    k_for_conv = kernel.squeeze(0).unsqueeze(1)  # [H, 1, K]
    k_flipped = torch.flip(k_for_conv, dims=[-1])
    return F.conv1d(x, k_flipped, groups=H, padding="same")


def _ref_causal_conv1d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Reference 1D causal convolution via F.conv1d with left padding."""
    _, H, _L = x.shape
    _, _, K = kernel.shape
    k_for_conv = kernel.squeeze(0).unsqueeze(1)  # [H, 1, K]
    k_flipped = torch.flip(k_for_conv, dims=[-1])
    padded = F.pad(x, (K - 1, 0))
    return F.conv1d(padded, k_flipped, groups=H, padding=0)


def _ref_conv2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Reference 2D 'same' convolution via F.conv2d with flipped kernel."""
    _, H, _X, _Y = x.shape
    k_for_conv = kernel.squeeze(0).unsqueeze(1)  # [H, 1, Kx, Ky]
    k_flipped = torch.flip(k_for_conv, dims=[-1, -2])
    return F.conv2d(x, k_flipped, groups=H, padding="same")


def _ref_conv3d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Reference 3D 'same' convolution via F.conv3d with flipped kernel."""
    _, H, _X, _Y, _Z = x.shape
    k_for_conv = kernel.squeeze(0).unsqueeze(1)  # [H, 1, Kx, Ky, Kz]
    k_flipped = torch.flip(k_for_conv, dims=[-1, -2, -3])
    return F.conv3d(x, k_flipped, groups=H, padding="same")


###############################################################################
# 1D non-causal
###############################################################################


class TestFFTConv1D:
    """Tests for 1D non-causal FFT convolution (fp32)."""

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
        """FFT conv matches spatial F.conv1d with 'same' padding."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)

        y_ref = _ref_conv1d(x, kernel)
        y_fft = fftconv1d_fp32_bhl(x, kernel)

        assert y_fft.shape == y_ref.shape
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL, atol=ATOL)

    def test_with_shortcut(self, device: str) -> None:
        """Shortcut residual is correctly added."""
        torch.manual_seed(42)
        B, H, L, K = 2, 16, 64, 7
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_no_sc = fftconv1d_fp32_bhl(x, kernel, None)
        y_sc = fftconv1d_fp32_bhl(x, kernel, shortcut)

        expected = y_no_sc + rearrange(shortcut, "h -> 1 h 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=1e-5, atol=1e-5)

    def test_no_shortcut(self, device: str) -> None:
        """Works without shortcut (no crash)."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 16, 7, device=device, dtype=torch.float32)

        y = fftconv1d_fp32_bhl(x, kernel, None)
        assert y.shape == (2, 16, 64)
        assert y.dtype == torch.float32

    def test_rejects_mismatched_shortcut_dtype(self, device: str) -> None:
        """Mismatched shortcut dtype raises AssertionError."""
        x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
        kernel = torch.randn(1, 16, 7, device=device, dtype=torch.bfloat16)
        shortcut_bad = torch.randn(16, device=device, dtype=torch.float32)

        with pytest.raises(AssertionError, match=r"shortcut\.dtype"):
            fftconv1d_fp32_bhl(x, kernel, shortcut_bad)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_preserves_caller_dtype(self, device: str, dtype: torch.dtype) -> None:
        """Output dtype matches input dtype."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64, device=device, dtype=dtype)
        kernel = torch.randn(1, 16, 7, device=device, dtype=dtype)

        y = fftconv1d_fp32_bhl(x, kernel)
        assert y.dtype == dtype

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly and matches direct BHL call."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 64, 16, device=device, dtype=torch.float32)  # [B, L, H]
        k_blh = torch.randn(1, 7, 16, device=device, dtype=torch.float32)  # [1, K, H]
        shortcut = torch.randn(16, device=device, dtype=torch.float32)

        y_wrap = fftconv1d_fp32_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y_wrap.shape == (2, 64, 16)  # [B, L, H]

        x_bhl = rearrange(x_blh, "b l h -> b h l")
        k_bhl = rearrange(k_blh, "b k h -> b h k")
        y_direct = fftconv1d_fp32_bhl(x_bhl, k_bhl, shortcut)
        y_direct_blh = rearrange(y_direct, "b h l -> b l h")

        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    def test_batched_kernel(self, device: str) -> None:
        """Supports batched kernels [B, H, K]."""
        torch.manual_seed(42)
        B, H, L, K = 2, 16, 64, 7
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, K, device=device, dtype=torch.float32)

        y = fftconv1d_fp32_bhl(x, kernel, None)
        assert y.shape == (B, H, L)

    @pytest.mark.parametrize(
        "B,H,L,K",
        [(2, 8, 64, 15), (1, 16, 128, 7)],
    )
    def test_backward_vs_reference(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """Gradients w.r.t. x and kernel match spatial reference."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)

        y = fftconv1d_fp32_bhl(x, kernel)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_conv1d(x_ref, k_ref)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


###############################################################################
# 1D causal
###############################################################################


class TestCausalFFTConv1D:
    """Tests for 1D causal FFT convolution (fp32)."""

    @pytest.mark.parametrize(
        "B,H,L,K",
        [
            (2, 16, 64, 7),
            (2, 32, 128, 32),
            (1, 8, 256, 64),
        ],
    )
    def test_vs_spatial_reference(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """Causal FFT conv matches left-padded spatial F.conv1d."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)

        y_ref = _ref_causal_conv1d(x, kernel)
        y_fft = causal_fftconv1d_fp32_bhl(x, kernel)

        assert y_fft.shape == y_ref.shape
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL, atol=ATOL)

    def test_is_causal(self, device: str) -> None:
        """Output at position i depends only on input[0..i]."""
        torch.manual_seed(42)
        B, H, L, K = 1, 8, 64, 15
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)

        y_orig = causal_fftconv1d_fp32_bhl(x, kernel)

        x_pert = x.clone()
        x_pert[:, :, L // 2] += 1.0
        y_pert = causal_fftconv1d_fp32_bhl(x_pert, kernel)

        diff_before = (y_orig[..., : L // 2] - y_pert[..., : L // 2]).abs().max()
        diff_after = (y_orig[..., L // 2 :] - y_pert[..., L // 2 :]).abs().max()

        assert diff_before < 1e-5, f"Causal violation: diff before perturbation = {diff_before}"
        assert diff_after > 0.01, "Perturbation had no effect after position L//2"

    def test_with_shortcut(self, device: str) -> None:
        """Shortcut residual is correctly added."""
        torch.manual_seed(42)
        B, H, L, K = 2, 16, 64, 7
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_no_sc = causal_fftconv1d_fp32_bhl(x, kernel, None)
        y_sc = causal_fftconv1d_fp32_bhl(x, kernel, shortcut)

        expected = y_no_sc + rearrange(shortcut, "h -> 1 h 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=1e-5, atol=1e-5)

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 64, 16, device=device, dtype=torch.float32)
        k_blh = torch.randn(1, 7, 16, device=device, dtype=torch.float32)
        shortcut = torch.randn(16, device=device, dtype=torch.float32)

        y_wrap = causal_fftconv1d_fp32_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y_wrap.shape == (2, 64, 16)

    @pytest.mark.parametrize(
        "B,H,L,K",
        [(2, 8, 64, 15), (1, 16, 128, 7)],
    )
    def test_backward_vs_reference(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """Gradients w.r.t. x and kernel match causal spatial reference."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)

        y = causal_fftconv1d_fp32_bhl(x, kernel)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_causal_conv1d(x_ref, k_ref)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


###############################################################################
# 2D
###############################################################################


class TestFFTConv2D:
    """Tests for 2D FFT convolution (fp32)."""

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
        """FFT conv matches spatial F.conv2d with 'same' padding."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_conv2d(x, kernel)
        y_fft = fftconv2d_fp32_bhl(x, kernel)

        assert y_fft.shape == y_ref.shape
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL, atol=ATOL)

    def test_with_shortcut(self, device: str) -> None:
        """Shortcut residual is correctly added."""
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 2, 8, 32, 32, 7, 7
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_no_sc = fftconv2d_fp32_bhl(x, kernel, None)
        y_sc = fftconv2d_fp32_bhl(x, kernel, shortcut)

        expected = y_no_sc + rearrange(shortcut, "h -> 1 h 1 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=1e-5, atol=1e-5)

    def test_no_shortcut(self, device: str) -> None:
        """Works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32, 32, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 8, 7, 7, device=device, dtype=torch.float32)

        y = fftconv2d_fp32_bhl(x, kernel, None)
        assert y.shape == (2, 8, 32, 32)
        assert y.dtype == torch.float32

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 32, 32, 8, device=device, dtype=torch.float32)
        k_blh = torch.randn(1, 7, 7, 8, device=device, dtype=torch.float32)
        shortcut = torch.randn(8, device=device, dtype=torch.float32)

        y_wrap = fftconv2d_fp32_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y_wrap.shape == (2, 32, 32, 8)

        x_bhl = rearrange(x_blh, "b x y h -> b h x y")
        k_bhl = rearrange(k_blh, "b kx ky h -> b h kx ky")
        y_direct = fftconv2d_fp32_bhl(x_bhl, k_bhl, shortcut)
        y_direct_blh = rearrange(y_direct, "b h x y -> b x y h")

        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky",
        [(2, 4, 16, 16, 5, 5), (1, 8, 32, 32, 7, 7)],
    )
    def test_backward_vs_reference(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """Gradients w.r.t. x and kernel match spatial reference."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)

        y = fftconv2d_fp32_bhl(x, kernel)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_conv2d(x_ref, k_ref)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


###############################################################################
# 3D
###############################################################################


class TestFFTConv3D:
    """Tests for 3D FFT convolution (fp32)."""

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
        """FFT conv matches spatial F.conv3d with 'same' padding."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32)

        y_ref = _ref_conv3d(x, kernel)
        y_fft = fftconv3d_fp32_bhl(x, kernel)

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

        y_no_sc = fftconv3d_fp32_bhl(x, kernel, None)
        y_sc = fftconv3d_fp32_bhl(x, kernel, shortcut)

        expected = y_no_sc + rearrange(shortcut, "h -> 1 h 1 1 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=1e-5, atol=1e-5)

    def test_no_shortcut(self, device: str) -> None:
        """Works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, 16, 16, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 4, 3, 3, 3, device=device, dtype=torch.float32)

        y = fftconv3d_fp32_bhl(x, kernel, None)
        assert y.shape == (2, 4, 16, 16, 16)
        assert y.dtype == torch.float32

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly."""
        torch.manual_seed(42)
        x_blh = torch.randn(2, 16, 16, 16, 4, device=device, dtype=torch.float32)
        k_blh = torch.randn(1, 3, 3, 3, 4, device=device, dtype=torch.float32)
        shortcut = torch.randn(4, device=device, dtype=torch.float32)

        y_wrap = fftconv3d_fp32_bhl_w_reshape(x_blh, k_blh, shortcut)
        assert y_wrap.shape == (2, 16, 16, 16, 4)

        x_bhl = rearrange(x_blh, "b x y z h -> b h x y z")
        k_bhl = rearrange(k_blh, "b kx ky kz h -> b h kx ky kz")
        y_direct = fftconv3d_fp32_bhl(x_bhl, k_bhl, shortcut)
        y_direct_blh = rearrange(y_direct, "b h x y z -> b x y z h")

        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz",
        [(1, 4, 8, 8, 8, 3, 3, 3), (2, 4, 16, 16, 16, 5, 5, 5)],
    )
    def test_backward_vs_reference(
        self, device: str, B: int, H: int, X: int, Y: int, Z: int, Kx: int, Ky: int, Kz: int
    ) -> None:
        """Gradients w.r.t. x and kernel match spatial reference."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32, requires_grad=True)

        y = fftconv3d_fp32_bhl(x, kernel)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_conv3d(x_ref, k_ref)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
