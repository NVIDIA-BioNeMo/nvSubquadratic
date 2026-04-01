"""Tests for FP16 FFT convolution operators (standard and chunked).

These tests verify that fp16 FFT convolutions:
1. Produce outputs matching f32 reference within expected numerical tolerance
2. Preserve correct output shapes for all dimensions (1D, 2D, 3D)
3. Handle both BHL and BLH (w_reshape) layouts
4. Work with and without the per-channel shortcut
5. Support channel chunking with fp16 precision
6. Produce consistent results across chunked and non-chunked fp16 paths
7. Support causal 1D convolutions in fp16

All tests require CUDA (cuFFT only supports fp16 FFT).

See tests/README.md for test suites, markers, and SLURM usage.
"""

from __future__ import annotations

import pytest
import torch

from nvsubquadratic.ops.fftconv import (
    causal_fftconv1d_fp32_bhl as causal_fftconv1d_f32,
)
from nvsubquadratic.ops.fftconv import (
    fftconv1d_fp32_bhl as fftconv1d_f32,
)
from nvsubquadratic.ops.fftconv import (
    fftconv1d_fp32_bhl_w_reshape as fftconv1d_f32_w_reshape,
)
from nvsubquadratic.ops.fftconv import (
    fftconv2d_fp32_bhl as fftconv2d_f32,
)
from nvsubquadratic.ops.fftconv import (
    fftconv2d_fp32_bhl_w_reshape as fftconv2d_f32_w_reshape,
)
from nvsubquadratic.ops.fftconv import (
    fftconv3d_fp32_bhl as fftconv3d_f32,
)
from nvsubquadratic.ops.fftconv import (
    fftconv3d_fp32_bhl_w_reshape as fftconv3d_f32_w_reshape,
)
from nvsubquadratic.ops.fftconv_fp16 import (
    causal_fftconv1d_fp16_bhl,
    causal_fftconv1d_fp16_bhl_chunked,
    fftconv1d_fp16_bhl,
    fftconv1d_fp16_bhl_chunked,
    fftconv1d_fp16_bhl_w_reshape,
    fftconv2d_fp16_bhl,
    fftconv2d_fp16_bhl_chunked,
    fftconv2d_fp16_bhl_w_reshape,
    fftconv3d_fp16_bhl,
    fftconv3d_fp16_bhl_chunked,
    fftconv3d_fp16_bhl_w_reshape,
)


# cuFFT fp16 is CUDA-only
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fp16 FFT (cuFFT)")

# Tolerances: fp16 has ~0.1% relative error vs f32 due to reduced precision
# and ortho normalization. These are empirically validated thresholds.
RTOL_FP16 = 0.1
ATOL_FP16 = 0.1
# Backward tolerances — fp16 backward is noisier than forward.
RTOL_FP16_BWD = 0.15
ATOL_FP16_BWD = 0.15
# Chunked fp16 vs non-chunked fp16 should be exact (same computation)
ATOL_CHUNKED = 1e-5


@pytest.fixture
def device() -> str:
    """Return CUDA device."""
    return "cuda"


###############################################################################
# 1D non-causal
###############################################################################


class TestFP16FFTConv1D:
    """Tests for 1D fp16 FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,L,K",
        [
            (2, 32, 64, 7),
            (2, 64, 256, 32),
            (1, 128, 512, 64),
            (4, 16, 128, 15),
        ],
    )
    def test_fp16_vs_f32(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """FP16 output matches f32 reference within tolerance."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_f32 = fftconv1d_f32(x, kernel, shortcut)
        y_fp16 = fftconv1d_fp16_bhl(x, kernel, shortcut)

        assert y_fp16.shape == y_f32.shape
        assert y_fp16.dtype == x.dtype
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_no_shortcut(self, device: str) -> None:
        """FP16 1D works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 32, 7, device=device, dtype=torch.float32)

        y_f32 = fftconv1d_f32(x, kernel, None)
        y_fp16 = fftconv1d_fp16_bhl(x, kernel, None)

        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly and matches f32 w_reshape."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 32, device=device, dtype=torch.float32)  # [B, L, H]
        kernel = torch.randn(1, 7, 32, device=device, dtype=torch.float32)  # [1, K, H]
        shortcut = torch.randn(32, device=device, dtype=torch.float32)

        y_f32 = fftconv1d_f32_w_reshape(x, kernel, shortcut)
        y_fp16 = fftconv1d_fp16_bhl_w_reshape(x, kernel, shortcut)

        assert y_fp16.shape == (2, 64, 32)  # [B, L, H]
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_accepts_any_input_dtype(self, device: str) -> None:
        """FP16 functions accept any input dtype and return in the caller's dtype."""
        x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 32, 7, device=device, dtype=torch.float32)

        y = fftconv1d_fp16_bhl(x, kernel, None)
        assert y.dtype == torch.float32
        assert y.shape == (2, 32, 64)

    def test_returns_in_caller_dtype(self, device: str) -> None:
        """Output dtype matches x's dtype when all inputs share the same dtype."""
        torch.manual_seed(42)

        # fp16 inputs → fp16 output
        x_fp16 = torch.randn(2, 32, 64, device=device, dtype=torch.float16)
        kernel_fp16 = torch.randn(1, 32, 7, device=device, dtype=torch.float16)
        shortcut_fp16 = torch.randn(32, device=device, dtype=torch.float16)

        y1 = fftconv1d_fp16_bhl(x_fp16, kernel_fp16, shortcut_fp16)
        assert y1.dtype == torch.float16

        # f32 inputs → f32 output
        x_f32 = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
        kernel_f32 = torch.randn(1, 32, 7, device=device, dtype=torch.float32)
        shortcut_f32 = torch.randn(32, device=device, dtype=torch.float32)
        y2 = fftconv1d_fp16_bhl(x_f32, kernel_f32, shortcut_f32)
        assert y2.dtype == torch.float32

    def test_rejects_mismatched_shortcut_dtype(self, device: str) -> None:
        """Mismatched shortcut dtype raises AssertionError."""
        x = torch.randn(2, 32, 64, device=device, dtype=torch.float16)
        kernel_fp16 = torch.randn(1, 32, 7, device=device, dtype=torch.float16)
        shortcut_f32 = torch.randn(32, device=device, dtype=torch.float32)

        with pytest.raises(AssertionError, match="shortcut.dtype"):
            fftconv1d_fp16_bhl(x, kernel_fp16, shortcut_f32)

    @pytest.mark.parametrize("chunk_size", [16, 32, 64])
    def test_chunked_matches_standard(self, device: str, chunk_size: int) -> None:
        """Chunked fp16 produces same result as non-chunked fp16."""
        torch.manual_seed(42)
        x = torch.randn(2, 128, 256, device=device, dtype=torch.float16)
        kernel = torch.randn(1, 128, 32, device=device, dtype=torch.float16)
        shortcut = torch.randn(128, device=device, dtype=torch.float16)

        y_std = fftconv1d_fp16_bhl(x, kernel, shortcut)
        y_chunk = fftconv1d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size=chunk_size)

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_CHUNKED, rtol=0)

    def test_chunked_h_not_divisible(self, device: str) -> None:
        """Chunked fp16 handles H not divisible by chunk_size."""
        torch.manual_seed(42)
        x = torch.randn(2, 100, 128, device=device, dtype=torch.float16)
        kernel = torch.randn(1, 100, 16, device=device, dtype=torch.float16)
        shortcut = torch.randn(100, device=device, dtype=torch.float16)

        y_std = fftconv1d_fp16_bhl(x, kernel, shortcut)
        y_chunk = fftconv1d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size=64)

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_CHUNKED, rtol=0)

    def test_batched_kernel(self, device: str) -> None:
        """FP16 1D supports batched kernels [B, H, K]."""
        torch.manual_seed(42)
        B, H, L, K = 2, 32, 64, 7
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, K, device=device, dtype=torch.float32)

        y_f32 = fftconv1d_f32(x, kernel, None)
        y_fp16 = fftconv1d_fp16_bhl(x, kernel, None)

        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    @pytest.mark.parametrize(
        "B,H,L,K",
        [(2, 16, 64, 15), (1, 32, 128, 7)],
    )
    def test_backward_vs_f32(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """FP16 gradients w.r.t. x and kernel match fp32 reference gradients."""
        torch.manual_seed(42)
        x1 = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)

        y1 = fftconv1d_fp16_bhl(x1, k1)
        grad_output = torch.randn_like(y1)
        y1.backward(grad_output)

        x2 = x1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        y2 = fftconv1d_f32(x2, k2)
        y2.backward(grad_output)

        torch.testing.assert_close(x1.grad, x2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)
        torch.testing.assert_close(k1.grad, k2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)


###############################################################################
# 1D causal
###############################################################################


class TestFP16CausalFFTConv1D:
    """Tests for causal 1D fp16 FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,L,K",
        [
            (2, 32, 64, 7),
            (2, 64, 256, 32),
            (1, 128, 512, 64),
        ],
    )
    def test_causal_fp16_vs_f32(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """Causal fp16 matches f32 reference within tolerance."""
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_f32 = causal_fftconv1d_f32(x, kernel, shortcut)
        y_fp16 = causal_fftconv1d_fp16_bhl(x, kernel, shortcut)

        assert y_fp16.shape == y_f32.shape
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_causal_is_causal(self, device: str) -> None:
        """Causal fp16 output at position i depends only on input[0..i].

        Perturb input at position L//2 and verify only outputs at >= L//2 change.
        """
        torch.manual_seed(42)
        B, H, L, K = 1, 16, 64, 15
        x = torch.randn(B, H, L, device=device, dtype=torch.float16)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float16)

        y_orig = causal_fftconv1d_fp16_bhl(x, kernel, None)

        x_perturbed = x.clone()
        x_perturbed[:, :, L // 2] += 1.0
        y_pert = causal_fftconv1d_fp16_bhl(x_perturbed, kernel, None)

        # Positions before the perturbation should be unchanged
        diff_before = (y_orig[..., : L // 2] - y_pert[..., : L // 2]).abs().max()
        diff_after = (y_orig[..., L // 2 :] - y_pert[..., L // 2 :]).abs().max()

        assert diff_before < 0.05, f"Causal violation: diff before perturbation = {diff_before}"
        assert diff_after > 0.01, "Perturbation had no effect after position L//2"

    @pytest.mark.parametrize("chunk_size", [16, 32])
    def test_causal_chunked(self, device: str, chunk_size: int) -> None:
        """Chunked causal fp16 matches non-chunked causal fp16."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 128, device=device, dtype=torch.float16)
        kernel = torch.randn(1, 64, 16, device=device, dtype=torch.float16)
        shortcut = torch.randn(64, device=device, dtype=torch.float16)

        y_std = causal_fftconv1d_fp16_bhl(x, kernel, shortcut)
        y_chunk = causal_fftconv1d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size=chunk_size)

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_CHUNKED, rtol=0)

    @pytest.mark.parametrize(
        "B,H,L,K",
        [(2, 16, 64, 15), (1, 32, 128, 7)],
    )
    def test_backward_vs_f32(self, device: str, B: int, H: int, L: int, K: int) -> None:
        """Causal FP16 gradients match fp32 reference gradients."""
        torch.manual_seed(42)
        x1 = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)

        y1 = causal_fftconv1d_fp16_bhl(x1, k1)
        grad_output = torch.randn_like(y1)
        y1.backward(grad_output)

        x2 = x1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        y2 = causal_fftconv1d_f32(x2, k2)
        y2.backward(grad_output)

        torch.testing.assert_close(x1.grad, x2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)
        torch.testing.assert_close(k1.grad, k2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)


###############################################################################
# 2D
###############################################################################


class TestFP16FFTConv2D:
    """Tests for 2D fp16 FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky",
        [
            (2, 32, 14, 14, 7, 7),
            (2, 64, 14, 14, 27, 27),
            (1, 128, 28, 28, 13, 13),
            (4, 16, 32, 32, 5, 5),
        ],
    )
    def test_fp16_vs_f32(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """FP16 2D output matches f32 reference within tolerance."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_f32 = fftconv2d_f32(x, kernel, shortcut)
        y_fp16 = fftconv2d_fp16_bhl(x, kernel, shortcut)

        assert y_fp16.shape == y_f32.shape
        assert y_fp16.dtype == x.dtype
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_no_shortcut(self, device: str) -> None:
        """FP16 2D works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 32, 14, 14, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 32, 7, 7, device=device, dtype=torch.float32)

        y_f32 = fftconv2d_f32(x, kernel, None)
        y_fp16 = fftconv2d_fp16_bhl(x, kernel, None)

        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly and matches f32 w_reshape."""
        torch.manual_seed(42)
        x = torch.randn(2, 14, 14, 32, device=device, dtype=torch.float32)  # [B, X, Y, H]
        kernel = torch.randn(1, 7, 7, 32, device=device, dtype=torch.float32)
        shortcut = torch.randn(32, device=device, dtype=torch.float32)

        y_f32 = fftconv2d_f32_w_reshape(x, kernel, shortcut)
        y_fp16 = fftconv2d_fp16_bhl_w_reshape(x, kernel, shortcut)

        assert y_fp16.shape == (2, 14, 14, 32)  # [B, X, Y, H]
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    @pytest.mark.parametrize("chunk_size", [16, 32, 64])
    def test_chunked_matches_standard(self, device: str, chunk_size: int) -> None:
        """Chunked fp16 2D produces same result as non-chunked fp16."""
        torch.manual_seed(42)
        x = torch.randn(2, 128, 14, 14, device=device, dtype=torch.float16)
        kernel = torch.randn(1, 128, 7, 7, device=device, dtype=torch.float16)
        shortcut = torch.randn(128, device=device, dtype=torch.float16)

        y_std = fftconv2d_fp16_bhl(x, kernel, shortcut)
        y_chunk = fftconv2d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size=chunk_size)

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_CHUNKED, rtol=0)

    def test_chunked_h_not_divisible(self, device: str) -> None:
        """Chunked fp16 2D handles H not divisible by chunk_size."""
        torch.manual_seed(42)
        x = torch.randn(2, 100, 14, 14, device=device, dtype=torch.float16)
        kernel = torch.randn(1, 100, 7, 7, device=device, dtype=torch.float16)
        shortcut = torch.randn(100, device=device, dtype=torch.float16)

        y_std = fftconv2d_fp16_bhl(x, kernel, shortcut)
        y_chunk = fftconv2d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size=64)

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_CHUNKED, rtol=0)

    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky",
        [(2, 8, 16, 16, 5, 5), (1, 16, 32, 32, 7, 7)],
    )
    def test_backward_vs_f32(self, device: str, B: int, H: int, X: int, Y: int, Kx: int, Ky: int) -> None:
        """FP16 2D gradients match fp32 reference gradients."""
        torch.manual_seed(42)
        x1 = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)

        y1 = fftconv2d_fp16_bhl(x1, k1)
        grad_output = torch.randn_like(y1)
        y1.backward(grad_output)

        x2 = x1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        y2 = fftconv2d_f32(x2, k2)
        y2.backward(grad_output)

        torch.testing.assert_close(x1.grad, x2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)
        torch.testing.assert_close(k1.grad, k2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)


###############################################################################
# 3D
###############################################################################


class TestFP16FFTConv3D:
    """Tests for 3D fp16 FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz",
        [
            (2, 16, 8, 8, 8, 5, 5, 5),
            (1, 32, 8, 16, 16, 3, 7, 7),
            (2, 8, 16, 16, 16, 3, 3, 3),
        ],
    )
    def test_fp16_vs_f32(
        self,
        device: str,
        B: int,
        H: int,
        X: int,
        Y: int,
        Z: int,
        Kx: int,
        Ky: int,
        Kz: int,
    ) -> None:
        """FP16 3D output matches f32 reference within tolerance."""
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_f32 = fftconv3d_f32(x, kernel, shortcut)
        y_fp16 = fftconv3d_fp16_bhl(x, kernel, shortcut)

        assert y_fp16.shape == y_f32.shape
        assert y_fp16.dtype == x.dtype
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_no_shortcut(self, device: str) -> None:
        """FP16 3D works without shortcut."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 8, 8, 8, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 16, 3, 3, 3, device=device, dtype=torch.float32)

        y_f32 = fftconv3d_f32(x, kernel, None)
        y_fp16 = fftconv3d_fp16_bhl(x, kernel, None)

        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    def test_w_reshape_layout(self, device: str) -> None:
        """BLH wrapper reshapes correctly and matches f32 w_reshape."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 8, 8, 16, device=device, dtype=torch.float32)  # [B, X, Y, Z, H]
        kernel = torch.randn(1, 3, 3, 3, 16, device=device, dtype=torch.float32)
        shortcut = torch.randn(16, device=device, dtype=torch.float32)

        y_f32 = fftconv3d_f32_w_reshape(x, kernel, shortcut)
        y_fp16 = fftconv3d_fp16_bhl_w_reshape(x, kernel, shortcut)

        assert y_fp16.shape == (2, 8, 8, 8, 16)  # [B, X, Y, Z, H]
        torch.testing.assert_close(y_fp16, y_f32, rtol=RTOL_FP16, atol=ATOL_FP16)

    @pytest.mark.parametrize("chunk_size", [8, 16])
    def test_chunked_matches_standard(self, device: str, chunk_size: int) -> None:
        """Chunked fp16 3D produces same result as non-chunked fp16."""
        torch.manual_seed(42)
        x = torch.randn(2, 32, 8, 8, 8, device=device, dtype=torch.float16)
        kernel = torch.randn(1, 32, 3, 3, 3, device=device, dtype=torch.float16)
        shortcut = torch.randn(32, device=device, dtype=torch.float16)

        y_std = fftconv3d_fp16_bhl(x, kernel, shortcut)
        y_chunk = fftconv3d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size=chunk_size)

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_CHUNKED, rtol=0)

    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz",
        [(1, 4, 8, 8, 8, 3, 3, 3), (2, 8, 16, 16, 16, 5, 5, 5)],
    )
    def test_backward_vs_f32(
        self, device: str, B: int, H: int, X: int, Y: int, Z: int, Kx: int, Ky: int, Kz: int
    ) -> None:
        """FP16 3D gradients match fp32 reference gradients."""
        torch.manual_seed(42)
        x1 = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32, requires_grad=True)
        k1 = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32, requires_grad=True)

        y1 = fftconv3d_fp16_bhl(x1, k1)
        grad_output = torch.randn_like(y1)
        y1.backward(grad_output)

        x2 = x1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        y2 = fftconv3d_f32(x2, k2)
        y2.backward(grad_output)

        torch.testing.assert_close(x1.grad, x2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)
        torch.testing.assert_close(k1.grad, k2.grad, rtol=RTOL_FP16_BWD, atol=ATOL_FP16_BWD)


###############################################################################
# CKConvND integration
###############################################################################


class TestCKConvNDFP16Integration:
    """Test that CKConvND correctly selects fp16 FFT functions."""

    def test_fp16_flag_selects_fp16_table(self) -> None:
        """CKConvND with use_fp16_fft=True uses the fp16 function table."""
        from nvsubquadratic.modules.ckconv_nd import (
            FFT_FUNCTIONS_FP16,
            FFT_FUNCTIONS_FP16_CHUNKED,
        )

        for padding_key in ("zero", "causal"):
            for dim in FFT_FUNCTIONS_FP16.get(padding_key, {}):
                w_reshape_fn, bhl_fn = FFT_FUNCTIONS_FP16[padding_key][dim]
                assert "fp16" in w_reshape_fn.__name__
                assert "fp16" in bhl_fn.__name__
                assert "chunked" not in w_reshape_fn.__name__
                assert "chunked" not in bhl_fn.__name__

        for padding_key in ("zero", "causal"):
            for dim in FFT_FUNCTIONS_FP16_CHUNKED.get(padding_key, {}):
                w_reshape_fn, bhl_fn = FFT_FUNCTIONS_FP16_CHUNKED[padding_key][dim]
                assert "fp16" in w_reshape_fn.__name__
                assert "fp16" in bhl_fn.__name__
                assert "chunked" in w_reshape_fn.__name__
                assert "chunked" in bhl_fn.__name__

    def test_circular_padding_warns(self) -> None:
        """CKConvND with use_fp16_fft=True and circular padding emits a warning."""
        import warnings

        from nvsubquadratic.lazy_config import LazyConfig
        from nvsubquadratic.modules.ckconv_nd import CKConvND

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CKConvND(
                data_dim=2,
                hidden_dim=32,
                kernel_cfg=LazyConfig(torch.nn.Identity)(),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="single",
                fft_padding="circular",
                use_fp16_fft=True,
            )
            assert any("power-of-2" in str(warning.message) for warning in w)


###############################################################################
# Nightly: full-model fp16 FFT validation
###############################################################################


@pytest.mark.nightly
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    "WANDB_API_KEY" not in __import__("os").environ,
    reason="WANDB_API_KEY not set (run `source .env`)",
)
@pytest.mark.skipif(
    not __import__("os").path.isdir("/shared/data/image_datasets/imagenet"),
    reason="ImageNet not found",
)
class TestNightlyFP16Validation:
    """Nightly test: validate GAP model with fp16 FFT on ImageNet.

    Downloads the best GAP checkpoint from W&B and runs validation twice
    (f32 FFT and fp16 FFT) to confirm accuracy parity.
    """

    def test_fp16_fft_accuracy_parity(self) -> None:
        """FP16 FFT produces same validation accuracy as f32 on GAP Hyena model.

        W&B run: tcji9tfx (v2 Hyena-GAP, ~81.5% top-1)
        """
        import re
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        import pytorch_lightning as pl

        from experiments.utils.checkpointing import (
            StripCompiledPrefix,
            download_checkpoint,
            load_checkpoint_state_dict,
        )
        from nvsubquadratic.lazy_config import instantiate
        from nvsubquadratic.modules.ckconv_nd import FFT_FUNCTIONS_FP16, CKConvND

        WANDB_ENTITY = "implicit-long-convs"
        WANDB_PROJECT = "nvsubquadratic"

        _SIREN_RE = re.compile(r"(\.kernel\.kernel_network)\.(\d+)\.(weight|bias)$")

        def _remap_siren(sd: dict) -> dict:
            out = {}
            for k, v in sd.items():
                m = _SIREN_RE.search(k)
                if m:
                    idx = int(m.group(2)) // 2
                    out[k[: m.start()] + f".kernel.hidden_linears.{idx}.{m.group(3)}"] = v
                else:
                    out[k] = v
            return out

        from examples.vit5_imagenet.v2.vit5_small_pretrain_hyena_gap_apex_gated_ema import (
            get_config,
        )

        config = get_config()
        config.train.do = False
        config.debug = True
        config.compile = False

        import nvsubquadratic.ops.fftconv as _fftconv

        _fftconv.COMPILE_COMPATIBLE = True

        pl.seed_everything(config.seed, workers=True)
        torch.set_float32_matmul_precision("high")

        datamodule = instantiate(config.dataset)
        datamodule.prepare_data()
        datamodule.setup()

        network = instantiate(config.net)
        model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)

        # The checkpoint (tcji9tfx) was trained before bias was removed from
        # patch_embed and out_proj (weight-decay hygiene in #74).  Re-add the
        # bias parameters so the checkpoint loads exactly and accuracy is valid.
        net = model.network
        pe = net.patch_embed
        net.patch_embed = torch.nn.Conv2d(
            pe.in_channels,
            pe.out_channels,
            kernel_size=pe.kernel_size,
            stride=pe.stride,
            padding=pe.padding,
            bias=True,
        )
        op = net.out_proj
        net.out_proj = torch.nn.Linear(op.in_features, op.out_features, bias=True)

        run_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/tcji9tfx"
        ckpt_path = download_checkpoint(run_path=run_path, alias="best")
        state_dict = load_checkpoint_state_dict(ckpt_path)
        state_dict = _remap_siren(state_dict)
        strip = StripCompiledPrefix()
        state_dict = strip(state_dict=state_dict, model=model)
        model.load_state_dict(state_dict, strict=True)

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

        # Run validation with f32 FFT (baseline)
        results_f32 = trainer.test(model, datamodule=datamodule)
        acc_f32 = results_f32[0]["test/acc"]

        # Switch all CKConvND modules to fp16 FFT
        count = 0
        for module in model.modules():
            if isinstance(module, CKConvND):
                module.use_fp16_fft = True
                eff_pad = "causal" if module.is_causal else module.fft_padding
                module.fftconv_fn, module.fftconv_fn_bhl_input = FFT_FUNCTIONS_FP16[eff_pad][module.data_dim]
                count += 1

        assert count > 0, "No CKConvND modules found — model structure may have changed"

        # Run validation with fp16 FFT
        results_fp16 = trainer.test(model, datamodule=datamodule)
        acc_fp16 = results_fp16[0]["test/acc"]

        # Accuracy should be within 0.5% — the original validation showed identical results
        assert abs(acc_f32 - acc_fp16) < 0.005, (
            f"FP16 FFT accuracy regression: f32={acc_f32:.4f}, fp16={acc_fp16:.4f}, diff={abs(acc_f32 - acc_fp16):.4f}"
        )

        # Both should exceed the known baseline (~81.5% top-1)
        assert acc_f32 >= 0.81, f"f32 baseline regression: {acc_f32:.4f} < 0.81"
        assert acc_fp16 >= 0.81, f"fp16 regression: {acc_fp16:.4f} < 0.81"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
