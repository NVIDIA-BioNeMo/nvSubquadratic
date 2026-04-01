"""Tests for the dual-centered FP16 circular FFT convolution (1D, 2D, 3D).

Validates the implementations in ``circular_fftconv_fp16.py`` against the
FP32 reference implementations from ``circular_fftconv.py``.  Each test
verifies:

1. **No NaNs** — the dual-centering technique prevents overflow.
2. **Low relative error** — mean relative error vs. FP32 stays below 5%
   across a range of input means/stds (including pathological cases like
   ``mean=50, std=5`` that overflow in naive FP16).
3. **API contract** — output dtype/shape match input, shortcut is applied,
   ``_w_reshape`` wrappers give equivalent results.

Requires a CUDA GPU.  Run via::

    conda run -n nv-subq pytest tests/test_circular_fftconv_fp16.py -v

Or inside the container for GPU tests::

    srun --gres=gpu:1 ... pytest tests/test_circular_fftconv_fp16.py -v
"""

import pytest
import torch

from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_fp32_bhl,
    circular_fftconv2d_fp32_bhl,
    circular_fftconv3d_fp32_bhl,
)
from nvsubquadratic.ops.circular_fftconv_fp16 import (
    circular_fftconv1d_fp16_bhl,
    circular_fftconv1d_fp16_bhl_w_reshape,
    circular_fftconv2d_fp16_bhl,
    circular_fftconv2d_fp16_bhl_w_reshape,
    circular_fftconv3d_fp16_bhl,
    circular_fftconv3d_fp16_bhl_w_reshape,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"

# (mean, std) pairs chosen to exercise the overflow paths that the dual-
# centering technique was designed to fix.  (0,1) is the benign baseline;
# (50,5) causes DC-bin overflow in naive FP16 even with ortho normalization.
MEAN_STD_COMBOS = [(0.0, 1.0), (1.0, 1.0), (10.0, 1.0), (50.0, 5.0)]

# 5% relative error tolerance — generous; typical is <0.1% for most shapes.
REL_ERR_TOL = 0.05


def _rel_err(y_fp16: torch.Tensor, y_ref: torch.Tensor) -> float:
    """Mean relative error, clamped to avoid div-by-zero."""
    return (y_fp16 - y_ref).abs().div(y_ref.abs().clamp(min=1e-6)).mean().item()


# ─── 1D ──────────────────────────────────────────────────────────────────────


class TestCircularFftconv1dFp16:
    """Tests for 1D fp16 circular FFT convolution."""

    @pytest.mark.parametrize("mean,std", MEAN_STD_COMBOS)
    @pytest.mark.parametrize("L,K", [(256, 255), (256, 256), (1024, 1023), (4096, 4095)])
    def test_no_nan_and_close_to_fp32(self, mean, std, L, K):
        B, H = 2, 32
        torch.manual_seed(42)
        x = (torch.randn(B, H, L, device=DEVICE) * std + mean).float()
        k = (torch.randn(1, H, K, device=DEVICE) * std + mean).float()
        sc = torch.randn(H, device=DEVICE, dtype=torch.float32)

        y_ref = circular_fftconv1d_fp32_bhl(x, k, sc)
        y_fp16 = circular_fftconv1d_fp16_bhl(x, k, sc)

        assert not y_fp16.isnan().any(), f"NaN detected (mean={mean}, std={std}, L={L}, K={K})"
        assert _rel_err(y_fp16, y_ref) < REL_ERR_TOL, (
            f"Relative error too large (mean={mean}, std={std}, L={L}, K={K})"
        )

    def test_output_dtype_matches_input(self):
        torch.manual_seed(0)
        x = torch.randn(2, 16, 256, device=DEVICE, dtype=torch.float32)
        k = torch.randn(1, 16, 255, device=DEVICE, dtype=torch.float32)
        y = circular_fftconv1d_fp16_bhl(x, k)
        assert y.dtype == x.dtype

    def test_output_shape(self):
        torch.manual_seed(0)
        x = torch.randn(4, 32, 512, device=DEVICE, dtype=torch.float32)
        k = torch.randn(1, 32, 511, device=DEVICE, dtype=torch.float32)
        y = circular_fftconv1d_fp16_bhl(x, k)
        assert y.shape == x.shape

    def test_w_reshape_matches_bhl(self):
        """The _w_reshape wrapper should give the same result as manual reshape + bhl."""
        torch.manual_seed(0)
        B, H, L, K = 2, 16, 256, 255
        x_bhl = torch.randn(B, H, L, device=DEVICE, dtype=torch.float32)
        k_bhl = torch.randn(1, H, K, device=DEVICE, dtype=torch.float32)
        sc = torch.randn(H, device=DEVICE, dtype=torch.float32)

        y_bhl = circular_fftconv1d_fp16_bhl(x_bhl, k_bhl, sc)

        x_blh = x_bhl.permute(0, 2, 1).contiguous()
        k_blh = k_bhl.permute(0, 2, 1).contiguous()
        y_blh = circular_fftconv1d_fp16_bhl_w_reshape(x_blh, k_blh, sc)
        y_from_blh = y_blh.permute(0, 2, 1)

        torch.testing.assert_close(y_bhl, y_from_blh, atol=1e-4, rtol=1e-3)

    def test_shortcut_applied(self):
        torch.manual_seed(0)
        x = torch.randn(2, 16, 256, device=DEVICE, dtype=torch.float32)
        k = torch.randn(1, 16, 255, device=DEVICE, dtype=torch.float32)
        sc = torch.randn(16, device=DEVICE, dtype=torch.float32)

        y_no_sc = circular_fftconv1d_fp16_bhl(x, k)
        y_sc = circular_fftconv1d_fp16_bhl(x, k, sc)
        assert not torch.allclose(y_no_sc, y_sc), "Shortcut should change the output"


# ─── 2D ──────────────────────────────────────────────────────────────────────


class TestCircularFftconv2dFp16:
    """Tests for 2D fp16 circular FFT convolution."""

    @pytest.mark.parametrize("mean,std", MEAN_STD_COMBOS)
    @pytest.mark.parametrize(
        "X,Y,Kx,Ky",
        [
            (32, 32, 31, 31),
            (32, 32, 32, 32),
            (64, 64, 63, 63),
            (128, 128, 127, 127),
        ],
    )
    def test_no_nan_and_close_to_fp32(self, mean, std, X, Y, Kx, Ky):
        B, H = 2, 8
        torch.manual_seed(42)
        x = (torch.randn(B, H, X, Y, device=DEVICE) * std + mean).float()
        k = (torch.randn(1, H, Kx, Ky, device=DEVICE) * std + mean).float()
        sc = torch.randn(H, device=DEVICE, dtype=torch.float32)

        y_ref = circular_fftconv2d_fp32_bhl(x, k, sc)
        y_fp16 = circular_fftconv2d_fp16_bhl(x, k, sc)

        assert not y_fp16.isnan().any(), f"NaN detected (mean={mean}, X={X}, K={Kx})"
        assert _rel_err(y_fp16, y_ref) < REL_ERR_TOL, f"Relative error too large (mean={mean}, X={X}, K={Kx})"

    def test_output_dtype_and_shape(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 64, 64, device=DEVICE, dtype=torch.float32)
        k = torch.randn(1, 8, 63, 63, device=DEVICE, dtype=torch.float32)
        y = circular_fftconv2d_fp16_bhl(x, k)
        assert y.dtype == x.dtype
        assert y.shape == x.shape

    def test_w_reshape_matches_bhl(self):
        torch.manual_seed(0)
        B, H, X, Y, Kx, Ky = 2, 8, 32, 32, 31, 31
        x_bhl = torch.randn(B, H, X, Y, device=DEVICE, dtype=torch.float32)
        k_bhl = torch.randn(1, H, Kx, Ky, device=DEVICE, dtype=torch.float32)
        sc = torch.randn(H, device=DEVICE, dtype=torch.float32)

        y_bhl = circular_fftconv2d_fp16_bhl(x_bhl, k_bhl, sc)

        x_blh = x_bhl.permute(0, 2, 3, 1).contiguous()
        k_blh = k_bhl.permute(0, 2, 3, 1).contiguous()
        y_blh = circular_fftconv2d_fp16_bhl_w_reshape(x_blh, k_blh, sc)
        y_from_blh = y_blh.permute(0, 3, 1, 2)

        torch.testing.assert_close(y_bhl, y_from_blh, atol=1e-4, rtol=1e-3)

    def test_shortcut_applied(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 32, 32, device=DEVICE, dtype=torch.float32)
        k = torch.randn(1, 8, 31, 31, device=DEVICE, dtype=torch.float32)
        sc = torch.randn(8, device=DEVICE, dtype=torch.float32)

        y_no_sc = circular_fftconv2d_fp16_bhl(x, k)
        y_sc = circular_fftconv2d_fp16_bhl(x, k, sc)
        assert not torch.allclose(y_no_sc, y_sc), "Shortcut should change the output"


# ─── 3D ──────────────────────────────────────────────────────────────────────


class TestCircularFftconv3dFp16:
    """Tests for 3D fp16 circular FFT convolution."""

    @pytest.mark.parametrize("mean,std", MEAN_STD_COMBOS)
    @pytest.mark.parametrize(
        "X,Y,Z,Kx,Ky,Kz",
        [
            (16, 16, 16, 15, 15, 15),
            (16, 16, 16, 16, 16, 16),
            (32, 32, 32, 31, 31, 31),
        ],
    )
    def test_no_nan_and_close_to_fp32(self, mean, std, X, Y, Z, Kx, Ky, Kz):
        B, H = 2, 4
        torch.manual_seed(42)
        x = (torch.randn(B, H, X, Y, Z, device=DEVICE) * std + mean).float()
        k = (torch.randn(1, H, Kx, Ky, Kz, device=DEVICE) * std + mean).float()
        sc = torch.randn(H, device=DEVICE, dtype=torch.float32)

        y_ref = circular_fftconv3d_fp32_bhl(x, k, sc)
        y_fp16 = circular_fftconv3d_fp16_bhl(x, k, sc)

        assert not y_fp16.isnan().any(), f"NaN detected (mean={mean}, X={X}, K={Kx})"
        assert _rel_err(y_fp16, y_ref) < REL_ERR_TOL, f"Relative error too large (mean={mean}, X={X}, K={Kx})"

    def test_output_dtype_and_shape(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16, 16, 16, device=DEVICE, dtype=torch.float32)
        k = torch.randn(1, 4, 15, 15, 15, device=DEVICE, dtype=torch.float32)
        y = circular_fftconv3d_fp16_bhl(x, k)
        assert y.dtype == x.dtype
        assert y.shape == x.shape

    def test_w_reshape_matches_bhl(self):
        torch.manual_seed(0)
        B, H = 2, 4
        X, Y, Z, Kx, Ky, Kz = 16, 16, 16, 15, 15, 15
        x_bhl = torch.randn(B, H, X, Y, Z, device=DEVICE, dtype=torch.float32)
        k_bhl = torch.randn(1, H, Kx, Ky, Kz, device=DEVICE, dtype=torch.float32)
        sc = torch.randn(H, device=DEVICE, dtype=torch.float32)

        y_bhl = circular_fftconv3d_fp16_bhl(x_bhl, k_bhl, sc)

        x_blh = x_bhl.permute(0, 2, 3, 4, 1).contiguous()
        k_blh = k_bhl.permute(0, 2, 3, 4, 1).contiguous()
        y_blh = circular_fftconv3d_fp16_bhl_w_reshape(x_blh, k_blh, sc)
        y_from_blh = y_blh.permute(0, 4, 1, 2, 3)

        torch.testing.assert_close(y_bhl, y_from_blh, atol=1e-3, rtol=1e-2)

    def test_shortcut_applied(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16, 16, 16, device=DEVICE, dtype=torch.float32)
        k = torch.randn(1, 4, 15, 15, 15, device=DEVICE, dtype=torch.float32)
        sc = torch.randn(4, device=DEVICE, dtype=torch.float32)

        y_no_sc = circular_fftconv3d_fp16_bhl(x, k)
        y_sc = circular_fftconv3d_fp16_bhl(x, k, sc)
        assert not torch.allclose(y_no_sc, y_sc), "Shortcut should change the output"


# ─── Geo cache ────────────────────────────────────────────────────────────────


class TestCenteringCorrectionCache:
    """Tests for the geometry correction caches."""

    def test_2d_cache_returns_none_when_no_correction_needed(self):
        from nvsubquadratic.ops.circular_fftconv_fp16 import _centering_geo_cache_2d

        assert _centering_geo_cache_2d.get(32, 32, 32, 32, torch.device(DEVICE)) is None

    def test_3d_cache_returns_none_when_no_correction_needed(self):
        from nvsubquadratic.ops.circular_fftconv_fp16 import _centering_geo_cache_3d

        assert _centering_geo_cache_3d.get(16, 16, 16, 16, 16, 16, torch.device(DEVICE)) is None

    def test_2d_cache_returns_tensor_when_correction_needed(self):
        from nvsubquadratic.ops.circular_fftconv_fp16 import _centering_geo_cache_2d

        geo = _centering_geo_cache_2d.get(31, 31, 32, 32, torch.device(DEVICE))
        assert geo is not None
        assert geo.shape == (32, 17)  # [X, Y//2+1]
        assert geo.dtype == torch.complex64

    def test_3d_cache_returns_tensor_when_correction_needed(self):
        from nvsubquadratic.ops.circular_fftconv_fp16 import _centering_geo_cache_3d

        geo = _centering_geo_cache_3d.get(15, 15, 15, 16, 16, 16, torch.device(DEVICE))
        assert geo is not None
        assert geo.shape == (16, 16, 9)  # [X, Y, Z//2+1]
        assert geo.dtype == torch.complex64

    def test_2d_cache_hit(self):
        """Second call with same args should return the same tensor object (cache hit)."""
        from nvsubquadratic.ops.circular_fftconv_fp16 import _centering_geo_cache_2d

        dev = torch.device(DEVICE)
        geo1 = _centering_geo_cache_2d.get(63, 63, 64, 64, dev)
        geo2 = _centering_geo_cache_2d.get(63, 63, 64, 64, dev)
        assert geo1 is geo2
