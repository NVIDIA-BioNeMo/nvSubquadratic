# TODO: Add license header here


"""Tests for mixed boundary-condition FFT convolution operators (1D/2D/3D).

These tests verify the new ``mixed_fftconv*`` ops in
:mod:`nvsubquadratic.ops.mixed_fftconv` against:

1. A spatial reference built by per-axis :func:`torch.nn.functional.pad`
   (mode ``"circular"`` for periodic axes, mode ``"constant"`` for
   non-periodic) followed by depthwise :func:`torch.nn.functional.conv`.
2. The existing :mod:`nvsubquadratic.ops.fftconv` ops when every axis is
   non-periodic.
3. The existing :mod:`nvsubquadratic.ops.circular_fftconv` ops when every
   axis is periodic.

We cover every per-axis combination in 2D and 3D, both BHL and BLH
layouts, chunked vs non-chunked, phase-shift vs spatial-roll alignment,
shortcut residuals, batched kernels, dtype preservation, gradients, and
validation errors.

All tests require CUDA (cuFFT for GPU FFT).
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_fp32_bhl,
    circular_fftconv2d_fp32_bhl,
    circular_fftconv3d_fp32_bhl,
)
from nvsubquadratic.ops.fftconv import (
    fftconv1d_fp32_bhl,
    fftconv2d_fp32_bhl,
    fftconv3d_fp32_bhl,
)
from nvsubquadratic.ops.mixed_fftconv import (
    mixed_fftconv1d_fp32_bhl,
    mixed_fftconv1d_fp32_bhl_chunked,
    mixed_fftconv1d_fp32_bhl_w_reshape,
    mixed_fftconv2d_fp32_bhl,
    mixed_fftconv2d_fp32_bhl_chunked,
    mixed_fftconv2d_fp32_bhl_w_reshape,
    mixed_fftconv3d_fp32_bhl,
    mixed_fftconv3d_fp32_bhl_chunked,
    mixed_fftconv3d_fp32_bhl_w_reshape,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FFT tests")

# fp32 tolerances. Mixed conv reuses the same FFT machinery as the existing
# linear / circular ops, so we follow their convention:
#
# - ``RTOL_SMALL_K`` / ``ATOL_SMALL_K`` (1e-4): for tests where the kernel
#   covers a small fraction of the input (K << N). Matches the existing
#   ``tests/ops/test_fftconv.py`` tolerance.
# - ``RTOL_FULL_K`` / ``ATOL_FULL_K`` (5e-4): for the K == N "full-size kernel"
#   corner, where each output is a sum of K * N random products and the
#   accumulated fp32 FFT noise is fundamentally higher (~|y| * log(F^2) * eps
#   ≈ 2e-4 with output magnitudes ~100). Matches the existing
#   ``tests/ops/test_circular_fftconv.py`` tolerance.
# - ``RTOL_BWD`` / ``ATOL_BWD`` (1e-3): for backward passes (matches both).
#
# Splitting these (instead of using a single loose 5e-4 everywhere) prevents a
# loose tolerance from masking small regressions in the common small-K regime.
RTOL_SMALL_K = 1e-4
ATOL_SMALL_K = 1e-4
RTOL_FULL_K = 5e-4
ATOL_FULL_K = 5e-4
RTOL_BWD = 1e-3
ATOL_BWD = 1e-3


@pytest.fixture
def device() -> str:
    return "cuda"


# ---------------------------------------------------------------------------
#  Reference helper — spatial depthwise conv with per-axis padding mode
# ---------------------------------------------------------------------------


def _ref_mixed_conv_nd(x: torch.Tensor, kernel: torch.Tensor, periodic: tuple[bool, ...]) -> torch.Tensor:
    """Reference N-D mixed-BC convolution via per-axis ``F.pad`` + ``F.conv{n}d``.

    Per-axis padding convention (matches the per-axis FFT recipe in
    :func:`nvsubquadratic.ops.mixed_fftconv._mixed_recipe`):

    - **Periodic axis** (circular conv): ``pad_left = K_d // 2``,
      ``pad_right = K_d - 1 - K_d // 2``. This matches the existing
      ``_ref_circular_conv*`` helpers in :mod:`tests.ops.test_circular_fftconv`.
    - **Non-periodic axis** ("same" linear conv): ``pad_left = (K_d - 1) // 2``,
      ``pad_right = K_d // 2``. This matches PyTorch's ``F.conv*(padding="same")``
      convention, which is the convention that the existing linear
      ``fftconv*`` ops implement (and that ``_mixed_recipe`` inherits via
      ``F_d = min(N_d + (K_d + 1) // 2, 2*N_d)`` + ``crop = [K//2 : K//2 + N]``).

    For odd ``K_d`` both conventions coincide; they differ by one element on
    opposite sides only for even ``K_d``.

    Modes are applied per axis (``F.pad`` only supports one mode per call);
    the order of per-axis calls does not affect the result because each
    call only touches one axis. After padding, a flipped-kernel depthwise
    convolution with ``padding=0`` produces the "same"-sized output.

    Args:
        x: ``[B, H, *spatial]`` input.
        kernel: ``[1, H, *K]`` kernel (broadcasted-across-batch case).
        periodic: per-axis periodicity flags, length matches the number of
            spatial axes.
    """
    D = len(periodic)
    assert x.ndim == 2 + D
    assert kernel.ndim == 2 + D
    assert kernel.shape[0] == 1, "Reference helper only supports shared kernel (B=1)"

    H = x.shape[1]
    k_spatial = kernel.shape[2:]

    k_for_conv = kernel.squeeze(0).unsqueeze(1)  # [H, 1, K_0, ..., K_{D-1}]
    flip_dims = tuple(range(-D, 0))
    k_flipped = torch.flip(k_for_conv, dims=flip_dims)

    x_padded = x
    for d in range(D):
        K_d = k_spatial[d]
        if periodic[d]:
            pad_left = K_d // 2
            pad_right = K_d - 1 - K_d // 2
        else:
            pad_left = (K_d - 1) // 2
            pad_right = K_d // 2
        axis_from_end = D - 1 - d
        pad_spec = [0] * (2 * D)
        pad_spec[2 * axis_from_end] = pad_left
        pad_spec[2 * axis_from_end + 1] = pad_right
        if periodic[d]:
            x_padded = F.pad(x_padded, tuple(pad_spec), mode="circular")
        else:
            x_padded = F.pad(x_padded, tuple(pad_spec), mode="constant", value=0.0)

    conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[D]
    return conv_fn(x_padded, k_flipped, groups=H, padding=0)


# ---------------------------------------------------------------------------
#  All per-axis BC combinations (for parametrising 2D / 3D tests)
# ---------------------------------------------------------------------------


_PERIODIC_COMBOS_2D = list(itertools.product([False, True], repeat=2))
_PERIODIC_COMBOS_3D = list(itertools.product([False, True], repeat=3))


###############################################################################
# 1D
###############################################################################


class TestMixedFFTConv1D:
    """Tests for 1D mixed-BC FFT convolution (fp32)."""

    @pytest.mark.parametrize("periodic", [(False,), (True,)])
    @pytest.mark.parametrize(
        "B,H,L,K",
        [
            (2, 16, 64, 7),
            (1, 8, 128, 32),
            (2, 8, 256, 15),
        ],
    )
    def test_vs_spatial_reference_small_k(
        self, device: str, B: int, H: int, L: int, K: int, periodic: tuple[bool, ...]
    ) -> None:
        # Small / medium kernel relative to input — tight tolerance.
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32)

        y_ref = _ref_mixed_conv_nd(x, kernel, periodic)
        y_fft = mixed_fftconv1d_fp32_bhl(x, kernel, periodic)

        assert y_fft.shape == y_ref.shape == (B, H, L)
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL_SMALL_K, atol=ATOL_SMALL_K)

    @pytest.mark.parametrize("periodic", [(False,), (True,)])
    @pytest.mark.parametrize(
        "B,H,L",
        [
            (1, 8, 64),
            (2, 4, 128),
        ],
    )
    def test_vs_spatial_reference_full_k(
        self, device: str, B: int, H: int, L: int, periodic: tuple[bool, ...]
    ) -> None:
        # K == L (kernel covers entire input). Looser tolerance because the
        # output accumulates O(L) random products and fp32 FFT precision is
        # the limiting factor.
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, L, device=device, dtype=torch.float32)

        y_ref = _ref_mixed_conv_nd(x, kernel, periodic)
        y_fft = mixed_fftconv1d_fp32_bhl(x, kernel, periodic)

        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL_FULL_K, atol=ATOL_FULL_K)

    def test_matches_existing_linear_when_all_false(self, device: str) -> None:
        # All-False dispatches *literally* to the same fftconv1d_fp32_bhl call,
        # so the result must be bit-identical (rtol=atol=0).
        torch.manual_seed(0)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 7, device=device, dtype=torch.float32)
        y_mixed = mixed_fftconv1d_fp32_bhl(x, k, (False,))
        y_linear = fftconv1d_fp32_bhl(x, k)
        torch.testing.assert_close(y_mixed, y_linear, rtol=0, atol=0)

    def test_matches_existing_circular_when_all_true(self, device: str) -> None:
        # All-True dispatches literally to circular_fftconv1d_fp32_bhl → bit-identical.
        torch.manual_seed(0)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 7, device=device, dtype=torch.float32)
        y_mixed = mixed_fftconv1d_fp32_bhl(x, k, (True,))
        y_circ = circular_fftconv1d_fp32_bhl(x, k)
        torch.testing.assert_close(y_mixed, y_circ, rtol=0, atol=0)

    def test_use_phase_shift_vs_roll(self, device: str) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 16, 128, device=device, dtype=torch.float32)
        k = torch.randn(1, 16, 15, device=device, dtype=torch.float32)
        for periodic in [(False,), (True,)]:
            y_phase = mixed_fftconv1d_fp32_bhl(x, k, periodic, use_phase_shift=True)
            y_roll = mixed_fftconv1d_fp32_bhl(x, k, periodic, use_phase_shift=False)
            torch.testing.assert_close(y_phase, y_roll, rtol=1e-5, atol=1e-5)

    def test_with_shortcut(self, device: str) -> None:
        # Shortcut is a deterministic per-channel scaled add after the FFT op,
        # so y_sc must equal y_no + sc * x exactly.
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64, device=device, dtype=torch.float32)
        k = torch.randn(1, 16, 7, device=device, dtype=torch.float32)
        shortcut = torch.randn(16, device=device, dtype=torch.float32)
        periodic = (True,)
        y_no = mixed_fftconv1d_fp32_bhl(x, k, periodic, shortcut=None)
        y_sc = mixed_fftconv1d_fp32_bhl(x, k, periodic, shortcut=shortcut)
        expected = y_no + rearrange(shortcut, "h -> 1 h 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=0, atol=0)

    def test_w_reshape_layout(self, device: str) -> None:
        torch.manual_seed(42)
        x_blh = torch.randn(2, 64, 16, device=device, dtype=torch.float32)
        k_blh = torch.randn(1, 7, 16, device=device, dtype=torch.float32)
        periodic = (True,)
        y_wrap = mixed_fftconv1d_fp32_bhl_w_reshape(x_blh, k_blh, periodic)
        assert y_wrap.shape == (2, 64, 16)
        x_bhl = rearrange(x_blh, "b l h -> b h l")
        k_bhl = rearrange(k_blh, "b k h -> b h k")
        y_direct = mixed_fftconv1d_fp32_bhl(x_bhl, k_bhl, periodic)
        y_direct_blh = rearrange(y_direct, "b h l -> b l h")
        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    def test_chunked_matches_non_chunked(self, device: str) -> None:
        # Chunking just slices channels and concatenates results — the FFT runs
        # on identical numerical data per chunk, so output is bit-identical.
        torch.manual_seed(42)
        x = torch.randn(2, 96, 64, device=device, dtype=torch.float32)
        k = torch.randn(1, 96, 7, device=device, dtype=torch.float32)
        for periodic in [(False,), (True,)]:
            y_std = mixed_fftconv1d_fp32_bhl(x, k, periodic)
            y_chunked = mixed_fftconv1d_fp32_bhl_chunked(x, k, periodic, chunk_size=32)
            torch.testing.assert_close(y_chunked, y_std, rtol=0, atol=0)

    def test_backward_vs_reference(self, device: str) -> None:
        torch.manual_seed(42)
        B, H, L, K = 2, 8, 64, 15
        periodic = (True,)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)

        y = mixed_fftconv1d_fp32_bhl(x, kernel, periodic)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_mixed_conv_nd(x_ref, k_ref, periodic)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


###############################################################################
# 2D
###############################################################################


class TestMixedFFTConv2D:
    """Tests for 2D mixed-BC FFT convolution (fp32)."""

    @pytest.mark.parametrize("periodic", _PERIODIC_COMBOS_2D)
    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky",
        [
            (2, 8, 32, 32, 5, 5),
            (1, 16, 64, 48, 7, 9),
        ],
    )
    def test_vs_spatial_reference_small_k(
        self,
        device: str,
        B: int,
        H: int,
        X: int,
        Y: int,
        Kx: int,
        Ky: int,
        periodic: tuple[bool, bool],
    ) -> None:
        # Small kernel relative to input — tight tolerance.
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_mixed_conv_nd(x, kernel, periodic)
        y_fft = mixed_fftconv2d_fp32_bhl(x, kernel, periodic)

        assert y_fft.shape == y_ref.shape == (B, H, X, Y)
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL_SMALL_K, atol=ATOL_SMALL_K)

    @pytest.mark.parametrize("periodic", _PERIODIC_COMBOS_2D)
    def test_vs_spatial_reference_full_k(self, device: str, periodic: tuple[bool, bool]) -> None:
        # Kernel covers the entire input on both axes — looser tolerance
        # (output magnitudes ~100, fp32 FFT noise ~|y| * log(F^2) * eps ~ 2e-4).
        torch.manual_seed(42)
        B, H, X, Y = 2, 8, 32, 32
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, X, Y, device=device, dtype=torch.float32)

        y_ref = _ref_mixed_conv_nd(x, kernel, periodic)
        y_fft = mixed_fftconv2d_fp32_bhl(x, kernel, periodic)

        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL_FULL_K, atol=ATOL_FULL_K)

    def test_matches_existing_linear_when_all_false(self, device: str) -> None:
        # Dispatches literally to fftconv2d_fp32_bhl → bit-identical.
        torch.manual_seed(0)
        x = torch.randn(2, 8, 32, 32, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 5, 5, device=device, dtype=torch.float32)
        y_mixed = mixed_fftconv2d_fp32_bhl(x, k, (False, False))
        y_linear = fftconv2d_fp32_bhl(x, k)
        torch.testing.assert_close(y_mixed, y_linear, rtol=0, atol=0)

    def test_matches_existing_circular_when_all_true(self, device: str) -> None:
        # Dispatches literally to circular_fftconv2d_fp32_bhl → bit-identical.
        torch.manual_seed(0)
        x = torch.randn(2, 8, 32, 32, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 5, 5, device=device, dtype=torch.float32)
        y_mixed = mixed_fftconv2d_fp32_bhl(x, k, (True, True))
        y_circ = circular_fftconv2d_fp32_bhl(x, k)
        torch.testing.assert_close(y_mixed, y_circ, rtol=0, atol=0)

    @pytest.mark.parametrize("periodic", _PERIODIC_COMBOS_2D)
    def test_use_phase_shift_vs_roll(self, device: str, periodic: tuple[bool, bool]) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32, 32, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 7, 7, device=device, dtype=torch.float32)
        y_phase = mixed_fftconv2d_fp32_bhl(x, k, periodic, use_phase_shift=True)
        y_roll = mixed_fftconv2d_fp32_bhl(x, k, periodic, use_phase_shift=False)
        torch.testing.assert_close(y_phase, y_roll, rtol=1e-5, atol=1e-5)

    def test_with_shortcut(self, device: str) -> None:
        # Shortcut is deterministic per-channel scaled add → bit-identical.
        torch.manual_seed(42)
        x = torch.randn(2, 8, 32, 32, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 5, 5, device=device, dtype=torch.float32)
        shortcut = torch.randn(8, device=device, dtype=torch.float32)
        periodic = (True, False)
        y_no = mixed_fftconv2d_fp32_bhl(x, k, periodic, shortcut=None)
        y_sc = mixed_fftconv2d_fp32_bhl(x, k, periodic, shortcut=shortcut)
        expected = y_no + rearrange(shortcut, "h -> 1 h 1 1") * x
        torch.testing.assert_close(y_sc, expected, rtol=0, atol=0)

    def test_w_reshape_layout(self, device: str) -> None:
        torch.manual_seed(42)
        x_blh = torch.randn(2, 32, 32, 8, device=device, dtype=torch.float32)
        k_blh = torch.randn(1, 5, 5, 8, device=device, dtype=torch.float32)
        periodic = (True, False)
        y_wrap = mixed_fftconv2d_fp32_bhl_w_reshape(x_blh, k_blh, periodic)
        assert y_wrap.shape == (2, 32, 32, 8)
        x_bhl = rearrange(x_blh, "b x y h -> b h x y")
        k_bhl = rearrange(k_blh, "b kx ky h -> b h kx ky")
        y_direct = mixed_fftconv2d_fp32_bhl(x_bhl, k_bhl, periodic)
        y_direct_blh = rearrange(y_direct, "b h x y -> b x y h")
        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("periodic", _PERIODIC_COMBOS_2D)
    def test_chunked_matches_non_chunked(self, device: str, periodic: tuple[bool, bool]) -> None:
        # Chunking is deterministic per-channel slicing → bit-identical to non-chunked.
        torch.manual_seed(42)
        x = torch.randn(2, 96, 16, 16, device=device, dtype=torch.float32)
        k = torch.randn(1, 96, 5, 5, device=device, dtype=torch.float32)
        y_std = mixed_fftconv2d_fp32_bhl(x, k, periodic)
        y_chunked = mixed_fftconv2d_fp32_bhl_chunked(x, k, periodic, chunk_size=32)
        torch.testing.assert_close(y_chunked, y_std, rtol=0, atol=0)

    @pytest.mark.parametrize("periodic", [(True, False), (False, True), (True, True)])
    def test_backward_vs_reference(self, device: str, periodic: tuple[bool, bool]) -> None:
        torch.manual_seed(42)
        B, H, X, Y, Kx, Ky = 2, 4, 16, 16, 5, 5
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)

        y = mixed_fftconv2d_fp32_bhl(x, kernel, periodic)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_mixed_conv_nd(x_ref, k_ref, periodic)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)

    def test_batched_kernel(self, device: str) -> None:
        torch.manual_seed(0)
        B, H, X, Y = 2, 8, 16, 16
        Kx, Ky = 5, 5
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32)
        y = mixed_fftconv2d_fp32_bhl(x, kernel, (True, False))
        assert y.shape == (B, H, X, Y)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_preserves_caller_dtype(self, device: str, dtype: torch.dtype) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 8, 16, 16, device=device, dtype=dtype)
        k = torch.randn(1, 8, 5, 5, device=device, dtype=dtype)
        shortcut = torch.randn(8, device=device, dtype=dtype)
        y = mixed_fftconv2d_fp32_bhl(x, k, (True, False), shortcut=shortcut)
        assert y.dtype == dtype


###############################################################################
# 3D
###############################################################################


class TestMixedFFTConv3D:
    """Tests for 3D mixed-BC FFT convolution (fp32)."""

    @pytest.mark.parametrize("periodic", _PERIODIC_COMBOS_3D)
    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz",
        [
            (2, 4, 12, 12, 12, 5, 5, 5),
            (1, 8, 16, 16, 8, 7, 7, 5),
        ],
    )
    def test_vs_spatial_reference_small_k(
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
        periodic: tuple[bool, bool, bool],
    ) -> None:
        # Small kernel relative to input — tight tolerance, covers every
        # per-axis BC combination in 3D.
        torch.manual_seed(42)
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32)

        y_ref = _ref_mixed_conv_nd(x, kernel, periodic)
        y_fft = mixed_fftconv3d_fp32_bhl(x, kernel, periodic)

        assert y_fft.shape == y_ref.shape == (B, H, X, Y, Z)
        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL_SMALL_K, atol=ATOL_SMALL_K)

    @pytest.mark.parametrize("periodic", _PERIODIC_COMBOS_3D)
    def test_vs_spatial_reference_full_k(self, device: str, periodic: tuple[bool, bool, bool]) -> None:
        # Kernel covers the entire input on all three axes — looser tolerance.
        torch.manual_seed(42)
        B, H, N = 1, 4, 8
        x = torch.randn(B, H, N, N, N, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, N, N, N, device=device, dtype=torch.float32)

        y_ref = _ref_mixed_conv_nd(x, kernel, periodic)
        y_fft = mixed_fftconv3d_fp32_bhl(x, kernel, periodic)

        torch.testing.assert_close(y_fft, y_ref, rtol=RTOL_FULL_K, atol=ATOL_FULL_K)

    def test_matches_existing_linear_when_all_false(self, device: str) -> None:
        # Dispatches literally to fftconv3d_fp32_bhl → bit-identical.
        torch.manual_seed(0)
        x = torch.randn(2, 4, 12, 12, 12, device=device, dtype=torch.float32)
        k = torch.randn(1, 4, 5, 5, 5, device=device, dtype=torch.float32)
        y_mixed = mixed_fftconv3d_fp32_bhl(x, k, (False, False, False))
        y_linear = fftconv3d_fp32_bhl(x, k)
        torch.testing.assert_close(y_mixed, y_linear, rtol=0, atol=0)

    def test_matches_existing_circular_when_all_true(self, device: str) -> None:
        # Dispatches literally to circular_fftconv3d_fp32_bhl → bit-identical.
        torch.manual_seed(0)
        x = torch.randn(2, 4, 12, 12, 12, device=device, dtype=torch.float32)
        k = torch.randn(1, 4, 5, 5, 5, device=device, dtype=torch.float32)
        y_mixed = mixed_fftconv3d_fp32_bhl(x, k, (True, True, True))
        y_circ = circular_fftconv3d_fp32_bhl(x, k)
        torch.testing.assert_close(y_mixed, y_circ, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "periodic",
        [(False, False, False), (True, True, False), (True, False, True), (False, True, True), (True, True, True)],
    )
    def test_use_phase_shift_vs_roll(self, device: str, periodic: tuple[bool, bool, bool]) -> None:
        # Spatial shape (16, 16, 16) matches the existing
        # ``test_circular_fftconv.py::TestCircularFFTConv3D::test_phase_shift_vs_roll``
        # test, which passes at the same atol=1e-5 tolerance with the same K.
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, 16, 16, device=device, dtype=torch.float32)
        k = torch.randn(1, 4, 5, 5, 5, device=device, dtype=torch.float32)
        y_phase = mixed_fftconv3d_fp32_bhl(x, k, periodic, use_phase_shift=True)
        y_roll = mixed_fftconv3d_fp32_bhl(x, k, periodic, use_phase_shift=False)
        torch.testing.assert_close(y_phase, y_roll, rtol=1e-5, atol=1e-5)

    def test_w_reshape_layout(self, device: str) -> None:
        torch.manual_seed(42)
        x_blh = torch.randn(2, 12, 12, 12, 4, device=device, dtype=torch.float32)
        k_blh = torch.randn(1, 5, 5, 5, 4, device=device, dtype=torch.float32)
        periodic = (True, True, False)
        y_wrap = mixed_fftconv3d_fp32_bhl_w_reshape(x_blh, k_blh, periodic)
        assert y_wrap.shape == (2, 12, 12, 12, 4)
        x_bhl = rearrange(x_blh, "b x y z h -> b h x y z")
        k_bhl = rearrange(k_blh, "b kx ky kz h -> b h kx ky kz")
        y_direct = mixed_fftconv3d_fp32_bhl(x_bhl, k_bhl, periodic)
        y_direct_blh = rearrange(y_direct, "b h x y z -> b x y z h")
        torch.testing.assert_close(y_wrap, y_direct_blh, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "periodic",
        [(False, False, False), (True, False, False), (True, True, False), (True, True, True)],
    )
    def test_chunked_matches_non_chunked(self, device: str, periodic: tuple[bool, bool, bool]) -> None:
        # Chunking is deterministic per-channel slicing → bit-identical to non-chunked.
        torch.manual_seed(42)
        x = torch.randn(2, 96, 8, 8, 8, device=device, dtype=torch.float32)
        k = torch.randn(1, 96, 5, 5, 5, device=device, dtype=torch.float32)
        y_std = mixed_fftconv3d_fp32_bhl(x, k, periodic)
        y_chunked = mixed_fftconv3d_fp32_bhl_chunked(x, k, periodic, chunk_size=32)
        torch.testing.assert_close(y_chunked, y_std, rtol=0, atol=0)

    @pytest.mark.parametrize("periodic", [(True, False, True)])
    def test_backward_vs_reference(self, device: str, periodic: tuple[bool, bool, bool]) -> None:
        torch.manual_seed(42)
        B, H, X, Y, Z, Kx, Ky, Kz = 2, 4, 8, 8, 8, 5, 5, 5
        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32, requires_grad=True)

        y = mixed_fftconv3d_fp32_bhl(x, kernel, periodic)
        grad_output = torch.randn_like(y)
        y.backward(grad_output)
        x_grad = x.grad.clone()
        k_grad = kernel.grad.clone()

        x_ref = x.detach().clone().requires_grad_(True)
        k_ref = kernel.detach().clone().requires_grad_(True)
        y_ref = _ref_mixed_conv_nd(x_ref, k_ref, periodic)
        y_ref.backward(grad_output)

        torch.testing.assert_close(x_grad, x_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)
        torch.testing.assert_close(k_grad, k_ref.grad, rtol=RTOL_BWD, atol=ATOL_BWD)


###############################################################################
# Analytical-truth tests (impulse response & DC response)
#
# These compare against a *closed-form* answer (machine-precision), so they
# cannot be silenced by a loose tolerance. They are the strongest guard
# against alignment bugs (off-by-one in shifts, wrong crop window) and
# normalisation bugs (missing/extra factor in the FFT).
###############################################################################


class TestMixedFFTConvAnalytical:
    """Closed-form sanity checks at machine precision.

    Two analytical settings are used:

    1. **Impulse response.** With input :math:`x[i] = \\delta_{i,0}` (a single
       1 at the origin, zeros elsewhere), depthwise convolution with a flipped
       kernel produces the kernel itself, placed at the position determined by
       the BC and alignment convention:

       - **Periodic axis** (alignment shift :math:`s_d = -\\lfloor (K_d-1)/2\\rfloor`):
         output at position :math:`n` is ``k[(n + s_d + K_d - 1) mod N]``
         (after the cross-correlation-to-convolution flip).
       - **Non-periodic axis** ("same"-padded zero-fill linear conv):
         output at position :math:`n` is ``k[n + s_d + K_d - 1]`` for
         positions where the index lies in ``[0, K_d)``, else 0.

       Rather than re-deriving the index map by hand, we obtain the
       analytical answer by running the **already-verified spatial reference**
       (``_ref_mixed_conv_nd``) on the same impulse input. The spatial
       reference is direct ``F.pad`` + ``F.conv*d`` with no FFT — independent
       of every line of code in ``mixed_fftconv.py``. The comparison is then
       between two algorithmically independent code paths on a *non-random*
       deterministic input.

    2. **DC response.** With constant input ``x[i] = 1`` and a periodic axis,
       the convolution output is exactly ``sum(kernel)`` at every position
       — independent of any alignment or normalisation choice. This is the
       cleanest possible bug-catcher for FFT normalisation errors (e.g. a
       missing ``1/N`` factor in the inverse transform).
    """

    @pytest.mark.parametrize("periodic", _PERIODIC_COMBOS_2D)
    def test_impulse_response_2d(self, device: str, periodic: tuple[bool, bool]) -> None:
        # Even kernel sizes (e.g. K=4) trigger the asymmetric "same" padding
        # convention — important to cover here.
        torch.manual_seed(0)
        B, H, X, Y, Kx, Ky = 1, 2, 16, 16, 5, 4
        x = torch.zeros(B, H, X, Y, device=device, dtype=torch.float32)
        x[..., 0, 0] = 1.0
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y_ref = _ref_mixed_conv_nd(x, kernel, periodic)
        y_fft = mixed_fftconv2d_fp32_bhl(x, kernel, periodic)

        # Spatial reference on an impulse is a few additions of zero plus the
        # kernel — should match the FFT op to machine precision modulo cuFFT
        # ULP noise. Two orders of magnitude tighter than the random-input tol.
        torch.testing.assert_close(y_fft, y_ref, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("periodic", _PERIODIC_COMBOS_3D)
    def test_impulse_response_3d(self, device: str, periodic: tuple[bool, bool, bool]) -> None:
        torch.manual_seed(0)
        B, H, X, Y, Z, Kx, Ky, Kz = 1, 2, 8, 8, 8, 3, 5, 4
        x = torch.zeros(B, H, X, Y, Z, device=device, dtype=torch.float32)
        x[..., 0, 0, 0] = 1.0
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32)

        y_ref = _ref_mixed_conv_nd(x, kernel, periodic)
        y_fft = mixed_fftconv3d_fp32_bhl(x, kernel, periodic)
        torch.testing.assert_close(y_fft, y_ref, rtol=1e-6, atol=1e-6)

    def test_dc_response_all_periodic_2d(self, device: str) -> None:
        """With constant input and all-periodic BC, output = sum(kernel) everywhere.

        This is a closed-form check that does NOT depend on any reference
        helper — purely on PyTorch reduction primitives — so it cannot share
        a bug with ``_ref_mixed_conv_nd``. Catches FFT normalisation bugs
        (missing or extra ``1/F_d`` factors) which would scale the constant
        output by an incorrect global factor.
        """
        torch.manual_seed(0)
        B, H, X, Y, Kx, Ky = 2, 4, 16, 16, 5, 5
        x = torch.ones(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32)

        y = mixed_fftconv2d_fp32_bhl(x, kernel, (True, True))
        expected_per_channel = kernel.squeeze(0).sum(dim=(-1, -2))  # [H]
        expected = expected_per_channel.view(1, H, 1, 1).expand_as(y)
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)

    def test_dc_response_all_periodic_3d(self, device: str) -> None:
        """3D DC response: constant input → sum(kernel) everywhere."""
        torch.manual_seed(0)
        B, H, X, Y, Z, Kx, Ky, Kz = 1, 2, 8, 8, 8, 5, 5, 5
        x = torch.ones(B, H, X, Y, Z, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32)

        y = mixed_fftconv3d_fp32_bhl(x, kernel, (True, True, True))
        expected_per_channel = kernel.squeeze(0).sum(dim=(-1, -2, -3))  # [H]
        expected = expected_per_channel.view(1, H, 1, 1, 1).expand_as(y)
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


###############################################################################
# Validation / argument-handling tests
###############################################################################


class TestMixedFFTConvValidation:
    """Tests for input validation and error handling."""

    def test_wrong_periodic_length_2d(self, device: str) -> None:
        x = torch.randn(2, 8, 16, 16, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 5, 5, device=device, dtype=torch.float32)
        with pytest.raises(AssertionError, match=r"periodic must have length 2"):
            mixed_fftconv2d_fp32_bhl(x, k, (True, False, True))

    def test_wrong_periodic_length_3d(self, device: str) -> None:
        x = torch.randn(2, 4, 8, 8, 8, device=device, dtype=torch.float32)
        k = torch.randn(1, 4, 3, 3, 3, device=device, dtype=torch.float32)
        with pytest.raises(AssertionError, match=r"periodic must have length 3"):
            mixed_fftconv3d_fp32_bhl(x, k, (True, False))

    def test_kernel_larger_than_input_on_periodic_axis_raises(self, device: str) -> None:
        # Periodic axis: K must be <= N (no padding headroom).
        x = torch.randn(2, 8, 16, 16, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 17, 5, device=device, dtype=torch.float32)
        with pytest.raises(AssertionError, match=r"K must be <= N on periodic axis"):
            mixed_fftconv2d_fp32_bhl(x, k, (True, False))

    def test_kernel_too_large_on_non_periodic_axis_raises(self, device: str) -> None:
        # Non-periodic axis: K can grow up to 2*N (the "double-grid" maximum
        # accepted by the legacy linear FFT conv). Larger than that fails.
        x = torch.randn(2, 8, 16, 16, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, 5, 33, device=device, dtype=torch.float32)
        with pytest.raises(AssertionError, match=r"K must be <= 2\*N on non-periodic axis"):
            mixed_fftconv2d_fp32_bhl(x, k, (False, False))

    def test_kernel_at_double_grid_size_on_non_periodic_axis_works(self, device: str) -> None:
        # K = 2N - 1 (the standard "double-grid" SIREN kernel size) must work
        # on a non-periodic axis: this is the regime CKConvND uses when the
        # tuple ``fft_padding`` puts a non-periodic axis next to a periodic one.
        torch.manual_seed(0)
        N = 16
        x = torch.randn(2, 8, N, N, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, N, 2 * N - 1, device=device, dtype=torch.float32)
        y = mixed_fftconv2d_fp32_bhl(x, k, (True, False))
        assert y.shape == x.shape

    def test_mismatched_shortcut_dtype_raises(self, device: str) -> None:
        x = torch.randn(2, 8, 16, 16, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, 8, 5, 5, device=device, dtype=torch.bfloat16)
        bad_shortcut = torch.randn(8, device=device, dtype=torch.float32)
        with pytest.raises(AssertionError, match=r"shortcut\.dtype"):
            mixed_fftconv2d_fp32_bhl(x, k, (True, False), shortcut=bad_shortcut)
