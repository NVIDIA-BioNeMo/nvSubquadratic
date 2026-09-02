# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the fused 2D FFT-conv wrappers in ``nvsubquadratic.ops.fftconv_custom``.

These wrap ``subquadratic_ops_torch.fused_fft_conv2d``, which differs from the
older ``fft_conv2d`` path in two ways that this file pins down:

  - It runs **natively** in fp32/fp16/bf16 rather than upcasting to fp32.
  - It crops the 'same' window at ``fft_size // 2`` rather than at ``K // 2``.
    The wrapper compensates with a top/left filter pad; the equivalence tests
    below are what make that compensation load-bearing rather than incidental.
    Dropping the pad shifts the output by ``fft_size // 2 - K // 2`` pixels,
    which shows up here as a ~1.41 (= sqrt(2), i.e. fully decorrelated) error.

Correctness is measured against ``fftconv2d_fp32_bhl`` / ``fftconv2d_fp32_blh``
with a normwise (L2) relative error rather than elementwise ``assert_close``:
FFT roundoff scales with ``||output||``, not with each element, so near-zero
output elements make elementwise relative error meaningless.

Usage (requires GPU — run inside SLURM):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/ops/test_fused_fftconv2d.py -v -o addopts=""
"""

import pytest
import torch

from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl, fftconv2d_fp32_blh
from nvsubquadratic.ops.fftconv_custom import (
    FUSED_FFT_SIZE_128_MIN_ARCH,
    fused_fftconv2d_arch_supported,
    fused_fftconv2d_bhl,
    fused_fftconv2d_bhl_chunked,
    fused_fftconv2d_blh,
    fused_fftconv2d_blh_chunked,
    fused_fftconv2d_max_spatial,
    fused_fftconv2d_supported,
    resolve_fused_fft_size,
)
from tests.conftest import requires_sm90, requires_subq_ops_fused


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

pytestmark = [requires_subq_ops_fused, requires_cuda]

# Normwise relative-error budgets, calibrated against the observed error of the
# fused kernel vs the fp32 torch reference (fp32 ~3e-7, fp16 ~3e-4, bf16 ~3e-3)
# with ~3x headroom. Gradients accumulate over the batch, so they get more room.
L2_TOL = {torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 8e-3}
L2_TOL_GRAD = {torch.float32: 1e-6, torch.float16: 2e-3, torch.bfloat16: 1.5e-2}

HIDDEN_DIM = 16
BATCH = 2


def l2_rel(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Normwise relative error, computed in fp64 so the metric adds no roundoff."""
    pred64, ref64 = pred.double(), ref.double()
    den = ref64.norm()
    return ((pred64 - ref64).norm() / den).item() if den > 0 else (pred64 - ref64).norm().item()


def assert_l2_close(pred, ref, dtype, tol_table=None, name=""):
    tol = (tol_table or L2_TOL)[dtype]
    rel = l2_rel(pred, ref)
    assert rel < tol, f"{name} L2 rel error {rel:.3e} exceeds tol {tol:.1e}"


def _inputs(spatial, kernel_size, dtype, kernel_batch=1, seed=0):
    """Build a BHL input/kernel/shortcut triple on CUDA."""
    torch.manual_seed(seed)
    x = torch.randn(BATCH, HIDDEN_DIM, spatial, spatial, device="cuda", dtype=dtype)
    # Scaled down so the convolution output stays in a sane range for fp16.
    kernel = (torch.randn(kernel_batch, HIDDEN_DIM, kernel_size, kernel_size, device="cuda") * 0.05).to(dtype)
    shortcut = torch.randn(HIDDEN_DIM, device="cuda", dtype=dtype)
    return x, kernel, shortcut


# ---------------------------------------------------------------------------
# fft_size resolution
# ---------------------------------------------------------------------------


class TestResolveFftSize:
    """The tile-size chooser and its guard rails."""

    @pytest.mark.parametrize(
        ("spatial", "expected"),
        [(4, 8), (7, 16), (8, 16), (16, 32), (32, 64), (64, 128)],
    )
    def test_default_for_double_grid_kernel(self, spatial, expected):
        """CKConvND's double-grid kernel (K = 2N-1) resolves to a 2*next_pow2(N) tile."""
        assert resolve_fused_fft_size(spatial, spatial, 2 * spatial - 1, 2 * spatial - 1) == expected

    def test_default_is_smallest_admissible(self):
        """A small kernel does not inflate the tile beyond what the input needs."""
        assert resolve_fused_fft_size(16, 16, 3, 3) == 32

    def test_kernel_can_drive_the_tile_size(self):
        """A kernel wider than 2x the input still gets enough headroom for the pre-pad."""
        # ceil(96 / 2) = 48 > 8, so the kernel, not the input, sets the tile.
        assert resolve_fused_fft_size(8, 8, 96, 96) == 128

    def test_non_square_uses_the_larger_axis(self):
        assert resolve_fused_fft_size(8, 32, 15, 63) == 64

    def test_minimum_tile_is_8(self):
        """Tiny shapes clamp up to the smallest tile the kernel is compiled for."""
        assert resolve_fused_fft_size(1, 1, 1, 1) == 8

    def test_rejects_oversized_spatial(self):
        with pytest.raises(ValueError, match="too large for the fused 2D FFT kernel"):
            resolve_fused_fft_size(65, 65, 129, 129)

    def test_rejects_unsupported_explicit_size(self):
        with pytest.raises(ValueError, match="fft_size must be one of"):
            resolve_fused_fft_size(8, 8, 15, 15, fft_size=48)

    def test_rejects_too_small_explicit_size(self):
        with pytest.raises(ValueError, match="too small for input"):
            resolve_fused_fft_size(32, 32, 63, 63, fft_size=16)

    def test_accepts_larger_explicit_size(self):
        """Over-provisioning the tile is allowed (costs speed, not correctness)."""
        assert resolve_fused_fft_size(8, 8, 15, 15, fft_size=128) == 128

    def test_supported_probe_matches_resolve(self):
        assert fused_fftconv2d_supported(64, 64, 127, 127) is True
        assert fused_fftconv2d_supported(65, 65, 129, 129) is False

    def test_max_spatial(self):
        assert fused_fftconv2d_max_spatial() == 64


# ---------------------------------------------------------------------------
# Forward equivalence with the torch.fft reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
class TestForwardMatchesReference:
    """The fused path reproduces fftconv2d_fp32_bhl up to dtype roundoff."""

    @pytest.mark.parametrize("spatial", [7, 8, 16, 32, pytest.param(64, marks=requires_sm90)])
    def test_double_grid_kernel(self, spatial, dtype):
        """K = 2N-1, the kernel size CKConvND generates on a double grid."""
        x, kernel, shortcut = _inputs(spatial, 2 * spatial - 1, dtype)
        assert_l2_close(fused_fftconv2d_bhl(x, kernel, shortcut), fftconv2d_fp32_bhl(x, kernel, shortcut), dtype)

    @pytest.mark.parametrize("kernel_size", [1, 3, 8, 15, 16, 17, 32])
    def test_kernel_sizes(self, kernel_size, dtype):
        """Both parities and both sides of K == N, since the pre-pad is K//2-dependent."""
        x, kernel, shortcut = _inputs(16, kernel_size, dtype)
        assert_l2_close(fused_fftconv2d_bhl(x, kernel, shortcut), fftconv2d_fp32_bhl(x, kernel, shortcut), dtype)

    def test_film_per_sample_kernel(self, dtype):
        """A per-sample (FiLM) kernel batch is supported on the fused path."""
        x, kernel, shortcut = _inputs(16, 31, dtype, kernel_batch=BATCH)
        assert_l2_close(fused_fftconv2d_bhl(x, kernel, shortcut), fftconv2d_fp32_bhl(x, kernel, shortcut), dtype)

    def test_without_shortcut(self, dtype):
        x, kernel, _ = _inputs(16, 31, dtype)
        assert_l2_close(fused_fftconv2d_bhl(x, kernel), fftconv2d_fp32_bhl(x, kernel), dtype)

    def test_blh_layout(self, dtype):
        """Channels-last entry point, against the channels-last reference."""
        x, kernel, shortcut = _inputs(16, 31, dtype)
        x_blh = x.permute(0, 2, 3, 1).contiguous()
        kernel_blh = kernel.permute(0, 2, 3, 1).contiguous()
        assert_l2_close(
            fused_fftconv2d_blh(x_blh, kernel_blh, shortcut),
            fftconv2d_fp32_blh(x_blh, kernel_blh, shortcut),
            dtype,
        )

    @pytest.mark.parametrize("chunk_size", [4, 8, HIDDEN_DIM, HIDDEN_DIM * 2])
    def test_chunked_matches_unchunked(self, chunk_size, dtype):
        """Channel chunking is a pure memory optimisation, including a ragged last chunk."""
        x, kernel, shortcut = _inputs(16, 31, dtype)
        chunked = fused_fftconv2d_bhl_chunked(x, kernel, shortcut, chunk_size=chunk_size)
        torch.testing.assert_close(chunked, fused_fftconv2d_bhl(x, kernel, shortcut), atol=0, rtol=0)

    def test_chunked_blh(self, dtype):
        x, kernel, shortcut = _inputs(16, 31, dtype)
        x_blh = x.permute(0, 2, 3, 1).contiguous()
        kernel_blh = kernel.permute(0, 2, 3, 1).contiguous()
        assert_l2_close(
            fused_fftconv2d_blh_chunked(x_blh, kernel_blh, shortcut, chunk_size=4),
            fftconv2d_fp32_blh(x_blh, kernel_blh, shortcut),
            dtype,
        )


# ---------------------------------------------------------------------------
# Crop alignment — the wrapper's reason for existing
# ---------------------------------------------------------------------------


class TestCropAlignment:
    """The top/left pre-pad is what aligns the fused crop with every other backend."""

    def test_raw_upstream_call_is_shifted(self):
        """Without the pre-pad the upstream op is offset, not merely less accurate.

        This is the regression guard: if someone "simplifies" the wrapper by
        calling fused_fft_conv2d directly, this is the error they would ship.
        """
        pytest.importorskip("subquadratic_ops_torch")
        from subquadratic_ops_torch.fused_fft_conv2d import fused_fft_conv2d

        spatial, kernel_size = 16, 31
        x, kernel, _ = _inputs(spatial, kernel_size, torch.float32)
        ref = fftconv2d_fp32_bhl(x, kernel)

        unpadded = fused_fft_conv2d(x.contiguous(), kernel.contiguous(), fft_size=2 * spatial)
        # ~sqrt(2): a shift by fft_size//2 - K//2 decorrelates the output entirely.
        assert l2_rel(unpadded, ref) > 1.0

        assert_l2_close(fused_fftconv2d_bhl(x, kernel), ref, torch.float32)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_batch", [1, BATCH])
def test_backward_matches_reference(dtype, kernel_batch):
    """Gradients w.r.t. x, kernel, and shortcut all match the fp32 reference."""
    spatial, kernel_size = 16, 31
    x0, k0, s0 = _inputs(spatial, kernel_size, torch.float32, kernel_batch=kernel_batch)
    grad_out = torch.randn(BATCH, HIDDEN_DIM, spatial, spatial, device="cuda")

    def run(fn, cast_to):
        x = x0.to(cast_to).detach().requires_grad_()
        k = k0.to(cast_to).detach().requires_grad_()
        s = s0.to(cast_to).detach().requires_grad_()
        fn(x, k, s).backward(grad_out.to(cast_to))
        return x.grad, k.grad, s.grad

    ref = run(fftconv2d_fp32_bhl, torch.float32)
    fused = run(fused_fftconv2d_bhl, dtype)

    for got, want, name in zip(fused, ref, ("grad_x", "grad_kernel", "grad_shortcut")):
        assert_l2_close(got, want, dtype, tol_table=L2_TOL_GRAD, name=name)


def test_backward_flows_through_chunked():
    """The chunked path is differentiable and agrees with the unchunked one."""
    x0, k0, s0 = _inputs(16, 31, torch.float32)

    def run(fn, **kw):
        x = x0.detach().requires_grad_()
        k = k0.detach().requires_grad_()
        s = s0.detach().requires_grad_()
        fn(x, k, s, **kw).sum().backward()
        return x.grad, k.grad, s.grad

    for got, want in zip(run(fused_fftconv2d_bhl_chunked, chunk_size=4), run(fused_fftconv2d_bhl)):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Dtype behaviour
# ---------------------------------------------------------------------------


class TestDtype:
    """Native-dtype passthrough — the reason to prefer this over ``fftconv2d_bhl``."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_output_dtype_matches_input(self, dtype):
        x, kernel, shortcut = _inputs(16, 31, dtype)
        assert fused_fftconv2d_bhl(x, kernel, shortcut).dtype == dtype
        assert fused_fftconv2d_bhl_chunked(x, kernel, shortcut).dtype == dtype

    def test_kernel_dtype_is_coerced_to_input(self):
        """A kernel in a different dtype is cast rather than raising from the CUDA op."""
        x, kernel, shortcut = _inputs(16, 31, torch.bfloat16)
        out = fused_fftconv2d_bhl(x, kernel.float(), shortcut)
        assert out.dtype == torch.bfloat16
        assert_l2_close(out, fused_fftconv2d_bhl(x, kernel, shortcut), torch.bfloat16)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_rejects_oversized_input():
    """Exceeding the 64x64 cap raises with a pointer to the other backends."""
    x = torch.randn(1, 4, 65, 65, device="cuda")
    kernel = torch.randn(1, 4, 129, 129, device="cuda")
    with pytest.raises(ValueError, match="too large for the fused 2D FFT kernel"):
        fused_fftconv2d_bhl(x, kernel)


def test_arch_predicate_matches_device_capability():
    """The 128 tile is gated on SM90+; smaller tiles run on any supported GPU."""
    device = torch.device("cuda:0")
    is_sm90plus = torch.cuda.get_device_capability(0) >= FUSED_FFT_SIZE_128_MIN_ARCH

    assert fused_fftconv2d_arch_supported(device, 64) is True
    assert fused_fftconv2d_arch_supported(device, 128) is is_sm90plus


def test_128_tile_raises_below_sm90():
    """A shape that needs the 128 tile fails loudly on SM80/SM86 rather than in the kernel.

    32x32 fits the 64 tile and must work everywhere; 64x64 escalates to the 128
    tile, which needs more shared memory than pre-Hopper parts provide. Both
    branches are asserted so this stays meaningful on either class of runner.
    """
    kernel = torch.randn(1, 4, 5, 5, device="cuda")

    # Always within reach: resolves to the 64 tile.
    small = torch.randn(1, 4, 32, 32, device="cuda")
    assert resolve_fused_fft_size(32, 32, 5, 5) == 64
    fused_fftconv2d_bhl(small, kernel)

    large = torch.randn(1, 4, 64, 64, device="cuda")
    assert resolve_fused_fft_size(64, 64, 5, 5) == 128

    if torch.cuda.get_device_capability(0) >= FUSED_FFT_SIZE_128_MIN_ARCH:
        fused_fftconv2d_bhl(large, kernel)
    else:
        with pytest.raises(RuntimeError, match="requires compute capability"):
            fused_fftconv2d_bhl(large, kernel)
