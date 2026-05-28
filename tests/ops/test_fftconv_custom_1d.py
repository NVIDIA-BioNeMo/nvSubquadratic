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

# TODO: Add license header here


"""Tests for the 1D causal subq_ops wrappers in fftconv_custom.

Validates :func:`nvsubquadratic.ops.fftconv_custom.causal_fftconv1d_*` against
the torch.fft reference :func:`nvsubquadratic.ops.fftconv.causal_fftconv1d_fp32_bhl`.

Covers:
  - Forward correctness: shared kernel, BHL and BLH layouts
  - Chunked vs non-chunked consistency
  - Backward correctness: gradient comparison for x and kernel
  - Shortcut semantics
  - Dtype handling: bf16, fp16, fp32 inputs
  - Negative case: per-sample FiLM weights raise NotImplementedError

Usage (requires GPU — run inside SLURM):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/ops/test_fftconv_custom_1d.py -v -o addopts=""
"""

import pytest
import torch


ATOL_F32 = 1e-3
RTOL_F32 = 1e-4
ATOL_GRAD = 1e-2
RTOL_GRAD = 1e-3


def _has_subq_ops_1d() -> bool:
    try:
        from subquadratic_ops_torch.fft_causal_conv1d import fft_causal_conv1d  # noqa: F401

        return True
    except ImportError:
        return False


requires_subq_ops_1d = pytest.mark.skipif(
    not _has_subq_ops_1d(),
    reason="subquadratic_ops_torch.fft_causal_conv1d not available",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

pytestmark = [requires_subq_ops_1d, requires_cuda]


@pytest.fixture
def device() -> str:
    return "cuda"


def _ref_bhl(x, kernel, shortcut=None):
    from nvsubquadratic.ops.fftconv import causal_fftconv1d_fp32_bhl

    return causal_fftconv1d_fp32_bhl(x, kernel, shortcut)


def _ref_blh(x, kernel, shortcut=None):
    from nvsubquadratic.ops.fftconv import causal_fftconv1d_fp32_bhl_w_reshape

    return causal_fftconv1d_fp32_bhl_w_reshape(x, kernel, shortcut)


def _custom_bhl(x, kernel, shortcut=None):
    from nvsubquadratic.ops.fftconv_custom import causal_fftconv1d_bhl

    return causal_fftconv1d_bhl(x, kernel, shortcut)


def _custom_blh(x, kernel, shortcut=None):
    from nvsubquadratic.ops.fftconv_custom import causal_fftconv1d_bhl_w_reshape

    return causal_fftconv1d_bhl_w_reshape(x, kernel, shortcut)


def _custom_bhl_chunked(x, kernel, shortcut=None, chunk_size=None):
    from nvsubquadratic.ops.fftconv_custom import causal_fftconv1d_bhl_chunked

    return causal_fftconv1d_bhl_chunked(x, kernel, shortcut, chunk_size)


def _custom_blh_chunked(x, kernel, shortcut=None, chunk_size=None):
    from nvsubquadratic.ops.fftconv_custom import causal_fftconv1d_bhl_w_reshape_chunked

    return causal_fftconv1d_bhl_w_reshape_chunked(x, kernel, shortcut, chunk_size)


SHARED_SHAPES = [
    # (B, H, L, K)
    (2, 64, 256, 7),
    (4, 128, 512, 7),
    (1, 256, 1024, 15),
    (2, 32, 128, 3),
    (1, 96, 2048, 65),
    (2, 64, 64, 64),  # K == L edge case
]


class TestForward:
    """Forward correctness vs torch.fft reference."""

    @pytest.mark.parametrize("B, H, L, K", SHARED_SHAPES)
    def test_bhl_shared_kernel(self, device, B, H, L, K):
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        k = torch.randn(1, H, K, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, k)
        y_custom = _custom_bhl(x, k)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    @pytest.mark.parametrize("B, H, L, K", SHARED_SHAPES)
    def test_blh_shared_kernel(self, device, B, H, L, K):
        torch.manual_seed(42)
        x = torch.randn(B, L, H, device=device, dtype=torch.float32)
        k = torch.randn(1, K, H, device=device, dtype=torch.float32)

        y_ref = _ref_blh(x, k)
        y_custom = _custom_blh(x, k)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)

    @pytest.mark.parametrize("B, H, L, K", SHARED_SHAPES[:3])
    def test_shortcut_semantics(self, device, B, H, L, K):
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        k = torch.randn(1, H, K, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, k, shortcut)
        y_custom = _custom_bhl(x, k, shortcut)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)


class TestChunked:
    """Chunked variant matches non-chunked output."""

    @pytest.mark.parametrize("B, H, L, K", SHARED_SHAPES[:4])
    @pytest.mark.parametrize("chunk_size", [16, 32, 128])
    def test_bhl_chunked_matches(self, device, B, H, L, K, chunk_size):
        torch.manual_seed(42)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        k = torch.randn(1, H, K, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_unchunked = _custom_bhl(x, k, shortcut)
        y_chunked = _custom_bhl_chunked(x, k, shortcut, chunk_size)

        torch.testing.assert_close(y_chunked, y_unchunked, atol=ATOL_F32, rtol=RTOL_F32)

    def test_blh_chunked_matches(self, device):
        torch.manual_seed(42)
        B, H, L, K = 2, 64, 256, 7
        x = torch.randn(B, L, H, device=device, dtype=torch.float32)
        k = torch.randn(1, K, H, device=device, dtype=torch.float32)

        y_unchunked = _custom_blh(x, k)
        y_chunked = _custom_blh_chunked(x, k, chunk_size=32)

        torch.testing.assert_close(y_chunked, y_unchunked, atol=ATOL_F32, rtol=RTOL_F32)


class TestBackward:
    """Gradient correctness vs torch.fft reference."""

    @pytest.mark.parametrize("B, H, L, K", [(2, 64, 256, 7), (4, 32, 128, 5)])
    def test_grad_matches_reference(self, device, B, H, L, K):
        torch.manual_seed(42)
        x_ref = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        k_ref = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)
        y_ref = _ref_bhl(x_ref, k_ref)
        y_ref.sum().backward()

        x_s = x_ref.detach().clone().requires_grad_(True)
        k_s = k_ref.detach().clone().requires_grad_(True)
        y_s = _custom_bhl(x_s, k_s)
        y_s.sum().backward()

        torch.testing.assert_close(x_s.grad, x_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)
        torch.testing.assert_close(k_s.grad, k_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)


class TestDtype:
    """Input dtype is preserved on output; non-fp32 inputs are upcast internally."""

    @pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_dtype_preserved(self, device, in_dtype):
        x = torch.randn(2, 32, 128, device=device, dtype=in_dtype)
        k = torch.randn(1, 32, 5, device=device, dtype=in_dtype)
        y = _custom_bhl(x, k)
        assert y.dtype == in_dtype
        assert torch.isfinite(y).all()


class TestLongKernel:
    """Long kernels — where the FFT kernel beats the direct conv (upstream docstring: K>=128)."""

    @pytest.mark.parametrize("K", [128, 512, 2048])
    def test_long_kernel_matches_reference(self, device, K):
        torch.manual_seed(42)
        B, H, L = 1, 32, max(K, 256)
        x = torch.randn(B, H, L, device=device, dtype=torch.float32)
        k = torch.randn(1, H, K, device=device, dtype=torch.float32)

        y_ref = _ref_bhl(x, k)
        y_custom = _custom_bhl(x, k)

        torch.testing.assert_close(y_custom, y_ref, atol=ATOL_F32, rtol=RTOL_F32)


class TestErrors:
    """Negative cases."""

    def test_per_sample_film_weights_rejected(self, device):
        """The upstream 1D kernel does not accept batched weights."""
        x = torch.randn(2, 32, 128, device=device, dtype=torch.float32)
        k = torch.randn(2, 32, 5, device=device, dtype=torch.float32)  # B=2 not B=1
        with pytest.raises(NotImplementedError, match="per-sample FiLM weights"):
            _custom_bhl(x, k)
