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

"""Tests for the 1D non-causal ("same") subq_ops wrappers in fftconv_custom.

Validates :func:`nvsubquadratic.ops.fftconv_custom.fftconv1d_*` against the
torch.fft reference :func:`nvsubquadratic.ops.fftconv.fftconv1d_fp32_bhl` on both
engines the wrapper dispatches to: ``fft_conv1d`` (fp32, or lengths the Toeplitz
kernel does not cover) and ``explicit_conv1d`` (bf16/fp16 at supported lengths
on compute capability 10.x).

Usage (requires GPU):
    PYTHONPATH=. python -m pytest tests/ops/test_fftconv_custom_1d_noncausal.py -v -o addopts=""
"""

import importlib.util

import pytest
import torch

from nvsubquadratic.ops.fftconv import fftconv1d_fp32_bhl, fftconv1d_fp32_bhl_w_reshape


pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("subquadratic_ops_torch") is None, reason="subquadratic_ops_torch not installed"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

TOL = {torch.float32: 1e-4, torch.bfloat16: 1e-2, torch.float16: 1e-2}
# (B, H, L, K): L=256 is an explicit_conv1d length, 100 and 37 are fft_conv1d-only.
SHAPES = [(2, 32, 256, 511), (2, 32, 100, 199), (1, 16, 37, 73), (2, 32, 256, 129)]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def make(B, H, L, K, dtype, device="cuda"):
    torch.manual_seed(0)
    x = torch.randn(B, H, L, device=device, dtype=dtype)
    kernel = torch.randn(1, H, K, device=device, dtype=dtype) / K**0.5
    shortcut = torch.randn(H, device=device, dtype=dtype)
    return x, kernel, shortcut


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B, H, L, K", SHAPES)
def test_forward_matches_torch_fft(B, H, L, K, dtype) -> None:
    from nvsubquadratic.ops.fftconv_custom import fftconv1d_bhl

    x, kernel, shortcut = make(B, H, L, K, dtype)
    got = fftconv1d_bhl(x, kernel, shortcut)
    assert got.dtype == dtype and got.shape == x.shape
    assert rel(got, fftconv1d_fp32_bhl(x, kernel, shortcut)) < TOL[dtype]


def test_uses_explicit_conv1d_where_it_applies() -> None:
    from nvsubquadratic.ops import fftconv_custom

    fftconv_custom.load_conv1d_kernels()
    explicit = fftconv_custom._get_explicit_conv1d()
    if explicit is None:
        pytest.skip("installed wheel has no explicit_conv1d")
    index = torch.cuda.current_device()
    eligible = torch.cuda.get_device_capability(index)[0] == fftconv_custom._EXPLICIT_CONV1D_CAPABILITY_MAJOR
    applies = fftconv_custom._explicit_conv1d_applies.__wrapped__
    assert applies(index, torch.bfloat16, 256, 511) is eligible
    assert applies(index, torch.float32, 256, 511) is False
    assert applies(index, torch.bfloat16, 100, 199) is False
    assert applies(index, torch.bfloat16, 256, 2 * 256 + 1) is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B, H, L, K", SHAPES[:2])
def test_backward_matches_torch_fft(B, H, L, K, dtype) -> None:
    from nvsubquadratic.ops.fftconv_custom import fftconv1d_bhl

    x, kernel, shortcut = make(B, H, L, K, dtype)
    grads = []
    for fn in (fftconv1d_bhl, fftconv1d_fp32_bhl):
        xi, ki, si = (t.clone().requires_grad_(True) for t in (x, kernel, shortcut))
        fn(xi, ki, si).square().sum().backward()
        grads.append((xi.grad, ki.grad, si.grad))
    for got, want in zip(*grads):
        assert rel(got, want) < 3 * TOL[dtype]


@pytest.mark.parametrize("B, H, L, K", SHAPES[:2])
def test_blh_layout_and_chunked_variants(B, H, L, K) -> None:
    from nvsubquadratic.ops.fftconv_custom import (
        fftconv1d_bhl,
        fftconv1d_bhl_chunked,
        fftconv1d_bhl_w_reshape,
        fftconv1d_bhl_w_reshape_chunked,
    )

    x, kernel, shortcut = make(B, H, L, K, torch.float32)
    want = fftconv1d_bhl(x, kernel, shortcut)
    assert rel(fftconv1d_bhl_chunked(x, kernel, shortcut, chunk_size=8), want) < 1e-6
    assert rel(fftconv1d_bhl_chunked(x, kernel[0], shortcut, chunk_size=8), want) < 1e-6
    x_blh, k_blh = x.transpose(1, 2), kernel.transpose(1, 2)
    assert (
        rel(fftconv1d_bhl_w_reshape(x_blh, k_blh, shortcut), fftconv1d_fp32_bhl_w_reshape(x_blh, k_blh, shortcut))
        < 1e-4
    )
    assert rel(fftconv1d_bhl_w_reshape_chunked(x_blh, k_blh, shortcut, chunk_size=8), want.transpose(1, 2)) < 1e-6


def test_per_sample_kernel_raises() -> None:
    from nvsubquadratic.ops.fftconv_custom import fftconv1d_bhl

    x, kernel, _ = make(2, 16, 64, 127, torch.float32)
    with pytest.raises(NotImplementedError, match="per-sample"):
        fftconv1d_bhl(x, kernel.expand(2, -1, -1))


def test_ckconv_1d_noncausal_subq_ops_matches_torch_fft() -> None:
    from nvsubquadratic.lazy_config import LazyConfig
    from nvsubquadratic.modules.ckconv_nd import CKConvND
    from nvsubquadratic.modules.kernels_nd import SIRENKernelND

    def build(backend):
        torch.manual_seed(0)
        return CKConvND(
            data_dim=1,
            hidden_dim=16,
            kernel_cfg=LazyConfig(SIRENKernelND)(
                out_dim=16,
                data_dim=1,
                mlp_hidden_dim=16,
                num_layers=2,
                embedding_dim=16,
                omega_0=10.0,
                L_cache=64,
                use_bias=True,
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="double",
            fft_padding="zero",
            fft_backend=backend,
        ).cuda()

    x = torch.randn(2, 64, 16, device="cuda")
    assert rel(build("subq_ops")(x), build("torch_fft")(x)) < 1e-4
