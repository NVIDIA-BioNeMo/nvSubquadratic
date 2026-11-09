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

"""Channel-last ``causal_conv1d`` wrapper: eager parity with ``F.conv1d`` and correctness under torch.compile.

The compiled case feeds the kernel a transposed intermediate, the layout that
trips the upstream primitive's stride check without the wrapper; ``torch.library.opcheck``
pins the fake kernels of both NWH ops.

Usage (requires GPU):
    PYTHONPATH=. python -m pytest tests/ops/test_causal_conv1d_nwh_compile.py -v -o addopts=""
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from nvsubquadratic.ops.causal_conv1d_custom import causal_conv1d


pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("subquadratic_ops_torch") is None, reason="subquadratic_ops_torch not installed"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


def reference(x, weight, bias):
    """Causal cross-correlation on channel-last ``x`` ``[N, L, C]`` with ``weight`` ``[K, C]``."""
    taps, channels = weight.shape
    y = F.conv1d(F.pad(x.transpose(1, 2), (taps - 1, 0)), weight.t().unsqueeze(1), bias, groups=channels)
    return y.transpose(1, 2)


def test_eager_matches_conv1d() -> None:
    """Channel-last ``causal_conv1d`` matches ``F.conv1d`` in output and gradients."""
    torch.manual_seed(0)
    x = torch.randn(4, 32, 16, device="cuda", requires_grad=True)
    weight = torch.randn(3, 16, device="cuda", requires_grad=True)
    bias = torch.randn(16, device="cuda", requires_grad=True)
    got = causal_conv1d(x, weight, bias, "identity", channel_last=True)
    want = reference(x, weight, bias)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    grads = torch.autograd.grad(got.square().sum(), (x, weight, bias))
    want_grads = torch.autograd.grad(want.square().sum(), (x, weight, bias))
    for g, w in zip(grads, want_grads):
        torch.testing.assert_close(g, w, atol=1e-3, rtol=1e-4)


def test_bias_none_is_zero_bias() -> None:
    """``bias=None`` gives the same output as a zero bias."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, device="cuda")
    weight = torch.randn(3, 16, device="cuda")
    assert torch.equal(
        causal_conv1d(x, weight, None, "identity", channel_last=True),
        causal_conv1d(x, weight, torch.zeros(16, device="cuda"), "identity", channel_last=True),
    )


def test_bias_none_backward_after_inference_mode() -> None:
    """``bias=None`` still trains after an ``inference_mode`` call with the same channel count."""
    weight = torch.randn(3, 40, device="cuda", requires_grad=True)
    with torch.inference_mode():
        causal_conv1d(torch.randn(2, 8, 40, device="cuda"), weight.detach(), None, "identity", channel_last=True)
    x = torch.randn(2, 8, 40, device="cuda", requires_grad=True)
    causal_conv1d(x, weight, None, "identity", channel_last=True).sum().backward()
    assert x.grad is not None and weight.grad is not None


@pytest.mark.parametrize(("dtype", "grad_rtol"), [(torch.float32, 1e-4), (torch.bfloat16, 1e-2)])
@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
def test_compiled_transposed_input_matches_eager(mode, dtype, grad_rtol) -> None:
    """Compiled conv on a transposed pair stream matches eager in output and gradients."""
    torch.manual_seed(0)
    weight = torch.randn(3, 16, device="cuda", dtype=dtype, requires_grad=True)

    def fn(pair):
        # [B, L, L, C] -> transposed rows -> [B*L, L, C], fed as a dense channel-major view.
        stream = pair.transpose(1, 2).reshape(-1, pair.shape[1], pair.shape[-1])
        stream = stream.transpose(1, 2).contiguous().transpose(1, 2)
        # Consumed in-graph so inductor asserts the output stride the fake kernel promises.
        return causal_conv1d(stream, weight, None, "identity", channel_last=True).square()

    pair = torch.randn(2, 24, 24, 16, device="cuda", dtype=dtype, requires_grad=True)
    want = fn(pair)
    want_grads = torch.autograd.grad(want.square().sum(), (pair, weight))
    compiled = torch.compile(fn, mode=mode)
    for _ in range(3):
        got = compiled(pair)
        grads = torch.autograd.grad(got.square().sum(), (pair, weight))
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    for g, w in zip(grads, want_grads):
        torch.testing.assert_close(g, w, atol=1e-3, rtol=grad_rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_opcheck(dtype) -> None:
    """NWH forward and backward custom ops pass ``torch.library.opcheck`` on a dense transposed input."""
    torch.manual_seed(0)
    x = torch.randn(4, 16, 32, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    weight = torch.randn(3, 16, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(16, device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn(4, 32, 16, device="cuda", dtype=dtype)
    torch.library.opcheck(torch.ops.nvsubquadratic.causal_conv1d_nwh, (x, weight, bias, "identity"))
    torch.library.opcheck(
        torch.ops.nvsubquadratic.causal_conv1d_nwh_backward,
        (grad_out, x.detach(), weight.detach(), bias.detach(), "identity"),
    )
