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
"""Tests for the precision contract of ``CKConvND.apply_convolution``.

By default each FFT backend keeps its own precision for bfloat16/float16
activations, and ``CKConvND`` casts the float32 ``shortcut`` parameter to the
activations' dtype, so every backend accepts them.  With
``allow_reduced_precision=False`` every backend computes the long convolution
and the shortcut term in float32 and the result is rounded once to the input
dtype.  The value tests compare against the backend op that ``CKConvND``
dispatches to, called directly, or against the same module with its shortcut
parameter cast to the activations' dtype.

The CPU tests use ``torch_fft``; the GPU tests cover the ``subq_ops`` backends.

Usage:
    PYTHONPATH=. python -m pytest tests/modules/test_ckconv_nd_precision.py -v -o addopts=""
"""

import copy
import importlib.util

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from tests.conftest import requires_subq_ops_fused


HIDDEN = 8

CPU_CASES = {
    "1d": {"data_dim": 1, "spatial": (16,)},
    "1d-causal": {"data_dim": 1, "spatial": (16,), "is_causal": True},
    "2d": {"data_dim": 2, "spatial": (8, 6)},
    "3d": {"data_dim": 3, "spatial": (4, 4, 4)},
}

_requires_cuda_subq_ops = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("subquadratic_ops_torch") is None,
    reason="needs CUDA and subquadratic_ops_torch",
)

GPU_CASES = {
    "subq_ops-2d": {"fft_backend": "subq_ops", "data_dim": 2, "spatial": (16, 12)},
    "subq_ops-1d-causal": {"fft_backend": "subq_ops", "data_dim": 1, "spatial": (64,), "is_causal": True},
    "subq_ops-1d": {"fft_backend": "subq_ops", "data_dim": 1, "spatial": (64,)},
    "subq_ops_fused-2d": {"fft_backend": "subq_ops_fused", "data_dim": 2, "spatial": (16, 12)},
}
# The default constructor, and the float32 mode.
MODES = pytest.mark.parametrize("mode", [{}, {"allow_reduced_precision": False}], ids=["default", "float32"])


def _gpu_params(names):
    """``pytest.param`` per GPU case, with the fused-kernel version gate where it applies."""
    return [
        pytest.param(
            name, marks=[requires_subq_ops_fused] if GPU_CASES[name]["fft_backend"] == "subq_ops_fused" else []
        )
        for name in names
    ]


def _ckconv(
    data_dim, spatial, fft_backend="torch_fft", is_causal=False, fft_padding="zero", grid_type="double", **kwargs
):
    """A small ``CKConvND`` with a SIREN kernel, zero-padded on a double grid unless told otherwise."""
    return CKConvND(
        data_dim=data_dim,
        hidden_dim=HIDDEN,
        kernel_cfg=LazyConfig(SIRENKernelND)(
            data_dim=data_dim,
            out_dim=HIDDEN,
            mlp_hidden_dim=16,
            num_layers=2,
            embedding_dim=16,
            omega_0=10.0,
            L_cache=spatial[0] if data_dim == 1 else list(spatial),
            use_bias=True,
        ),
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type=grid_type,
        fft_padding=fft_padding,
        is_causal=is_causal,
        fft_backend=fft_backend,
        **kwargs,
    )


def _operands(spatial, is_causal, dtype, device="cpu"):
    """Channels-first input ``[2, C, *spatial]`` and channels-last kernel ``[1, *kernel_spatial, C]``."""
    generator = torch.Generator().manual_seed(0)
    kernel_spatial = spatial if is_causal else tuple(2 * n - 1 for n in spatial)
    x = torch.randn(2, HIDDEN, *spatial, generator=generator)
    kernel = torch.randn(1, *kernel_spatial, HIDDEN, generator=generator)
    return x.to(device=device, dtype=dtype), kernel.to(device=device, dtype=dtype)


def _backend_op(ck, x, kernel, shortcut):
    """Call the op ``ck`` dispatches to for channels-first input, bypassing ``apply_convolution``."""
    return ck.fftconv_fn_bhl_input(x, kernel.movedim(-1, 1), shortcut)


def _split(case):
    """Split a case into ``_ckconv`` keyword arguments and the spatial shape."""
    case = dict(case)
    spatial = case.pop("spatial")
    return case, spatial


# ---------------------------------------------------------------------------
# CPU: torch_fft
# ---------------------------------------------------------------------------


@MODES
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", CPU_CASES.values(), ids=CPU_CASES.keys())
def test_full_precision_input_is_unchanged(case, dtype, mode):
    """Float32 and float64 inputs reach the op unchanged in either mode (float64 is not narrowed)."""
    kwargs, spatial = _split(case)
    ck = _ckconv(spatial=spatial, **mode, **kwargs).to(dtype)
    x, kernel = _operands(spatial, kwargs.get("is_causal", False), dtype)
    got = ck.apply_convolution(x, kernel, ck.shortcut, is_bhl_input=True)
    assert got.dtype == dtype
    assert torch.equal(got, _backend_op(ck, x, kernel, ck.shortcut))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", CPU_CASES.values(), ids=CPU_CASES.keys())
def test_low_precision_input_works_with_the_float32_shortcut(case, dtype):
    """By default ``torch_fft`` gets the input dtype, with the fp32 shortcut cast to it.

    ``torch_fft`` convolves in fp32, rounds, then adds the shortcut in the
    input dtype, as the ``subq_ops`` wrappers do.  Its ops reject a shortcut
    whose dtype differs from the input's, so this used to raise.
    """
    kwargs, spatial = _split(case)
    ck = _ckconv(spatial=spatial, **kwargs)
    x, kernel = _operands(spatial, kwargs.get("is_causal", False), dtype)
    got = ck.apply_convolution(x, kernel, ck.shortcut, is_bhl_input=True)
    assert got.dtype == dtype
    assert torch.equal(got, _backend_op(ck, x, kernel, ck.shortcut.to(dtype)))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", CPU_CASES.values(), ids=CPU_CASES.keys())
def test_float32_mode_computes_low_precision_input_in_float32(case, dtype):
    """``allow_reduced_precision=False``: fp32 convolution and shortcut term, rounded once."""
    kwargs, spatial = _split(case)
    ck = _ckconv(spatial=spatial, allow_reduced_precision=False, **kwargs)
    x, kernel = _operands(spatial, kwargs.get("is_causal", False), dtype)
    got = ck.apply_convolution(x, kernel, ck.shortcut, is_bhl_input=True)
    assert got.dtype == dtype
    assert torch.equal(got, _backend_op(ck, x.float(), kernel.float(), ck.shortcut).to(dtype))


@MODES
def test_bf16_forward_and_backward_reach_the_fp32_shortcut(mode):
    """A bf16 forward through ``torch_fft`` returns bf16 and trains the fp32 shortcut parameter."""
    torch.manual_seed(0)
    ck = _ckconv(data_dim=2, spatial=(8, 6), **mode)
    shortcut = ck.shortcut
    x = torch.randn(2, HIDDEN, 8, 6, dtype=torch.bfloat16, requires_grad=True)
    y = ck(x, is_bhl_input=True)
    assert y.dtype == torch.bfloat16
    y.float().square().sum().backward()
    assert ck.shortcut is shortcut and shortcut.dtype == torch.float32
    assert shortcut.grad.dtype == torch.float32
    assert torch.isfinite(shortcut.grad).all() and torch.count_nonzero(shortcut.grad) > 0
    assert x.grad.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "fft_padding, grid_type", [("circular", "single"), (["circular", "zero"], None)], ids=["circular", "mixed"]
)
@pytest.mark.parametrize("is_bhl_input", [True, False], ids=["bhl", "blh"])
def test_default_matches_casting_the_shortcut_parameter(fft_padding, grid_type, is_bhl_input):
    """Circular and mixed boundaries, both layouts: bf16 activations work with the fp32 shortcut.

    The output equals that of the same module with its shortcut parameter cast
    to bf16, the workaround these ``torch_fft`` ops needed before ``CKConvND``
    cast the shortcut.
    """
    torch.manual_seed(0)
    ck = _ckconv(data_dim=2, spatial=(8, 6), fft_padding=fft_padding, grid_type=grid_type)
    cast = copy.deepcopy(ck)
    cast.shortcut.data = cast.shortcut.data.to(torch.bfloat16)
    shape = (2, HIDDEN, 8, 6) if is_bhl_input else (2, 8, 6, HIDDEN)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    got = ck(x, is_bhl_input=is_bhl_input)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, cast(x, is_bhl_input=is_bhl_input))


def test_extra_repr_reports_precision():
    assert "allow_reduced_precision=True" in repr(_ckconv(data_dim=1, spatial=(16,)))
    assert "allow_reduced_precision=False" in repr(_ckconv(data_dim=1, spatial=(16,), allow_reduced_precision=False))


# ---------------------------------------------------------------------------
# GPU: subq_ops backends
# ---------------------------------------------------------------------------


@_requires_cuda_subq_ops
@pytest.mark.parametrize("name", _gpu_params(GPU_CASES))
def test_gpu_backends_get_the_input_dtype_by_default(name):
    """By default every backend gets bf16 operands, with the fp32 shortcut cast to bf16.

    ``subq_ops_fused`` and 1D non-causal ``subq_ops`` then compute in bf16;
    the 2D and 1D causal ``subq_ops`` paths upcast internally.  The wrappers
    cast the shortcut themselves too, so this matches the behaviour before
    ``CKConvND`` cast it.
    """
    kwargs, spatial = _split(GPU_CASES[name])
    ck = _ckconv(spatial=spatial, **kwargs).cuda()
    x, kernel = _operands(spatial, kwargs.get("is_causal", False), torch.bfloat16, device="cuda")
    got = ck.apply_convolution(x, kernel, ck.shortcut, is_bhl_input=True)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, _backend_op(ck, x, kernel, ck.shortcut))
    assert torch.equal(got, _backend_op(ck, x, kernel, ck.shortcut.to(torch.bfloat16)))


@_requires_cuda_subq_ops
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("name", _gpu_params(GPU_CASES))
def test_gpu_float32_mode_computes_every_backend_in_float32(name, dtype):
    """``allow_reduced_precision=False`` computes low-precision activations in fp32 on every backend.

    The forward is the op's fp32 result rounded once.  The input, kernel and
    shortcut gradients match the op's fp32 backward up to that rounding (some
    backward kernels accumulate with atomics, so the comparison is not
    bitwise).
    """
    torch.manual_seed(0)
    kwargs, spatial = _split(GPU_CASES[name])
    ck = _ckconv(spatial=spatial, allow_reduced_precision=False, **kwargs).cuda()
    x, kernel = _operands(spatial, kwargs.get("is_causal", False), dtype, device="cuda")
    x.requires_grad_()
    kernel.requires_grad_()
    got = ck.apply_convolution(x, kernel, ck.shortcut, is_bhl_input=True)
    x32 = x.detach().float().requires_grad_()
    kernel32 = kernel.detach().float().requires_grad_()
    shortcut32 = ck.shortcut.detach().clone().requires_grad_()
    want = _backend_op(ck, x32, kernel32, shortcut32)
    assert got.dtype == dtype
    assert torch.equal(got, want.to(dtype))

    cotangent = torch.randn(want.shape, device="cuda").to(dtype)
    got.backward(cotangent)
    want.backward(cotangent.float())
    assert x.grad.dtype == dtype and kernel.grad.dtype == dtype
    assert ck.shortcut.grad.dtype == torch.float32
    torch.testing.assert_close(x.grad, x32.grad.to(dtype))
    torch.testing.assert_close(kernel.grad, kernel32.grad.to(dtype))
    torch.testing.assert_close(ck.shortcut.grad, shortcut32.grad)
