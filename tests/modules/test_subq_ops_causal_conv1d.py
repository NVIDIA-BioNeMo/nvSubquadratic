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

"""Tests for SubqOpsCausalConv1d module.

Validates :class:`nvsubquadratic.modules.subq_ops_causal_conv1d.SubqOpsCausalConv1d`
against the existing :class:`nvsubquadratic.modules.causal_conv1d.CausalConv1D`
torch reference.

Covers:
  - Forward correctness (no bias and with bias)
  - Backward gradient correctness
  - SiLU activation path
  - Construction-time constraint asserts (depthwise-only, valid activation)
  - State-dict round-trip with the torch reference (shared parameter layout)

Usage (requires GPU):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/modules/test_subq_ops_causal_conv1d.py -v -o addopts=""
"""

import pytest
import torch
import torch.nn.functional as F

from nvsubquadratic.modules.causal_conv1d import CausalConv1D


ATOL = 1e-3
RTOL = 1e-4
ATOL_GRAD = 1e-2
RTOL_GRAD = 1e-3


def _has_subq_ops() -> bool:
    try:
        from subquadratic_ops_torch.causal_conv1d import causal_conv1d  # noqa: F401

        return True
    except ImportError:
        return False


requires_subq_ops = pytest.mark.skipif(
    not _has_subq_ops(),
    reason="subquadratic_ops_torch.causal_conv1d not available",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

pytestmark = [requires_subq_ops, requires_cuda]


def _import_module():
    from nvsubquadratic.modules.subq_ops_causal_conv1d import SubqOpsCausalConv1d

    return SubqOpsCausalConv1d


class TestForward:
    @pytest.mark.parametrize("C, L, K", [(36, 256, 3), (64, 512, 7), (16, 128, 4)])
    def test_matches_causal_conv1d(self, C, L, K):
        SubqOpsCausalConv1d = _import_module()
        torch.manual_seed(42)
        ref = CausalConv1D(C, C, K, groups=C, bias=False).cuda()
        fast = SubqOpsCausalConv1d(C, C, K, groups=C, bias=False).cuda()
        fast.load_state_dict(ref.state_dict())

        x = torch.randn(2, C, L, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(fast(x), ref(x), atol=ATOL, rtol=RTOL)

    def test_with_bias(self):
        SubqOpsCausalConv1d = _import_module()
        torch.manual_seed(42)
        C, L, K = 36, 256, 3
        ref = CausalConv1D(C, C, K, groups=C, bias=True).cuda()
        fast = SubqOpsCausalConv1d(C, C, K, groups=C, bias=True).cuda()
        fast.load_state_dict(ref.state_dict())
        x = torch.randn(2, C, L, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(fast(x), ref(x), atol=ATOL, rtol=RTOL)

    def test_silu_activation(self):
        """activation='silu' applies SiLU after the conv."""
        SubqOpsCausalConv1d = _import_module()
        torch.manual_seed(42)
        C, L, K = 36, 256, 3
        ref = CausalConv1D(C, C, K, groups=C, bias=False).cuda()
        fast = SubqOpsCausalConv1d(C, C, K, groups=C, bias=False, activation="silu").cuda()
        fast.load_state_dict(ref.state_dict())
        x = torch.randn(2, C, L, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(fast(x), F.silu(ref(x)), atol=ATOL, rtol=RTOL)


class TestAutocast:
    """Mixed precision, where the module previously could not run at all.

    Every other test here feeds fp32 activations to fp32 parameters, so all
    tensors happen to agree and the kernel's dtype check passes. Under
    ``torch.autocast`` the activations are bf16/fp16 while the parameters stay
    fp32, and the kernel — which selects its specialisation from the *input*
    dtype and then requires an exact match — aborted with
    ``ValueError: in_w expected dtype (code=4, bits=16) but got (code=2, bits=32)``.
    """

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_runs_under_autocast(self, dtype):
        SubqOpsCausalConv1d = _import_module()
        torch.manual_seed(42)
        C, L, K = 36, 256, 3
        fast = SubqOpsCausalConv1d(C, C, K, groups=C, bias=False).cuda()
        x = torch.randn(2, C, L, device="cuda", dtype=torch.float32)

        with torch.autocast("cuda", dtype=dtype):
            out = fast(x.to(dtype))

        assert out.dtype is dtype
        # fp32 master parameters must survive: only the kernel's inputs are narrowed.
        assert fast.weight.dtype is torch.float32

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_autocast_matches_reference(self, dtype):
        SubqOpsCausalConv1d = _import_module()
        torch.manual_seed(42)
        C, L, K = 36, 256, 3
        ref = CausalConv1D(C, C, K, groups=C, bias=False).cuda()
        fast = SubqOpsCausalConv1d(C, C, K, groups=C, bias=False).cuda()
        fast.load_state_dict(ref.state_dict())
        x = torch.randn(2, C, L, device="cuda", dtype=torch.float32)

        with torch.autocast("cuda", dtype=dtype):
            got = fast(x.to(dtype))

        # Tolerance follows the narrowed dtype, not the fp32 ATOL above.
        tol = 5e-2 if dtype is torch.bfloat16 else 5e-3
        torch.testing.assert_close(got.float(), ref(x).float(), atol=tol, rtol=tol)


class TestBackward:
    def test_grad_matches_causal_conv1d(self):
        SubqOpsCausalConv1d = _import_module()
        torch.manual_seed(42)
        C, L, K = 36, 256, 3
        ref = CausalConv1D(C, C, K, groups=C, bias=True).cuda()
        fast = SubqOpsCausalConv1d(C, C, K, groups=C, bias=True).cuda()
        fast.load_state_dict(ref.state_dict())

        x_ref = torch.randn(2, C, L, device="cuda", dtype=torch.float32, requires_grad=True)
        x_fast = x_ref.detach().clone().requires_grad_(True)
        ref(x_ref).sum().backward()
        fast(x_fast).sum().backward()

        torch.testing.assert_close(x_fast.grad, x_ref.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)
        torch.testing.assert_close(fast.weight.grad, ref.weight.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)
        torch.testing.assert_close(fast.bias.grad, ref.bias.grad, atol=ATOL_GRAD, rtol=RTOL_GRAD)


class TestConstruction:
    def test_rejects_non_depthwise(self):
        SubqOpsCausalConv1d = _import_module()
        with pytest.raises(ValueError, match="depthwise-only"):
            SubqOpsCausalConv1d(in_channels=16, out_channels=16, kernel_size=3, groups=4)

    def test_rejects_in_neq_out(self):
        SubqOpsCausalConv1d = _import_module()
        with pytest.raises(ValueError, match="depthwise-only"):
            SubqOpsCausalConv1d(in_channels=16, out_channels=32, kernel_size=3, groups=16)

    def test_rejects_invalid_activation(self):
        SubqOpsCausalConv1d = _import_module()
        with pytest.raises(ValueError, match="activation"):
            SubqOpsCausalConv1d(in_channels=16, out_channels=16, kernel_size=3, groups=16, activation="relu")

    def test_is_nn_conv1d_subclass(self):
        """Hyena's isinstance(short_conv, nn.Conv1d) check must pass."""
        SubqOpsCausalConv1d = _import_module()
        m = SubqOpsCausalConv1d(in_channels=16, out_channels=16, kernel_size=3, groups=16)
        assert isinstance(m, torch.nn.Conv1d)
