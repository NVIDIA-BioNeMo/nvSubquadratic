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

"""Integration tests for CKConvND with fft_backend='subq_ops_fused'.

Companion to ``test_ckconv_nd_subq.py`` (which covers ``fft_backend="subq_ops"``).
The fused backend adds two things that backend does not have: native
fp16/bf16 execution, and a hard 64x64 spatial cap enforced at forward time
rather than at construction.

Usage (requires GPU — run inside SLURM):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/modules/test_ckconv_nd_subq_fused.py -v -o addopts=""
"""

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from tests.conftest import requires_sm90, requires_subq_ops_fused


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

pytestmark = [requires_subq_ops_fused, requires_cuda]

HIDDEN_DIM = 32
SPATIAL = 8

# Normwise relative-error budgets against the fp32 torch_fft backend.
L2_TOL = {torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 8e-3}
L2_TOL_GRAD = {torch.float32: 1e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}


def _l2_rel(pred, ref):
    pred64, ref64 = pred.double(), ref.double()
    den = ref64.norm()
    return ((pred64 - ref64).norm() / den).item() if den > 0 else (pred64 - ref64).norm().item()


def _assert_l2_close(pred, ref, dtype, tol_table=None, name=""):
    tol = (tol_table or L2_TOL)[dtype]
    rel = _l2_rel(pred, ref)
    assert rel < tol, f"{name} L2 rel error {rel:.3e} exceeds tol {tol:.1e}"


def _make_ckconv(grid_type, fft_backend, use_chunked=False, spatial=SPATIAL, data_dim=2, **kw):
    """Build a CKConvND with a small SIREN kernel, mirroring test_ckconv_nd_subq.py."""
    kernel_cfg = LazyConfig(SIRENKernelND)(
        data_dim=data_dim,
        out_dim=HIDDEN_DIM,
        mlp_hidden_dim=16,
        num_layers=2,
        embedding_dim=16,
        omega_0=10.0,
        L_cache=spatial,
        use_bias=True,
    )
    kw.setdefault("fft_padding", "zero")
    return CKConvND(
        data_dim=data_dim,
        hidden_dim=HIDDEN_DIM,
        kernel_cfg=kernel_cfg,
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type=grid_type,
        fft_backend=fft_backend,
        use_chunked_fftconv=use_chunked,
        **kw,
    )


def _paired_models(grid_type, use_chunked=False, spatial=SPATIAL):
    """Build torch_fft and subq_ops_fused models sharing identical weights."""
    reference = _make_ckconv(grid_type, "torch_fft", spatial=spatial).cuda()
    fused = _make_ckconv(grid_type, "subq_ops_fused", use_chunked, spatial=spatial).cuda()
    fused.load_state_dict(reference.state_dict())
    return reference, fused


# ---------------------------------------------------------------------------
# Forward / backward equivalence with torch_fft
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid_type", ["single", "double"])
@pytest.mark.parametrize("use_chunked", [False, True])
class TestForwardMatchesTorchFft:
    def test_blh_layout(self, grid_type, use_chunked):
        torch.manual_seed(42)
        reference, fused = _paired_models(grid_type, use_chunked)
        x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda")
        _assert_l2_close(fused(x), reference(x), torch.float32)

    def test_bhl_layout(self, grid_type, use_chunked):
        torch.manual_seed(42)
        reference, fused = _paired_models(grid_type, use_chunked)
        x = torch.randn(2, HIDDEN_DIM, SPATIAL, SPATIAL, device="cuda")
        _assert_l2_close(fused(x, is_bhl_input=True), reference(x, is_bhl_input=True), torch.float32)


def _cast_params(model, dtype):
    """Cast parameters only, leaving buffers alone.

    ``model.to(dtype)`` would also cast ``SIRENKernelND.grid_cache``, which the
    kernel asserts must stay fp32 (see the TODO at ``kernels_nd.py``) — the
    positional embedding casts the grid to the weight dtype internally instead.
    The buffer is non-persistent, so skipping it keeps ``state_dict`` parity.
    """
    for param in model.parameters():
        param.data = param.data.to(dtype)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_native_dtype(dtype):
    """The fused backend runs natively in the input dtype.

    Both backends are cast to ``dtype`` so the SIREN generates the *same*
    low-precision kernel for each; the only remaining difference is the
    convolution itself, which torch_fft still evaluates in fp32 while the fused
    kernel evaluates in ``dtype``. Comparing against an fp32 reference model
    instead would fold in the SIREN's own low-precision error (~3e-2 in bf16),
    which has nothing to do with this backend.
    """
    torch.manual_seed(42)
    reference, fused = _paired_models("double")
    _cast_params(reference, dtype)
    _cast_params(fused, dtype)
    x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda", dtype=dtype)

    out = fused(x)
    assert out.dtype == dtype
    _assert_l2_close(out, reference(x), dtype)


@pytest.mark.parametrize("grid_type", ["single", "double"])
def test_backward_matches_torch_fft(grid_type):
    """Gradients flow to the SIREN kernel parameters and match torch_fft."""
    torch.manual_seed(42)
    reference, fused = _paired_models(grid_type)
    x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda")
    grad_out = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda")

    reference(x).backward(grad_out)
    fused(x).backward(grad_out)

    ref_grads = dict(reference.named_parameters())
    assert any(p.grad is not None for p in ref_grads.values())
    for name, param in fused.named_parameters():
        want = ref_grads[name].grad
        if want is None:
            assert param.grad is None, f"{name}: fused produced a grad where torch_fft did not"
            continue
        _assert_l2_close(param.grad, want, torch.float32, tol_table=L2_TOL_GRAD, name=name)


@pytest.mark.parametrize("spatial", [7, 16, 32, pytest.param(64, marks=requires_sm90)])
def test_spatial_sizes_within_cap(spatial):
    """Every supported spatial size, including the 64 boundary and a non-power-of-2."""
    torch.manual_seed(42)
    reference, fused = _paired_models("double", spatial=spatial)
    x = torch.randn(1, spatial, spatial, HIDDEN_DIM, device="cuda")
    _assert_l2_close(fused(x), reference(x), torch.float32)


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


class TestRejectsInvalidConfigs:
    """Unsupported configurations fail at construction, except the input-size cap."""

    @pytest.mark.parametrize("data_dim", [1, 3])
    def test_rejects_non_2d(self, data_dim):
        with pytest.raises(AssertionError, match="only supports data_dim=2"):
            _make_ckconv("double", "subq_ops_fused", data_dim=data_dim)

    def test_rejects_causal(self):
        # data_dim=1 is rejected first, so causal must be probed on the 2D path,
        # where CKConvND's own "causal is 1D only" assert fires first.
        with pytest.raises(AssertionError, match="Causal CKConvND only supports 1D"):
            _make_ckconv("double", "subq_ops_fused", is_causal=True)

    def test_rejects_circular_padding(self):
        with pytest.raises(AssertionError, match="only supports zero-padded"):
            _make_ckconv("single", "subq_ops_fused", fft_padding="circular")

    def test_rejects_per_axis_padding(self):
        with pytest.raises(ValueError, match="does not support a per-axis fft_padding"):
            _make_ckconv(None, "subq_ops_fused", fft_padding=["circular", "zero"])

    def test_rejects_unknown_backend(self):
        with pytest.raises(AssertionError, match="Invalid fft_backend"):
            _make_ckconv("double", "subq_ops_fusedd")

    def test_oversized_input_raises_at_forward(self):
        """The 64x64 cap depends on the input, so it is enforced on the first call."""
        torch.manual_seed(42)
        model = _make_ckconv("double", "subq_ops_fused", spatial=65).cuda()
        x = torch.randn(1, 65, 65, HIDDEN_DIM, device="cuda")
        with pytest.raises(ValueError, match="too large for the fused 2D FFT kernel"):
            model(x)


def test_extra_repr_reports_backend():
    assert "fft_backend='subq_ops_fused'" in repr(_make_ckconv("double", "subq_ops_fused"))
