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

"""Tests for the MLP module, including QuACK backend validation."""

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.mlp import MLP, _quack_mlp_available


# The quack backend is deliberately disabled in _validate_quack_backend
# (raises NotImplementedError) while backward correctness + benchmarks
# are still pending.  Detect this so quack-specific tests can skip.
_quack_backend_enabled: bool = False
if _quack_mlp_available:
    try:
        MLP(dim=384, activation="glu", dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0), backend="quack")
        _quack_backend_enabled = True
    except NotImplementedError:
        pass
    except Exception:
        _quack_backend_enabled = True  # different error → backend is enabled, test will surface the real bug


# ── Helpers ────────────────────────────────────────────────────────────────────

_GATED = ["glu", "swiglu"]
_NON_GATED = ["relu", "gelu", "silu"]
_ALL_ACTIVATIONS = _NON_GATED + _GATED

_dropout_cfg = LazyConfig(torch.nn.Dropout)(p=0.0)


def _make_mlp(
    dim: int = 384,
    activation: str = "glu",
    expansion_factor: float = 1.0,
    bias: bool = False,
    backend: str = "torch",
) -> MLP:
    return MLP(
        dim=dim,
        activation=activation,
        dropout_cfg=_dropout_cfg,
        expansion_factor=expansion_factor,
        bias=bias,
        backend=backend,
    )


# ── Shape tests (CPU, torch backend) ──────────────────────────────────────────


@pytest.mark.parametrize("activation", _ALL_ACTIVATIONS)
def test_mlp_output_shape(activation: str) -> None:
    """MLP preserves batch and sequence dims; output last dim == input last dim."""
    dim = 64
    mlp = _make_mlp(dim=dim, activation=activation)
    x = torch.randn(2, 16, dim)
    y = mlp(x)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"


@pytest.mark.parametrize("activation", _ALL_ACTIVATIONS)
def test_mlp_expansion_factor(activation: str) -> None:
    """Hidden dim scales correctly with expansion_factor."""
    dim, expansion = 64, 3.0
    mlp = _make_mlp(dim=dim, activation=activation, expansion_factor=expansion)
    assert mlp.hidden_dim == int(dim * expansion)
    glu_factor = 2 if activation in _GATED else 1
    assert mlp.layer1.out_features == mlp.hidden_dim * glu_factor


def test_mlp_bias() -> None:
    """Bias flag propagates to both linear layers."""
    mlp_no_bias = _make_mlp(bias=False)
    assert mlp_no_bias.layer1.bias is None
    assert mlp_no_bias.layer2.bias is None

    mlp_with_bias = _make_mlp(bias=True)
    assert mlp_with_bias.layer1.bias is not None
    assert mlp_with_bias.layer2.bias is not None


# ── Gradient tests (CPU, torch backend) ───────────────────────────────────────


@pytest.mark.parametrize("activation", _ALL_ACTIVATIONS)
def test_mlp_backward_cpu(activation: str) -> None:
    """Backward pass produces gradients for all parameters."""
    dim = 64
    mlp = _make_mlp(dim=dim, activation=activation)
    x = torch.randn(2, 16, dim, requires_grad=True)
    y = mlp(x)
    y.sum().backward()
    assert x.grad is not None
    for name, p in mlp.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


# ── FLOP count test ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("activation", _ALL_ACTIVATIONS)
def test_flop_count_positive(activation: str) -> None:
    """FLOP count returns a positive integer."""
    mlp = _make_mlp(dim=64, activation=activation)
    flops = mlp.flop_count(num_tokens=128)
    assert flops > 0


# ── Backend validation (no GPU needed) ────────────────────────────────────────


def test_quack_backend_rejects_bias() -> None:
    """backend='quack' must raise if bias=True."""
    if not _quack_backend_enabled:
        pytest.skip("quack backend not enabled (experimental / not installed)")
    with pytest.raises(ValueError, match="bias=False"):
        _make_mlp(dim=384, bias=True, backend="quack")


def test_quack_backend_rejects_bad_dim() -> None:
    """backend='quack' must raise if dim not divisible by 8."""
    if not _quack_backend_enabled:
        pytest.skip("quack backend not enabled (experimental / not installed)")
    with pytest.raises(ValueError, match="divisible by 8"):
        _make_mlp(dim=65, backend="quack")


def test_quack_backend_rejects_unsupported_activation() -> None:
    """backend='quack' must raise for unsupported activations."""
    if not _quack_backend_enabled:
        pytest.skip("quack backend not enabled (experimental / not installed)")
    with pytest.raises(ValueError, match="does not support activation"):
        _make_mlp(dim=384, activation="silu", backend="quack")


def test_quack_backend_unavailable_gives_clear_error() -> None:
    """When quack is not installed / version too old, backend='quack' must raise."""
    if _quack_mlp_available:
        pytest.skip("quack-kernels MLP is available — cannot test unavailability")
    with pytest.raises(ValueError, match="quack-kernels"):
        _make_mlp(dim=384, backend="quack")


def test_torch_backend_always_works() -> None:
    """backend='torch' never raises, regardless of dim/bias/activation."""
    mlp = _make_mlp(dim=65, activation="silu", bias=True, backend="torch")
    x = torch.randn(2, 16, 65)
    y = mlp(x)
    assert y.shape == x.shape


# ── GPU tests (skipped if no CUDA) ────────────────────────────────────────────

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@_requires_cuda
@pytest.mark.parametrize("activation", _ALL_ACTIVATIONS)
def test_mlp_forward_cuda(activation: str) -> None:
    """Forward pass on CUDA produces correct shapes (torch backend)."""
    dim = 384
    mlp = _make_mlp(dim=dim, activation=activation).cuda()
    x = torch.randn(2, 64, dim, device="cuda")
    y = mlp(x)
    assert y.shape == x.shape


@_requires_cuda
@pytest.mark.parametrize("activation", _ALL_ACTIVATIONS)
def test_mlp_backward_cuda(activation: str) -> None:
    """Backward pass on CUDA produces gradients for all parameters."""
    dim = 384
    mlp = _make_mlp(dim=dim, activation=activation).cuda()
    x = torch.randn(2, 64, dim, device="cuda", requires_grad=True)
    y = mlp(x)
    y.sum().backward()
    assert x.grad is not None
    for name, p in mlp.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


@_requires_cuda
@pytest.mark.parametrize("activation", ["glu", "swiglu", "gelu", "relu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mlp_quack_matches_pytorch(activation: str, dtype: torch.dtype) -> None:
    """When QuACK is available, fused output should be close to the PyTorch path."""
    if not _quack_backend_enabled:
        pytest.skip("quack backend not enabled (experimental / not installed)")

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("QuACK requires SM >= 9 (Hopper/Blackwell)")

    dim = 384
    torch.manual_seed(42)
    mlp_ref = _make_mlp(dim=dim, activation=activation, backend="torch").cuda().to(dtype)
    mlp_quack = _make_mlp(dim=dim, activation=activation, backend="quack").cuda().to(dtype)
    mlp_quack.load_state_dict(mlp_ref.state_dict())

    x = torch.randn(4, 64, dim, device="cuda", dtype=dtype)

    with torch.no_grad():
        ref = mlp_ref(x)
        fused = mlp_quack(x)

    atol = 0.1 if activation == "gelu" else 0.05
    torch.testing.assert_close(fused, ref, atol=atol, rtol=atol)


@_requires_cuda
@pytest.mark.parametrize("activation", ["glu", "swiglu"])
def test_mlp_quack_backward_matches_pytorch(activation: str) -> None:
    """Gradients from the QuACK backend should be close to the PyTorch backend."""
    if not _quack_backend_enabled:
        pytest.skip("quack backend not enabled (experimental / not installed)")

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("QuACK requires SM >= 9 (Hopper/Blackwell)")

    dim, dtype = 384, torch.bfloat16
    torch.manual_seed(42)

    mlp_ref = _make_mlp(dim=dim, activation=activation, backend="torch").cuda().to(dtype)
    mlp_quack = _make_mlp(dim=dim, activation=activation, backend="quack").cuda().to(dtype)
    mlp_quack.load_state_dict(mlp_ref.state_dict())

    x = torch.randn(4, 64, dim, device="cuda", dtype=dtype)

    x_ref = x.clone().requires_grad_(True)
    mlp_ref(x_ref).sum().backward()

    x_q = x.clone().requires_grad_(True)
    mlp_quack(x_q).sum().backward()

    torch.testing.assert_close(x_q.grad, x_ref.grad, atol=5e-2, rtol=5e-2)
