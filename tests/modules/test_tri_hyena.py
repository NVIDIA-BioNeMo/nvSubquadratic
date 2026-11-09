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

"""Tests for TriangleHyena.

Covers the pair-tensor contract, exact parity with an eager re-implementation of
the reference TriHyena core (both node types, padding, hard window), the
reference checkpoint port, row/column structure, padding semantics, hard and
soft windows, integration with QKVSequenceMixer / ResidualBlock, edge sizes and
the raised errors.  CPU on the torch_fft backend; the subq_ops backend is
compared against it on GPU when available.

Usage:
    PYTHONPATH=. python -m pytest tests/modules/test_tri_hyena.py -v -o addopts=""
"""

import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import BoxModulationND, GaussianModulationND
from nvsubquadratic.modules.residual_block import AdaLNZeroResidualBlock, ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.tri_hyena import TriangleHyena, _StencilConv, load_reference_state_dict


B, L, C = 2, 6, 8
_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
_requires_subq_ops = pytest.mark.skipif(
    importlib.util.find_spec("subquadratic_ops_torch") is None, reason="subquadratic_ops_torch not installed"
)


def mixer_cfg(
    start_node=True,
    mask_cfg=None,
    seq_len=L,
    hidden_dim=C,
    fft_backend="torch_fft",
    short_conv_backend="torch",
    L_cache=None,
    use_shortcut=True,
    qkv_bias=False,
    out_proj_bias=True,
):
    """LazyConfig for a ``QKVSequenceMixer`` wrapping ``TriangleHyena`` with a SIREN ``CKConvND``."""
    if mask_cfg is None:
        mask_cfg = LazyConfig(torch.nn.Identity)()
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=hidden_dim,
        qkv_bias=qkv_bias,
        out_proj_bias=out_proj_bias,
        mixer_cfg=LazyConfig(TriangleHyena)(
            hidden_dim=hidden_dim,
            start_node=start_node,
            short_conv_backend=short_conv_backend,
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=1,
                hidden_dim=hidden_dim,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    out_dim=hidden_dim,
                    data_dim=1,
                    mlp_hidden_dim=16,
                    num_layers=3,
                    embedding_dim=16,
                    omega_0=30.0,
                    L_cache=L_cache or seq_len,
                    use_bias=True,
                ),
                mask_cfg=mask_cfg,
                grid_type="double",
                fft_padding="zero",
                is_causal=False,
                fft_backend=fft_backend,
                use_shortcut=use_shortcut,
            ),
        ),
    )


def make_mixer(seed=0, dtype=torch.float32, **kwargs):
    """Seeded mixer instantiated from ``mixer_cfg`` with parameters cast to ``dtype``."""
    torch.manual_seed(seed)
    return cast_params(instantiate(mixer_cfg(**kwargs)), dtype)


def cast_params(module, dtype):
    """Cast parameters only; the SIREN ``grid_cache`` buffer must stay fp32."""
    for param in module.parameters():
        param.data = param.data.to(dtype)
    return module


def box_cfg(max_offset):
    """LazyConfig for a 1D ``BoxModulationND`` window."""
    return LazyConfig(BoxModulationND)(data_dim=1, max_offset=max_offset)


class FakeGroup:
    """Stand-in process group of size 2."""

    def size(self):
        """Return the group size."""
        return 2


# ---------------------------------------------------------------------------
# Eager re-implementation of the TriHyena core ("nwh" layout)
# ---------------------------------------------------------------------------


def ref_causal_conv(x, w):
    """Reference short conv: out[t] = sum_k w[k] x[t - (K - 1) + k]; x [N, L, C], w [C, K]."""
    kernel = w.shape[-1]
    return F.conv1d(F.pad(x.transpose(1, 2), (kernel - 1, 0)), w.unsqueeze(1), groups=w.shape[0]).transpose(1, 2)


def ref_long_conv(z, h):
    """Reference long conv: out[j] = sum_k h[L - 1 + j - k] z[k]; z [N, L, C], h [C, 2L - 1]."""
    length = z.shape[1]
    out = torch.zeros_like(z)
    for j in range(length):
        for k in range(length):
            out[:, j] += h[:, length - 1 + j - k] * z[:, k]
    return out


def ref_core(x1, x2, v, h, w_conv, padding, start_node):
    """Reference TriHyena core on projections [B, L, L, C]; w_conv [3C, 3] over [x1 | x2 | v]."""
    b, length, _, c = x1.shape
    row = (lambda t: t) if start_node else (lambda t: t.transpose(1, 2))
    col = (lambda t: t.transpose(1, 2)) if start_node else (lambda t: t)

    def flat(t):
        return t.reshape(b * length, length, c)

    pair = (padding[:, :, None] | padding[:, None, :])[..., None]
    x1, x2, v = (t.masked_fill(pair, 0) for t in (x1, x2, v))
    x1 = ref_causal_conv(flat(row(x1)), w_conv[:c])
    x2 = ref_causal_conv(flat(row(x2)), w_conv[c : 2 * c])
    v = ref_causal_conv(flat(col(v)), w_conv[2 * c :])
    mask = (padding[:, :, None] | padding[:, None, :]).reshape(b * length, length, 1)
    x1 = x1.masked_fill(mask, 0)
    x2 = x2.masked_fill(mask, 0)
    y = (x2 * ref_long_conv(x1 * v, h)).reshape(b, length, length, c)
    return y if start_node else y.transpose(1, 2)


def ref_from_mixer(mixer, x, padding):
    """Evaluate ref_core with the parameters of a `QKVSequenceMixer(TriangleHyena)`."""
    tri = mixer.mixer
    c = mixer.qkv_proj.out_features // 3
    q, k, v = mixer.qkv_proj(x).chunk(3, dim=-1)
    short = tri.short_conv_weight.t()
    w_conv = torch.cat([short[:c], short[2 * c :], short[c : 2 * c]])
    kernel, grid = tri.global_conv.kernel([x.shape[1]])
    mask = tri.global_conv.mask
    h = (kernel if isinstance(mask, torch.nn.Identity) else mask(grid=grid, x=kernel))[0].t()
    y = ref_core(q, v, k, h, w_conv, padding, tri.start_node)
    return mixer.out_proj(y)


def ref_siren_filter(ref, length, hidden_layers):
    """Reference SIREN filter for L_cache = 2L - 1: coordinates linspace(-0.5, 0.5, 2L - 1)."""
    t = torch.linspace(-0.5, 0.5, 2 * length - 1, dtype=torch.float64).unsqueeze(-1)
    h = torch.sin(
        F.linear(
            t,
            ref["siren_kernel.positional_embedding.linear.weight"],
            ref["siren_kernel.positional_embedding.linear.bias"],
        )
    )
    for i in range(hidden_layers):
        h = torch.sin(
            F.linear(
                h, ref[f"siren_kernel.kernel_network.{2 * i}.weight"], ref[f"siren_kernel.kernel_network.{2 * i}.bias"]
            )
        )
    return F.linear(h, ref["siren_kernel.out_linear.weight"], ref["siren_kernel.out_linear.bias"]).t()


def reference_state_dict(c, seed):
    """Random float64 state dict in the reference TriHyena checkpoint layout."""
    torch.manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, dtype=torch.float64)

    return {
        "norm.weight": r(c),
        "norm.bias": r(c),
        "linear_x12i.weight": r(2 * c, c) / math.sqrt(c),
        "linear_vj.weight": r(c, c) / math.sqrt(c),
        "to_out.weight": r(c, c) / math.sqrt(c),
        "to_out.bias": r(c),
        "W_conv_x12i_vj": r(3 * c, 3),
        "siren_kernel.positional_embedding.linear.weight": r(16, 1) * 30.0,
        "siren_kernel.positional_embedding.linear.bias": r(16),
        "siren_kernel.kernel_network.0.weight": r(16, 16) / 4,
        "siren_kernel.kernel_network.0.bias": r(16),
        "siren_kernel.kernel_network.2.weight": r(16, 16) / 4,
        "siren_kernel.kernel_network.2.bias": r(16),
        "siren_kernel.out_linear.weight": r(c, 16) / 4,
        "siren_kernel.out_linear.bias": r(c),
    }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start_node", [True, False])
def test_output_shape_and_gradients(start_node) -> None:
    """Output keeps the input shape and every parameter receives a finite, non-zero gradient."""
    mixer = make_mixer(start_node=start_node, mask_cfg=LazyConfig(GaussianModulationND)(data_dim=1, num_channels=C))
    x = torch.randn(B, L, L, C, requires_grad=True)
    y = mixer(x)
    assert y.shape == (B, L, L, C)
    assert y.is_contiguous()
    assert torch.isfinite(y).all()
    y.square().sum().backward()
    tri = mixer.mixer
    for name, param in [
        ("qkv_proj", mixer.qkv_proj.weight),
        ("out_proj", mixer.out_proj.weight),
        ("short_conv", tri.short_conv_weight),
        ("siren", tri.global_conv.kernel.out_linear.weight),
        ("mask std", tri.global_conv.mask.std_param),
        ("shortcut", tri.global_conv.shortcut),
    ]:
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0, name
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_bfloat16_runs() -> None:
    """bfloat16 mixer returns a finite bfloat16 output."""
    mixer = cast_params(make_mixer(), torch.bfloat16)
    y16 = mixer(torch.randn(B, L, L, C, dtype=torch.bfloat16))
    assert y16.dtype == torch.bfloat16
    assert torch.isfinite(y16).all()


# ---------------------------------------------------------------------------
# Parity with the reference core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start_node", [True, False])
@pytest.mark.parametrize("mask_cfg", [None, box_cfg(2)], ids=["no-window", "box-2"])
def test_matches_reference_core(start_node, mask_cfg) -> None:
    """Mixer matches the eager reference core with and without a box window."""
    mixer = make_mixer(start_node=start_node, mask_cfg=mask_cfg, dtype=torch.float64)
    mixer.mixer.global_conv.shortcut.data.zero_()
    x = torch.randn(B, L, L, C, dtype=torch.float64)
    padding = torch.zeros(B, L, dtype=torch.bool)
    padding[1, -2:] = True
    with torch.no_grad():
        got = mixer(x, padding_mask=padding)
        want = ref_from_mixer(mixer, x, padding)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("start_node", [True, False])
def test_reference_checkpoint_port(start_node) -> None:
    """A ported reference checkpoint reproduces the reference forward pass."""
    ref = reference_state_dict(C, seed=3)
    mixer = make_mixer(start_node=start_node, mask_cfg=box_cfg(2), L_cache=2 * L - 1, dtype=torch.float64)
    norm = load_reference_state_dict(mixer, {f"core.{k}": v for k, v in ref.items()})
    assert norm["weight"] is ref["norm.weight"] and norm["bias"] is ref["norm.bias"]
    assert torch.equal(mixer.mixer.global_conv.shortcut, torch.zeros(C, dtype=torch.float64))

    x = torch.randn(B, L, L, C, dtype=torch.float64)
    padding = torch.zeros(B, L, dtype=torch.bool)
    padding[0, 0] = True
    x1 = x @ ref["linear_x12i.weight"][:C].t()
    x2 = x @ ref["linear_x12i.weight"][C:].t()
    v = x @ ref["linear_vj.weight"].t()
    h = ref_siren_filter(ref, L, hidden_layers=2)
    offset = (torch.arange(2 * L - 1) - (L - 1)).abs()
    h = h.masked_fill(offset > 2, 0)
    y = ref_core(x1, x2, v, h, ref["W_conv_x12i_vj"], padding, start_node)
    want = y @ ref["to_out.weight"].t() + ref["to_out.bias"]
    with torch.no_grad():
        got = mixer(x, padding_mask=padding)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_reference_checkpoint_port_without_shortcut() -> None:
    """Porting into a mixer without a shortcut matches one with a zeroed shortcut."""
    ref = reference_state_dict(C, seed=3)
    with_shortcut = make_mixer(mask_cfg=box_cfg(2), L_cache=2 * L - 1, dtype=torch.float64)
    without = make_mixer(mask_cfg=box_cfg(2), L_cache=2 * L - 1, dtype=torch.float64, use_shortcut=False)
    assert without.mixer.global_conv.shortcut is None
    assert "mixer.global_conv.shortcut" not in without.state_dict()
    load_reference_state_dict(with_shortcut, ref)
    load_reference_state_dict(without, ref)
    w_x1, w_x2, w_v = ref["W_conv_x12i_vj"].split(C)
    assert torch.equal(without.mixer.short_conv_weight, torch.cat([w_x1, w_v, w_x2]).t())
    x = torch.randn(B, L, L, C, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(without(x), with_shortcut(x), atol=1e-12, rtol=1e-12)
    assert without.mixer.flop_count((L, L)) < with_shortcut.mixer.flop_count((L, L))


def test_reference_checkpoint_port_zeroes_qkv_bias() -> None:
    """Porting zeroes an existing qkv bias."""
    ref = reference_state_dict(C, seed=3)
    without = make_mixer(mask_cfg=box_cfg(2), L_cache=2 * L - 1, dtype=torch.float64)
    with_bias = make_mixer(mask_cfg=box_cfg(2), L_cache=2 * L - 1, dtype=torch.float64, qkv_bias=True)
    with_bias.qkv_proj.bias.data.fill_(1.0)
    load_reference_state_dict(without, ref)
    load_reference_state_dict(with_bias, ref)
    assert torch.equal(with_bias.qkv_proj.bias, torch.zeros(3 * C, dtype=torch.float64))
    x = torch.randn(B, L, L, C, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(with_bias(x), without(x), atol=1e-12, rtol=1e-12)


def test_reference_checkpoint_port_shape_mismatch() -> None:
    """Mis-shaped or missing checkpoint tensors raise ``ValueError`` naming the key."""
    ref = reference_state_dict(C, seed=3)
    mixer = make_mixer(L_cache=2 * L - 1)
    ref["W_conv_x12i_vj"] = ref["W_conv_x12i_vj"][:, :2]
    with pytest.raises(ValueError, match="short_conv_weight"):
        load_reference_state_dict(mixer, ref)
    del ref["to_out.bias"]
    with pytest.raises(ValueError, match=r"to_out\.bias"):
        load_reference_state_dict(mixer, ref)


def test_reference_checkpoint_port_siren_depth_mismatch() -> None:
    """A SIREN depth mismatch raises ``ValueError`` and leaves the mixer unchanged."""
    mixer = make_mixer(L_cache=2 * L - 1)
    before = {k: v.clone() for k, v in mixer.state_dict().items()}
    deeper = {**reference_state_dict(C, seed=3), "siren_kernel.kernel_network.4.weight": torch.zeros(16, 16)}
    with pytest.raises(ValueError, match="more than the mixer's 2 hidden layers"):
        load_reference_state_dict(mixer, deeper)
    shallower = {k: v for k, v in reference_state_dict(C, seed=3).items() if "kernel_network.2" not in k}
    with pytest.raises(ValueError, match=r"kernel_network\.2"):
        load_reference_state_dict(mixer, shallower)
    for k, v in mixer.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)


@pytest.mark.parametrize("key", ["norm.weight", "norm.bias"])
def test_reference_checkpoint_port_missing_norm_leaves_mixer_unchanged(key) -> None:
    """A missing norm tensor raises ``ValueError`` and leaves the mixer unchanged."""
    ref = reference_state_dict(C, seed=3)
    del ref[key]
    mixer = make_mixer(L_cache=2 * L - 1)
    before = {k: v.clone() for k, v in mixer.state_dict().items()}
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        load_reference_state_dict(mixer, ref)
    assert all(torch.equal(before[k], v) for k, v in mixer.state_dict().items())


# ---------------------------------------------------------------------------
# Row / column structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start_node", [True, False])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("with_mask", [False, True], ids=["no-mask", "mask"])
def test_fused_qkv_matches_separate_tensors(start_node, requires_grad, with_mask) -> None:
    """Fused ``[..., 3C]`` qkv input matches separate q, k, v tensors in output and gradients."""
    tri = make_mixer(start_node=start_node, dtype=torch.float64).mixer
    qkv = torch.randn(B, L, L, 3 * C, dtype=torch.float64, requires_grad=requires_grad)
    copies = [t.detach().clone().requires_grad_(requires_grad) for t in qkv.chunk(3, dim=-1)]
    padding = torch.zeros(B, L, dtype=torch.bool)
    padding[1, -2:] = with_mask
    got, want = tri(qkv, padding_mask=padding), tri(*copies, padding_mask=padding)
    assert torch.equal(got, want)
    if requires_grad:
        got.square().sum().backward()
        want.square().sum().backward()
        assert torch.equal(qkv.grad, torch.cat([t.grad for t in copies], dim=-1))


class RecordingMixer(torch.nn.Module):
    """Inner mixer that records its call arguments."""

    def __init__(self, fused):
        """Optionally declare ``takes_fused_qkv``."""
        super().__init__()
        if fused:
            self.takes_fused_qkv = True
        self.calls = []

    def forward(self, *args, **kwargs):
        """Record the call and return the first ``C`` channels of the first argument."""
        self.calls.append((args, kwargs))
        return args[0][..., :C]


@pytest.mark.parametrize("fused", [False, True])
def test_sequence_mixer_fused_qkv_contract(fused) -> None:
    """``QKVSequenceMixer`` passes fused or split qkv according to ``takes_fused_qkv``."""
    assert TriangleHyena.takes_fused_qkv is True
    mixer = QKVSequenceMixer(hidden_dim=C, mixer_cfg=LazyConfig(RecordingMixer)(fused=fused))
    x = torch.randn(B, L, L, C)
    mixer(x, conditioning="cond")
    ((args, kwargs),) = mixer.mixer.calls
    qkv = mixer.qkv_proj(x)
    if fused:
        assert len(args) == 1 and torch.equal(args[0], qkv)
        assert kwargs == {"cp_group": None, "conditioning": "cond"}
    else:
        assert len(args) == 4 and args[3] is None
        assert all(torch.equal(a, c) for a, c in zip(args[:3], qkv.chunk(3, dim=-1)))
        assert kwargs == {"conditioning": "cond"}


@pytest.mark.parametrize("taps", [1, 3, 4])
def test_stencil_conv_matches_conv1d_and_gradcheck(taps) -> None:
    """``_StencilConv`` matches the reference causal conv and passes gradcheck."""
    torch.manual_seed(0)
    x = torch.randn(3, 7, 2 * C, dtype=torch.float64)[:, :, ::2].requires_grad_()
    w = torch.randn(taps, C, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(_StencilConv.apply(x, w), ref_causal_conv(x, w.t()), atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(_StencilConv.apply, (x, w))


def test_partial_qkv_raises() -> None:
    """Two tensors or a fused input of the wrong width raise ``ValueError``."""
    tri = make_mixer().mixer
    q, k = torch.randn(2, B, L, L, C)
    with pytest.raises(ValueError, match="fused qkv"):
        tri(q, k)
    with pytest.raises(ValueError, match=r"C=8"):
        tri(torch.randn(B, L, L, 2 * C))


def test_end_node_is_start_node_on_transposed_pair() -> None:
    """Ending-node update equals the starting-node update on the transposed pair."""
    start = make_mixer(start_node=True, dtype=torch.float64)
    end = make_mixer(start_node=False, dtype=torch.float64)
    end.load_state_dict(start.state_dict())
    x = torch.randn(B, L, L, C, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(end(x), start(x.transpose(1, 2)).transpose(1, 2), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("start_node", [True, False])
def test_rows_are_independent_except_through_the_column_stream(start_node) -> None:
    """Perturbing one row of q or v changes only that row; perturbing k changes all rows."""
    tri = make_mixer(start_node=start_node, dtype=torch.float64).mixer
    q, k, v = (torch.randn(B, L, L, C, dtype=torch.float64) for _ in range(3))
    axis = 1 if start_node else 2
    others = torch.tensor([i for i in range(L) if i != 2])
    with torch.no_grad():
        base = tri(q, k, v)
        for name, tensor in (("query", q), ("value", v)):
            perturbed = tensor.clone()
            perturbed.select(axis, 2).normal_()
            out = tri(perturbed if name == "query" else q, k, perturbed if name == "value" else v)
            delta = (out - base).abs()
            assert delta.select(axis, 2).max() > 1e-3, name
            assert torch.equal(out.index_select(axis, others), base.index_select(axis, others)), name
        perturbed = k.clone()
        perturbed.select(axis, 2).normal_()
        delta = (tri(q, perturbed, v) - base).abs()
    assert (delta.index_select(axis, others).amax(dim=(0, 3 - axis, 3)) > 1e-3).all()


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start_node", [True, False])
def test_padding_mask_zeroes_padded_rows_and_columns(start_node) -> None:
    """Padded rows and columns of the inner mixer output are zero."""
    mixer = make_mixer(start_node=start_node, dtype=torch.float64)
    x = torch.randn(B, L, L, C, dtype=torch.float64)
    padding = torch.zeros(B, L, dtype=torch.bool)
    padding[0, 3] = True
    with torch.no_grad():
        inner = mixer.mixer(*mixer.qkv_proj(x).chunk(3, dim=-1), padding_mask=padding)
    assert torch.equal(inner[0, 3], torch.zeros(L, C, dtype=torch.float64))
    assert torch.equal(inner[0, :, 3], torch.zeros(L, C, dtype=torch.float64))
    assert (inner[1] != 0).any()


@pytest.mark.parametrize("start_node", [True, False])
@pytest.mark.parametrize("padded", [[0, 1], [3], [1, 4], [4, 5]], ids=["left", "interior", "two_gaps", "suffix"])
def test_padded_inputs_do_not_reach_unpadded_outputs(start_node, padded) -> None:
    """Outputs at valid pairs ignore padded values (random or NaN) and match the unpadded run."""
    mixer = make_mixer(start_node=start_node, dtype=torch.float64)
    x = torch.randn(B, L, L, C, dtype=torch.float64)
    padding = torch.zeros(B, L, dtype=torch.bool)
    padding[0, padded] = True
    pair = padding[:, :, None] | padding[:, None, :]
    with torch.no_grad():
        y = mixer(x, padding_mask=padding)
        for fill in (torch.randn_like(x), torch.full_like(x, float("nan"))):
            y2 = mixer(torch.where(pair[..., None], fill, x), padding_mask=padding)
            assert torch.isfinite(y2).all()
            assert torch.equal(y[~pair], y2[~pair])
        keep = (~padding[0]).nonzero().squeeze(1)
        if keep[-1] - keep[0] + 1 == len(keep):
            torch.testing.assert_close(y[0][keep][:, keep], mixer(x[:1, keep][:, :, keep])[0])
        assert not torch.equal(mixer(x), y)


def test_all_false_padding_mask_is_a_no_op() -> None:
    """An all-False padding mask matches passing no mask."""
    mixer = make_mixer()
    x = torch.randn(B, L, L, C)
    with torch.no_grad():
        assert torch.equal(mixer(x, padding_mask=torch.zeros(B, L, dtype=torch.bool)), mixer(x))


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def flat_filter_change_per_position(mask_cfg, seq_len=12, j=5):
    """|output change| along row 0 after perturbing (0, j), with a flat long filter."""
    mixer = make_mixer(mask_cfg=mask_cfg, seq_len=seq_len, dtype=torch.float64)
    mixer.mixer.short_conv_weight.data.zero_()
    mixer.mixer.short_conv_weight.data[-1] = 1.0
    kernel = mixer.mixer.global_conv.kernel
    kernel.forward = lambda seq_lens, conditioning=None: (
        torch.ones(1, 2 * seq_lens[0] - 1, C, dtype=torch.float64),
        torch.linspace(-1, 1, 2 * seq_lens[0] - 1).view(1, -1, 1),
    )
    x = torch.randn(1, seq_len, seq_len, C, dtype=torch.float64)
    x2 = x.clone()
    x2[0, 0, j] = torch.randn(C, dtype=torch.float64)
    with torch.no_grad():
        return (mixer(x2) - mixer(x))[0, 0].abs().amax(-1)


def test_box_window_limits_the_receptive_field() -> None:
    """Box window confines the receptive field to ``max_offset`` positions."""
    seq_len, j, max_offset = 12, 5, 3
    change = flat_filter_change_per_position(box_cfg(max_offset), seq_len, j)
    distance = (torch.arange(seq_len) - j).abs()
    assert change[distance <= max_offset].min() > 1e-3
    assert change[distance > max_offset].max() < 1e-6


def test_no_window_reaches_every_position() -> None:
    """Without a window a perturbation reaches every position in the row."""
    assert flat_filter_change_per_position(None).min() > 1e-3


def test_gaussian_window_is_learnable_and_local() -> None:
    """Gaussian window has a learnable width and decays away from the centre."""
    mixer = make_mixer(
        mask_cfg=LazyConfig(GaussianModulationND)(data_dim=1, num_channels=C), seq_len=12, dtype=torch.float64
    )
    mask = mixer.mixer.global_conv.mask
    assert mask.std_param.requires_grad
    kernel = torch.ones(1, 23, C, dtype=torch.float64)
    grid = torch.linspace(-1, 1, 23).view(1, -1, 1)
    with torch.no_grad():
        windowed = mask(grid=grid, x=kernel)[0, :, 0]
    assert windowed[11] == pytest.approx(1.0)
    assert windowed[0] < windowed[5] < windowed[11]


# ---------------------------------------------------------------------------
# Integration and FLOPs
# ---------------------------------------------------------------------------


def test_residual_block_integration() -> None:
    """Mixer runs inside a ``ResidualBlock`` and preserves the input shape."""
    torch.manual_seed(0)
    block = ResidualBlock(
        sequence_mixer_cfg=mixer_cfg(),
        sequence_mixer_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=C),
        condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
        condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
        mlp_cfg=LazyConfig(torch.nn.Identity)(),
        mlp_norm_cfg=LazyConfig(torch.nn.Identity)(),
        dropout_cfg=LazyConfig(torch.nn.Identity)(),
    )
    x = torch.randn(B, L, L, C)
    assert block(x, None).shape == x.shape


def test_flop_count() -> None:
    """Hand count for L=8, C=8, K=3 and the SIREN of ``mixer_cfg`` (16-wide, 2 hidden layers, 15 grid points)."""
    seq_len, taps, grid, width = 8, 3, 15, 16
    mixer = make_mixer(seq_len=seq_len)
    short_conv = 2 * 3 * C * taps * seq_len**2
    gates = 2 * C * seq_len**2
    siren = (2 * grid * width + grid * width) + 2 * (2 * grid * width * width + grid * width) + 2 * grid * width * C
    padded = 2 * seq_len
    long_conv = 3 * 5 * C * padded * 4 + 6 * C * padded + C * seq_len
    want = short_conv + gates + seq_len * (siren + long_conv) - (seq_len - 1) * siren
    assert want == 98736
    assert mixer.mixer.flop_count((seq_len, seq_len)) == want
    assert isinstance(mixer.flop_count((seq_len, seq_len)), int)
    assert mixer.flop_count((seq_len, seq_len)) > want


# ---------------------------------------------------------------------------
# Edge sizes and errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch, seq_len", [(1, 1), (1, 2), (2, 5), (3, 2)])
def test_edge_sizes(batch, seq_len) -> None:
    """Small batch and sequence sizes produce correctly shaped outputs and finite gradients."""
    mixer = make_mixer(seq_len=max(seq_len, 2), mask_cfg=box_cfg(8))
    x = torch.randn(batch, seq_len, seq_len, C, requires_grad=True)
    y = mixer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_non_square_pair_raises() -> None:
    """A non-square pair tensor raises ``ValueError``."""
    with pytest.raises(ValueError, match="square"):
        make_mixer()(torch.randn(B, L, L + 1, C))


def test_context_parallel_raises() -> None:
    """A context-parallel group raises ``ValueError``."""
    mixer = make_mixer()
    with pytest.raises(ValueError, match="context parallelism"):
        mixer(torch.randn(B, L, L, C), cp_group=FakeGroup())


def test_invalid_short_conv_backend_raises() -> None:
    """An unknown ``short_conv_backend`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="short_conv_backend"):
        instantiate(mixer_cfg(short_conv_backend="cudnn"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_causal": True},
        {"data_dim": 2},
        {"hidden_dim": 2 * C},
        {"grid_type": "single", "fft_padding": "circular"},
        {"grid_type": None, "fft_padding": ["circular"]},
        {"grid_type": None, "fft_padding": ["CIRCULAR"]},
    ],
    ids=["causal", "2d", "hidden-dim", "single-grid", "per-axis-circular", "per-axis-circular-upper"],
)
def test_invalid_global_conv_raises(overrides) -> None:
    """An unsupported ``global_conv`` configuration raises ``ValueError``."""
    cfg = mixer_cfg()
    for name, value in overrides.items():
        setattr(cfg.mixer_cfg.global_conv_cfg, name, value)
    with pytest.raises(ValueError, match="global_conv must be"):
        instantiate(cfg)


def test_conditioning_is_ignored_and_adaln_zero_block_runs() -> None:
    """Conditioning does not change the output, other kwargs raise, and the mixer runs in an ``AdaLNZeroResidualBlock``."""
    mixer = make_mixer()
    x = torch.randn(B, L, L, C)
    torch.testing.assert_close(mixer(x, conditioning=torch.randn(B, C)), mixer(x), rtol=0, atol=0)
    with pytest.raises(TypeError, match="padding_maks"):
        mixer(x, padding_maks=torch.zeros(B, L, dtype=torch.bool))
    block = AdaLNZeroResidualBlock(
        sequence_mixer_cfg=mixer_cfg(),
        sequence_mixer_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=C),
        mlp_cfg=LazyConfig(torch.nn.Identity)(),
        mlp_norm_cfg=LazyConfig(torch.nn.Identity)(),
        condition_norm_cfg=LazyConfig(torch.nn.Identity)(),
        dropout_cfg=LazyConfig(torch.nn.Identity)(),
        hidden_dim=C,
    )
    assert block(x, torch.randn(B, C)).shape == x.shape


# ---------------------------------------------------------------------------
# GPU: subq_ops backend
# ---------------------------------------------------------------------------


@_requires_cuda
@_requires_subq_ops
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [37, 256], ids=["fft", "explicit-eligible"])
def test_subq_ops_backend_matches_torch_fft(dtype, seq_len) -> None:
    """subq_ops FFT and short-conv backends match the torch backends with gradients."""
    hidden = 32
    reference = make_mixer(
        seq_len=seq_len, hidden_dim=hidden, mask_cfg=box_cfg(64), dtype=dtype, out_proj_bias=False
    ).cuda()
    fast = make_mixer(
        seq_len=seq_len,
        hidden_dim=hidden,
        mask_cfg=box_cfg(64),
        fft_backend="subq_ops",
        short_conv_backend="subq_ops",
        dtype=dtype,
        out_proj_bias=False,
    ).cuda()
    fast.load_state_dict(reference.state_dict())
    x = torch.randn(1, seq_len, seq_len, hidden, device="cuda", dtype=dtype)
    padding = torch.zeros(1, seq_len, dtype=torch.bool, device="cuda")
    padding[0, -3:] = True
    want = reference(x, padding_mask=padding)
    got = fast(x, padding_mask=padding)
    tol = 1e-5 if dtype == torch.float32 else 1e-3
    assert (got.float() - want.float()).norm() / want.float().norm() < tol
    got.square().sum().backward()
    assert fast.mixer.global_conv.kernel.out_linear.weight.grad is not None
    assert fast.mixer.short_conv_weight.grad is not None


@_requires_subq_ops
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="explicit_conv1d needs compute capability 10.x",
)
def test_bf16_subq_ops_backend_calls_explicit_conv1d(monkeypatch) -> None:
    """bfloat16 subq_ops backend dispatches to ``explicit_conv1d`` on compute capability 10.x."""
    from nvsubquadratic.ops import fftconv_custom

    explicit = fftconv_custom._get_explicit_conv1d()
    if explicit is None:
        pytest.skip("installed subquadratic_ops_torch has no explicit_conv1d")
    calls = []
    monkeypatch.setattr(fftconv_custom, "_explicit_conv1d", lambda x, k: calls.append(x.shape) or explicit(x, k))
    mixer = make_mixer(
        seq_len=256,
        hidden_dim=32,
        mask_cfg=box_cfg(64),
        fft_backend="subq_ops",
        short_conv_backend="subq_ops",
        dtype=torch.bfloat16,
    ).cuda()
    mixer(torch.randn(1, 256, 256, 32, device="cuda", dtype=torch.bfloat16))
    assert calls


@_requires_cuda
@_requires_subq_ops
def test_compiled_subq_ops_backends_match_eager() -> None:
    """``reduce-overhead`` compiled subq_ops mixer matches eager in output and input gradient."""
    seq_len, hidden = 256, 32
    mixer = make_mixer(
        seq_len=seq_len,
        hidden_dim=hidden,
        mask_cfg=box_cfg(64),
        fft_backend="subq_ops",
        short_conv_backend="subq_ops",
        dtype=torch.bfloat16,
        out_proj_bias=False,
    ).cuda()
    x = torch.randn(1, seq_len, seq_len, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    want = mixer(x)
    want_grad = torch.autograd.grad(want.float().square().sum(), x)[0]
    compiled = torch.compile(mixer, mode="reduce-overhead")
    for _ in range(3):
        got = compiled(x)
        got_grad = torch.autograd.grad(got.float().square().sum(), x)[0]
    assert (got.float() - want.float()).norm() / want.float().norm() < 5e-2
    assert (got_grad.float() - want_grad.float()).norm() / want_grad.float().norm() < 5e-2


@_requires_cuda
@_requires_subq_ops
@pytest.mark.parametrize("start_node", [True, False])
def test_fullgraph_compile_matches_eager(start_node) -> None:
    """``fullgraph`` compile matches eager and has no graph breaks."""
    seq_len, hidden = 256, 32
    mixer = make_mixer(
        seq_len=seq_len,
        hidden_dim=hidden,
        start_node=start_node,
        mask_cfg=box_cfg(64),
        fft_backend="subq_ops",
        short_conv_backend="subq_ops",
        dtype=torch.bfloat16,
        out_proj_bias=False,
    ).cuda()
    x = torch.randn(1, seq_len, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(mixer, fullgraph=True)
    with torch.no_grad():
        want = mixer(x)
        got = compiled(x)
    assert (got.float() - want.float()).norm() / want.float().norm() < 5e-2
    explanation = torch._dynamo.explain(mixer)(x)
    assert explanation.graph_break_count == 0
