# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-equivalence tests: refactored JiT components vs the reference JiT.

The refactor in :mod:`nvsubquadratic.networks.jit` replaces two locally
defined modules with their already-existing project counterparts:

- ``RMSNorm`` (was in ``jit_utils.py``)  ->  :class:`nvsubquadratic.modules.rms_norm.RMSNorm`
- ``SwiGLUFFN`` (was in ``jit.py``)      ->  :class:`nvsubquadratic.modules.mlp.MLP` (``activation="swiglu"``)

This test file pins the equivalence with **inline copies of the original
JiT code** (taken verbatim from ``~/projects/JiT/util/model_util.py`` and
``~/projects/JiT/model_jit.py``) so the byte-level match is checked
independently of any future drift in either the project's modules or the
JiT reference repo.

A high-level smoke test of the full refactored JiT network is included too
(forward shape only) so the refactor catches any structural regression.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsubquadratic.modules.rms_norm import RMSNorm as ProjectRMSNorm
from nvsubquadratic.networks.jit import JiTBlock, _make_swiglu_mlp


# =============================================================================
# Reference implementations — verbatim copy from the JiT repo
# =============================================================================
#
# These are the original JiT classes inlined here so the test is self-contained
# and the equivalence check does not depend on having the JiT repo cloned.  If
# the upstream JiT repo changes these definitions, update both halves and
# re-run.


class JiTReferenceRMSNorm(nn.Module):
    """Verbatim copy of ``JiT/util/model_util.py::RMSNorm`` (LTH14, 2025)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class JiTReferenceSwiGLUFFN(nn.Module):
    """Verbatim copy of ``JiT/model_jit.py::SwiGLUFFN`` (LTH14, 2025)."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


# =============================================================================
# RMSNorm equivalence
# =============================================================================


@pytest.mark.parametrize("dim", [64, 768, 1024, 1280])  # JiT head_dim + B/L/H widths
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rmsnorm_equivalence_fp(dim: int, dtype: torch.dtype) -> None:
    """Project's RMSNorm (PyTorch backend) matches JiT's RMSNorm exactly.

    JiT's implementation always upcasts to fp32 inside the forward; the
    project's PyTorch path does the same, so for fp32 / fp64 inputs the
    results are byte-identical (within standard float associativity).
    """
    torch.manual_seed(0)
    project = ProjectRMSNorm(dim=dim, eps=1e-6, use_quack=False)
    reference = JiTReferenceRMSNorm(hidden_size=dim, eps=1e-6)

    # Match weights exactly.
    with torch.no_grad():
        reference.weight.copy_(project.weight)

    x = torch.randn(4, 32, dim, dtype=dtype)
    y_project = project(x)
    y_reference = reference(x)

    assert y_project.shape == y_reference.shape
    assert y_project.dtype == y_reference.dtype == dtype
    assert torch.equal(y_project, y_reference), (
        f"RMSNorm output mismatch (dim={dim}, dtype={dtype}): "
        f"max abs diff = {(y_project - y_reference).abs().max().item()}"
    )


def test_rmsnorm_equivalence_bf16() -> None:
    """RMSNorm equivalence under bf16: both paths cast to fp32 internally.

    The output is finally downcast back to bf16, so we expect identical
    bf16 outputs after the round-trip.
    """
    torch.manual_seed(0)
    dim = 768
    project = ProjectRMSNorm(dim=dim, eps=1e-6, use_quack=False)
    reference = JiTReferenceRMSNorm(hidden_size=dim, eps=1e-6)
    with torch.no_grad():
        reference.weight.copy_(project.weight)

    x = torch.randn(4, 32, dim, dtype=torch.bfloat16)
    y_project = project(x)
    y_reference = reference(x)
    assert y_project.dtype == torch.bfloat16
    assert torch.equal(y_project, y_reference)


def test_rmsnorm_qk_norm_shape() -> None:
    """JiT applies RMSNorm to per-head Q/K tensors of shape [B, H, N, D].

    The reference call site is::

        self.q_norm = RMSNorm(head_dim)  # operates on last-dim only

    so the project's RMSNorm must work on 4-D inputs too (it normalises
    over the last dim, which is what we need).
    """
    torch.manual_seed(0)
    head_dim = 64
    project = ProjectRMSNorm(dim=head_dim, eps=1e-6, use_quack=False)
    reference = JiTReferenceRMSNorm(hidden_size=head_dim, eps=1e-6)
    with torch.no_grad():
        reference.weight.copy_(project.weight)

    q = torch.randn(2, 12, 256, head_dim)  # [B, num_heads, num_tokens, head_dim]
    assert torch.equal(project(q), reference(q))


def test_rmsnorm_weight_tagged_no_weight_decay() -> None:
    """The project's RMSNorm tags ``weight._no_weight_decay`` for the optimizer.

    This is one of the *advantages* of using the project's RMSNorm: the
    optimizer's parameter-grouping helper picks up the flag automatically,
    matching JiT's ``add_weight_decay`` rule (1-d params go to no-decay).
    """
    norm = ProjectRMSNorm(dim=64, eps=1e-6, use_quack=False)
    assert getattr(norm.weight, "_no_weight_decay", False) is True


# =============================================================================
# SwiGLU FFN equivalence
# =============================================================================


@pytest.mark.parametrize(
    ("hidden_size", "mlp_ratio"),
    [
        (768, 4.0),  # JiT-B
        (1024, 4.0),  # JiT-L
        (1280, 4.0),  # JiT-H
        (384, 4.0),  # smaller hidden, to exercise the rounding edge
        (513, 4.0),  # odd hidden (rare but ensures no even-division assumptions)
    ],
)
def test_swiglu_inner_width_matches_reference(hidden_size: int, mlp_ratio: float) -> None:
    """``_make_swiglu_mlp`` and JiT's ``SwiGLUFFN`` agree on the inner width.

    JiT: ``int(int(hidden_size * mlp_ratio) * 2/3)``
    Project: ``int(hidden_size * (mlp_ratio * 2/3))``

    The test pins the equivalence numerically across the JiT model sizes.
    """
    mlp_outer = int(hidden_size * mlp_ratio)
    expected = int(mlp_outer * 2 / 3)

    project_mlp = _make_swiglu_mlp(hidden_size=hidden_size, mlp_ratio=mlp_ratio, drop=0.0)
    reference = JiTReferenceSwiGLUFFN(dim=hidden_size, hidden_dim=mlp_outer, drop=0.0)

    assert project_mlp.hidden_dim == expected, (
        f"project inner width mismatch: got {project_mlp.hidden_dim}, expected {expected}"
    )
    # Reference stores the inner width on w3's in_features.
    assert reference.w3.in_features == expected
    assert project_mlp.layer2.in_features == expected
    assert project_mlp.layer1.out_features == 2 * expected
    assert reference.w12.out_features == 2 * expected


@pytest.mark.parametrize("hidden_size", [768, 1024])
def test_swiglu_forward_equivalence(hidden_size: int) -> None:
    """Project's MLP(swiglu) gives bit-identical output to JiT's SwiGLUFFN.

    We construct both modules, copy the JiT weights into the project's MLP
    (``w12 -> layer1``, ``w3 -> layer2``) — matching the chunk semantics
    used in both forward passes — and verify exact output equality on a
    deterministic input.
    """
    torch.manual_seed(0)
    mlp_ratio = 4.0
    drop = 0.0

    project_mlp = _make_swiglu_mlp(hidden_size=hidden_size, mlp_ratio=mlp_ratio, drop=drop)
    reference = JiTReferenceSwiGLUFFN(
        dim=hidden_size,
        hidden_dim=int(hidden_size * mlp_ratio),
        drop=drop,
        bias=True,
    )

    # Both modules expect ``[first half = gate (silu arg), second half = value]``
    # after the chunk(2, dim=-1) split, so the weights map straight across.
    with torch.no_grad():
        project_mlp.layer1.weight.copy_(reference.w12.weight)
        project_mlp.layer1.bias.copy_(reference.w12.bias)
        project_mlp.layer2.weight.copy_(reference.w3.weight)
        project_mlp.layer2.bias.copy_(reference.w3.bias)

    project_mlp.eval()
    reference.eval()
    x = torch.randn(2, 16, hidden_size)
    y_project = project_mlp(x)
    y_reference = reference(x)

    assert y_project.shape == y_reference.shape
    assert torch.equal(y_project, y_reference), (
        f"SwiGLU output mismatch (hidden_size={hidden_size}): "
        f"max abs diff = {(y_project - y_reference).abs().max().item()}"
    )


def test_swiglu_dropout_position_matches_reference() -> None:
    """Dropout is applied between activation and final linear in both impls.

    JiT:  ``w3(ffn_dropout(silu(x1) * x2))``
    Project's MLP:  ``layer2(dropout(activation(layer1(x))))``

    We test this by running both with a non-zero dropout and the same
    seeded RNG state; the outputs should be bit-identical when the same
    pseudo-random mask is generated.
    """
    torch.manual_seed(0)
    hidden_size = 384
    mlp_ratio = 4.0
    drop = 0.5

    project_mlp = _make_swiglu_mlp(hidden_size=hidden_size, mlp_ratio=mlp_ratio, drop=drop)
    reference = JiTReferenceSwiGLUFFN(
        dim=hidden_size,
        hidden_dim=int(hidden_size * mlp_ratio),
        drop=drop,
        bias=True,
    )
    with torch.no_grad():
        project_mlp.layer1.weight.copy_(reference.w12.weight)
        project_mlp.layer1.bias.copy_(reference.w12.bias)
        project_mlp.layer2.weight.copy_(reference.w3.weight)
        project_mlp.layer2.bias.copy_(reference.w3.bias)

    project_mlp.train()
    reference.train()
    x = torch.randn(2, 8, hidden_size)

    torch.manual_seed(123)
    y_project = project_mlp(x)
    torch.manual_seed(123)
    y_reference = reference(x)

    assert torch.equal(y_project, y_reference), (
        f"SwiGLU+dropout mismatch: max abs diff = {(y_project - y_reference).abs().max().item()}"
    )


# =============================================================================
# JiTBlock smoke test — confirms the refactor doesn't break the block forward
# =============================================================================


def test_jit_block_forward_shape() -> None:
    """A single JiTBlock preserves (B, N, hidden_size) through the forward.

    Uses tiny dimensions so the test runs on CPU in seconds.  The token
    sequence length must equal ``pt_seq_len ** 2`` so the precomputed 2D
    RoPE table aligns.
    """
    torch.manual_seed(0)
    hidden_size = 96
    num_heads = 6  # head_dim = 16
    pt_seq_len = 4
    seq_len = pt_seq_len * pt_seq_len  # 16 — must equal pt_seq_len ** 2 for the RoPE table

    block = JiTBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=4.0)
    block.eval()

    # Build the same RoPE the JiT network would: half_head_dim = hidden//heads//2.
    from nvsubquadratic.networks.jit_utils import VisionRotaryEmbeddingFast

    half_head_dim = hidden_size // num_heads // 2
    rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=pt_seq_len, num_cls_token=0)

    x = torch.randn(2, seq_len, hidden_size)
    cond = torch.randn(2, hidden_size)
    y = block(x, cond, feat_rope=rope)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_jit_block_adaln_zero_at_init() -> None:
    """At init the adaLN modulation produces gate=0, so the block is identity.

    JiT's ``initialize_weights`` zero-inits the last layer of
    ``adaLN_modulation`` so the gate is exactly 0 and both residual
    branches contribute nothing.  We replicate that here on a single
    block to confirm the refactor preserves the property.
    """
    torch.manual_seed(0)
    hidden_size = 96
    num_heads = 6
    pt_seq_len = 4
    seq_len = pt_seq_len * pt_seq_len  # 16

    block = JiTBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=4.0)
    # JiT's init zero-outs the AdaLN final linear (weight + bias).
    nn.init.constant_(block.adaLN_modulation[-1].weight, 0.0)
    nn.init.constant_(block.adaLN_modulation[-1].bias, 0.0)
    block.eval()

    from nvsubquadratic.networks.jit_utils import VisionRotaryEmbeddingFast

    half_head_dim = hidden_size // num_heads // 2
    rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=pt_seq_len, num_cls_token=0)

    x = torch.randn(2, seq_len, hidden_size)
    cond = torch.randn(2, hidden_size)
    y = block(x, cond, feat_rope=rope)

    # Zero-init AdaLN -> gate_msa = gate_mlp = 0 -> y == x.
    assert torch.allclose(y, x, atol=0.0, rtol=0.0), (
        f"AdaLN-zero init violated: max diff = {(y - x).abs().max().item()}"
    )


# =============================================================================
# Full network smoke test — confirms the refactor wires together end-to-end
# =============================================================================


def test_jit_network_forward_smoke() -> None:
    """Full ``JiT_B_16`` instantiates and forwards on a 256x256 dummy input.

    Uses ``input_size=64, patch_size=16`` to keep the test fast on CPU.
    """
    from nvsubquadratic.networks.jit import JiT

    torch.manual_seed(0)
    net = JiT(
        input_size=64,
        patch_size=16,
        in_channels=3,
        hidden_size=96,
        depth=2,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=10,
        bottleneck_dim=32,
        in_context_len=4,
        in_context_start=1,
    )
    net.eval()

    x = torch.randn(2, 64, 64, 3)  # BHWC, channels-last per wrapper
    condition = torch.randn(2, 96)  # t_emb + y_emb
    class_emb = torch.randn(2, 96)  # raw label emb for in-context tokens

    out = net({"input": x, "condition": condition, "class_emb": class_emb})

    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == x.shape, f"Expected {x.shape}, got {out['logits'].shape}"
    assert torch.isfinite(out["logits"]).all()
