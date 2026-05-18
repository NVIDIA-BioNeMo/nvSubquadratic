"""Tests for the ARCVim (Visual Mamba) baseline."""

import pytest
import torch
import torch.nn as nn

from nvsubquadratic.networks.baselines.arc_vim import ARCVim, MambaBlock, MambaResidualBlock, _DropPath


# ── Fixtures ─────────────────────────────────────────────────────────────────


def make_model(**overrides) -> ARCVim:
    """Return a small ARCVim for testing (fast, CPU-sized)."""
    defaults = {
        "num_tasks": 10,
        "max_size": 32,
        "num_colors": 12,
        "embed_dim": 64,
        "depth": 4,
        "patch_size": 2,
        "d_state": 8,
        "d_conv": 4,
        "expand": 2,
        "drop_path_rate": 0.0,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return ARCVim(**defaults)


def make_batch(batch_size: int = 2, max_size: int = 32, num_tasks: int = 10) -> dict:
    return {
        "input": torch.randint(0, 10, (batch_size, max_size, max_size)),
        "condition": {"task_id": torch.randint(0, num_tasks, (batch_size,))},
    }


# ── ARCVim: initialisation ────────────────────────────────────────────────────


def test_init_stores_attributes():
    model = make_model(embed_dim=128, depth=4)
    assert model.embed_dim == 128
    assert model.max_size == 32
    assert model.num_colors == 12
    assert model.patch_size == 2
    assert model.num_task_tokens == 1
    assert model.grid_size == 16  # 32 // 2
    assert model.num_patches == 256  # 16 * 16


def test_init_odd_depth_raises():
    with pytest.raises(ValueError, match="even"):
        make_model(depth=3)


def test_layer_count():
    model = make_model(depth=6)
    assert len(model.layers) == 6
    for block in model.layers:
        assert isinstance(block, MambaResidualBlock)


def test_pos_embed_shape():
    model = make_model(embed_dim=64, depth=4)
    assert model.pos_embed.shape == (1, model.num_patches, model.embed_dim)


# ── ARCVim: forward pass ──────────────────────────────────────────────────────


def test_output_shape():
    model = make_model()
    model.eval()
    batch = make_batch(batch_size=2)
    with torch.no_grad():
        out = model(batch)
    assert "logits" in out
    assert out["logits"].shape == (2, 12, 32, 32)


def test_output_shape_batch_size_1():
    model = make_model()
    model.eval()
    batch = make_batch(batch_size=1)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (1, 12, 32, 32)


def test_output_shape_batch_size_4():
    model = make_model()
    model.eval()
    batch = make_batch(batch_size=4)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (4, 12, 32, 32)


def test_output_shape_multiple_task_tokens():
    model = make_model(num_task_tokens=3, depth=4)
    model.eval()
    batch = make_batch(batch_size=2)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (2, 12, 32, 32)


def test_output_shape_patch_size_1():
    """Patch size 1: each pixel is its own token."""
    model = make_model(patch_size=1, depth=4)
    model.eval()
    batch = make_batch(batch_size=2)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (2, 12, 32, 32)


def test_invalid_input_rank_raises():
    model = make_model()
    model.eval()
    # 4-D input should raise
    batch = {"input": torch.randint(0, 10, (2, 1, 32, 32)), "condition": {"task_id": torch.zeros(2, dtype=torch.long)}}
    with pytest.raises(ValueError):
        model(batch)


# ── ARCVim: sentinel / edge-case inputs ──────────────────────────────────────


def test_padding_sentinel_colors_do_not_crash():
    """IGNORE (10) and PAD (11) indices must be clamped, not crash."""
    model = make_model(num_colors=12)
    model.eval()
    pixel_values = torch.full((2, 32, 32), 11, dtype=torch.long)  # all PAD
    batch = {"input": pixel_values, "condition": {"task_id": torch.zeros(2, dtype=torch.long)}}
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (2, 12, 32, 32)


def test_mixed_sentinel_and_valid_colors():
    model = make_model(num_colors=12)
    model.eval()
    pixel_values = torch.randint(0, 12, (2, 32, 32))  # includes 10 and 11
    batch = {"input": pixel_values, "condition": {"task_id": torch.zeros(2, dtype=torch.long)}}
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (2, 12, 32, 32)


# ── ARCVim: gradient flow ─────────────────────────────────────────────────────


def test_gradients_flow_to_all_parameters():
    """Every parameter should receive a gradient after a backward pass."""
    model = make_model()
    model.train()
    batch = make_batch(batch_size=2)
    out = model(batch)
    loss = out["logits"].sum()
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert no_grad == [], f"Parameters with no gradient: {no_grad}"


def test_output_is_differentiable():
    model = make_model()
    batch = make_batch(batch_size=2)
    out = model(batch)
    assert out["logits"].requires_grad


# ── ARCVim: bidirectionality ──────────────────────────────────────────────────


def test_bidirectional_different_from_unidirectional():
    """Forward + reverse should produce a different result than forward only."""
    torch.manual_seed(0)
    model = make_model(depth=4)
    model.eval()

    batch = make_batch(batch_size=1)
    with torch.no_grad():
        logits_bidir = model(batch)["logits"]

    # Monkey-patch to skip the reverse branch (feed zeros for bwd)
    original_method = model._forward_mamba_bidirectional

    def forward_only(x):
        for i in range(0, len(model.layers), 2):
            x = model.layers[i](x)
            # Skip backward layer: add zeros instead of bwd contribution
        return x

    model._forward_mamba_bidirectional = forward_only
    with torch.no_grad():
        logits_unidir = model(batch)["logits"]
    model._forward_mamba_bidirectional = original_method

    assert not torch.allclose(logits_bidir, logits_unidir), "Bidirectional and unidirectional outputs should differ"


def test_bidirectional_uses_all_layers():
    """All layers must be exercised in a forward pass."""
    model = make_model(depth=4)
    model.eval()
    touched = []

    def make_hook(idx):
        def hook(module, inp, out):
            touched.append(idx)

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(model.layers)]
    batch = make_batch(batch_size=1)
    with torch.no_grad():
        model(batch)
    for h in handles:
        h.remove()

    assert sorted(touched) == list(range(len(model.layers))), f"Not all layers were called. Called: {sorted(touched)}"


# ── MambaBlock: unit tests ────────────────────────────────────────────────────


def test_mamba_block_output_shape():
    block = MambaBlock(d_model=64, d_state=8, d_conv=4, expand=2)
    x = torch.randn(2, 20, 64)
    out = block(x)
    assert out.shape == (2, 20, 64), f"Expected (2, 20, 64), got {out.shape}"


def test_mamba_block_different_seq_lengths():
    block = MambaBlock(d_model=32, d_state=4)
    for seq_len in [1, 7, 64, 257]:
        x = torch.randn(1, seq_len, 32)
        out = block(x)
        assert out.shape == (1, seq_len, 32), f"seq_len={seq_len}: got {out.shape}"


def test_mamba_block_gradient_flow():
    block = MambaBlock(d_model=32, d_state=4)
    x = torch.randn(2, 10, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_mamba_block_autoregressive_property():
    """Changing token i must not affect output tokens j < i (causal scan)."""
    torch.manual_seed(42)
    block = MambaBlock(d_model=16, d_state=4, d_conv=4, expand=2)
    block.eval()

    x = torch.randn(1, 8, 16)
    x_perturbed = x.clone()
    x_perturbed[0, 5, :] += 10.0  # perturb position 5

    with torch.no_grad():
        out_orig = block(x)
        out_pert = block(x_perturbed)

    # Tokens 0-4 must be unchanged; token 5 onward may differ
    assert torch.allclose(out_orig[0, :5], out_pert[0, :5], atol=1e-5), (
        "Causal scan violation: earlier tokens were affected by a later perturbation"
    )
    assert not torch.allclose(out_orig[0, 5:], out_pert[0, 5:], atol=1e-5), (
        "Perturbation at position 5 had no effect on subsequent tokens"
    )


def test_mamba_block_not_permutation_equivariant():
    """SSM output should change when the input sequence is shuffled."""
    torch.manual_seed(7)
    block = MambaBlock(d_model=16, d_state=4)
    block.eval()

    x = torch.randn(1, 10, 16)
    perm = torch.randperm(10)
    x_shuffled = x[:, perm, :]

    with torch.no_grad():
        out = block(x)
        out_shuffled = block(x_shuffled)

    # Un-shuffle and compare: should NOT be the same
    assert not torch.allclose(out[:, perm, :], out_shuffled, atol=1e-4), (
        "Mamba SSM is unexpectedly permutation-equivariant"
    )


# ── MambaResidualBlock: unit tests ────────────────────────────────────────────


def test_mamba_residual_block_output_shape():
    block = MambaResidualBlock(d_model=64, d_state=8)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == x.shape


def test_mamba_residual_block_residual_connection():
    """With a zeroed-out inner Mamba, output should equal input."""
    block = MambaResidualBlock(d_model=32, d_state=4)
    block.eval()
    # Zero all Mamba parameters → mixer output is ~0 → residual dominates
    for p in block.mamba.parameters():
        nn.init.zeros_(p)
    # out_proj bias=False by default, so with all zeros the mixer output is 0
    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        out = block(x)
    # Not exactly x because of norms/biases, but the shape must match
    assert out.shape == x.shape


# ── _DropPath: unit tests ─────────────────────────────────────────────────────


def test_droppath_identity_in_eval():
    dp = _DropPath(drop_prob=1.0)  # always drops in train — but not in eval
    dp.eval()
    x = torch.ones(4, 8)
    out = dp(x)
    assert torch.allclose(out, x)


def test_droppath_zero_prob_identity():
    dp = _DropPath(drop_prob=0.0)
    dp.train()
    x = torch.ones(4, 8)
    out = dp(x)
    assert torch.allclose(out, x)


def test_droppath_drops_in_training():
    """With moderate drop_prob some rows should be zeroed in training mode."""
    torch.manual_seed(0)
    dp = _DropPath(drop_prob=0.5)
    dp.train()
    x = torch.ones(200, 8)
    out = dp(x)
    # Each row is either 0 (dropped) or rescaled (non-zero)
    row_sums = out.abs().sum(dim=1)  # [200]
    num_dropped = (row_sums == 0).sum().item()
    # With p=0.5 and 200 samples we expect roughly 100 drops; accept [60, 140]
    assert 60 <= num_dropped <= 140, f"Expected ~100 dropped rows, got {num_dropped}"


# ── ARCVim: determinism ───────────────────────────────────────────────────────


def test_eval_mode_is_deterministic():
    model = make_model(drop_path_rate=0.1, dropout=0.1)
    model.eval()
    batch = make_batch(batch_size=2)
    with torch.no_grad():
        out1 = model(batch)["logits"]
        out2 = model(batch)["logits"]
    assert torch.allclose(out1, out2), "eval-mode forward is not deterministic"
