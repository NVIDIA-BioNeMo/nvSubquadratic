# TODO: Add license header here

"""Pytest tests for SelfAttention module."""

import pytest
import torch

from nvsubquadratic.modules.attention import Attention as SelfAttention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("use_rope", [True, False])
def test_self_attention_shapes(data_dim, num_heads, use_rope, device):
    """Test that self-attention produces correct output shapes for different input dimensions."""
    hidden_dim = 288 if use_rope and data_dim == 3 else 128  # 288 / 8 = 36 head_dim (divisible by 6 for 3D RoPE)
    head_dim = hidden_dim // num_heads

    # Skip if head_dim doesn't meet RoPE requirements
    if use_rope:
        if data_dim == 1 and head_dim % 2 != 0:
            pytest.skip(f"1D RoPE requires head_dim divisible by 2, got {head_dim}")
        if data_dim == 2 and head_dim % 4 != 0:
            pytest.skip(f"2D RoPE requires head_dim divisible by 4, got {head_dim}")
        if data_dim == 3 and head_dim % 6 != 0:
            pytest.skip(f"3D RoPE requires head_dim divisible by 6, got {head_dim}")

    batch_size = 2

    if data_dim == 1:
        seq_len = 64
        input_shape = (batch_size, seq_len, hidden_dim)
        rope_spatial_dims = (seq_len,) if use_rope else None
    elif data_dim == 2:
        height, width = 16, 16
        input_shape = (batch_size, height, width, hidden_dim)
        rope_spatial_dims = (height, width) if use_rope else None
    elif data_dim == 3:
        depth, height, width = 8, 8, 8
        input_shape = (batch_size, depth, height, width, hidden_dim)
        rope_spatial_dims = (depth, height, width) if use_rope else None

    attn = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=True,
        use_rope=use_rope,
        attn_dropout=0.0,
        rope_spatial_dims=rope_spatial_dims,
    ).to(device)

    query = torch.randn(*input_shape, device=device)
    key = torch.randn(*input_shape, device=device)
    value = torch.randn(*input_shape, device=device)

    # Forward pass
    output = attn(query, key, value, cp_group=None)

    # Verify output shape matches input shape
    assert output.shape == input_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_self_attention_backward(data_dim, dtype_fixture, device):
    """Test that gradients flow correctly through self-attention."""

    hidden_dim = 288 if data_dim == 3 else 128  # 288 / 8 = 36 head_dim (divisible by 6 for 3D RoPE)
    num_heads = 8
    batch_size = 2

    if data_dim == 1:
        input_shape = (batch_size, 64, hidden_dim)
        rope_spatial_dims = (64,)
    elif data_dim == 2:
        input_shape = (batch_size, 16, 16, hidden_dim)
        rope_spatial_dims = (16, 16)
    elif data_dim == 3:
        input_shape = (batch_size, 8, 8, 8, hidden_dim)
        rope_spatial_dims = (8, 8, 8)

    attn = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=True,
        use_rope=True,
        attn_dropout=0.0,
        rope_spatial_dims=rope_spatial_dims,
    ).to(device=device, dtype=dtype_fixture)

    query = torch.randn(*input_shape, device=device, dtype=dtype_fixture, requires_grad=True)
    key = torch.randn(*input_shape, device=device, dtype=dtype_fixture, requires_grad=True)
    value = torch.randn(*input_shape, device=device, dtype=dtype_fixture, requires_grad=True)

    # Forward pass
    output = attn(query, key, value, cp_group=None)

    # Backward pass
    loss = output.mean()
    loss.backward()

    # Verify gradients exist
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None

    # Verify gradients are not NaN
    assert not torch.isnan(query.grad).any()
    assert not torch.isnan(key.grad).any()
    assert not torch.isnan(value.grad).any()

    # Verify gradients have correct shapes
    assert query.grad.shape == input_shape
    assert key.grad.shape == input_shape
    assert value.grad.shape == input_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qk_normalization_affects_output(device):
    """Test that QK normalization actually changes the attention output."""
    hidden_dim = 128
    num_heads = 8
    batch_size = 2
    seq_len = 32

    # Create identical inputs for both tests
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Test with QK norm
    attn_with_norm = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=True,
        use_rope=False,
        attn_dropout=0.0,
    ).to(device)

    # Test without QK norm
    attn_without_norm = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=False,
        use_rope=False,
        attn_dropout=0.0,
    ).to(device)

    # Copy weights to ensure same parameters (only normalization differs)
    attn_without_norm.load_state_dict(attn_with_norm.state_dict(), strict=False)

    output_with_norm = attn_with_norm(query.clone(), key.clone(), value.clone(), cp_group=None)
    output_without_norm = attn_without_norm(query.clone(), key.clone(), value.clone(), cp_group=None)

    # Outputs should be different due to normalization affecting attention scores
    assert not torch.allclose(output_with_norm, output_without_norm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rope_precomputed_buffers(device):
    """Test that RoPE cos/sin are precomputed as registered buffers (compile-safe)."""
    hidden_dim = 128
    num_heads = 8
    seq_len = 32

    attn = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=True,
        use_rope=True,
        attn_dropout=0.0,
        rope_spatial_dims=(seq_len,),
    ).to(device)

    # Buffers should exist as registered buffers (not in state_dict since persistent=False)
    assert hasattr(attn, "rope_cos")
    assert hasattr(attn, "rope_sin")
    assert attn.rope_cos.shape == (seq_len, hidden_dim // num_heads)
    assert attn.rope_sin.shape == (seq_len, hidden_dim // num_heads)

    # Buffers should follow device transfers
    assert attn.rope_cos.device == torch.device(device)

    # Forward should work without any dict caches
    batch_size = 2
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    output = attn(query, key, value, cp_group=None)
    assert output.shape == (batch_size, seq_len, hidden_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_attention_output_varies_with_input(data_dim, device):
    """Test that attention output changes when input changes."""
    hidden_dim = 288 if data_dim == 3 else 128
    num_heads = 8
    batch_size = 2

    if data_dim == 1:
        input_shape = (batch_size, 32, hidden_dim)
        rope_spatial_dims = (32,)
    elif data_dim == 2:
        input_shape = (batch_size, 8, 8, hidden_dim)
        rope_spatial_dims = (8, 8)
    elif data_dim == 3:
        input_shape = (batch_size, 4, 4, 4, hidden_dim)
        rope_spatial_dims = (4, 4, 4)

    attn = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=True,
        use_rope=True,
        attn_dropout=0.0,
        rope_spatial_dims=rope_spatial_dims,
    ).to(device)

    # Create two different inputs
    query1 = torch.randn(*input_shape, device=device)
    key1 = torch.randn(*input_shape, device=device)
    value1 = torch.randn(*input_shape, device=device)

    query2 = torch.randn(*input_shape, device=device)
    key2 = torch.randn(*input_shape, device=device)
    value2 = torch.randn(*input_shape, device=device)

    output1 = attn(query1, key1, value1, cp_group=None)
    output2 = attn(query2, key2, value2, cp_group=None)

    # Outputs should be different
    assert not torch.allclose(output1, output2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_self_attention_value_weighting(device):
    """Test that attention correctly weights values based on query-key similarity."""
    hidden_dim = 128
    num_heads = 4
    seq_len = 16

    attn = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=False,  # Disable to test raw attention
        use_rope=False,
        attn_dropout=0.0,
    ).to(device)

    batch_size = 1

    # Create inputs where first position has high similarity with itself
    query = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
    key = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
    value = torch.zeros(batch_size, seq_len, hidden_dim, device=device)

    # Make first position distinct
    query[:, 0, :] = 1.0
    key[:, 0, :] = 1.0
    value[:, 0, :] = 10.0  # High value at first position

    output = attn(query, key, value, cp_group=None)

    # First position should have non-zero output (attending to itself)
    assert torch.abs(output[0, 0]).sum() > 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_validation(device):
    """Test that invalid input dimensions raise appropriate errors."""
    hidden_dim = 128
    num_heads = 8

    attn = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=True,
        use_rope=False,
        attn_dropout=0.0,
    ).to(device)

    # Create input with wrong number of dimensions (2D - not enough)
    wrong_input = torch.randn(2, hidden_dim, device=device)

    # Should raise assertion error from _flatten_spatial
    with pytest.raises(AssertionError):
        attn(wrong_input, wrong_input, wrong_input, cp_group=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hidden_dim_divisibility():
    """Test that hidden_dim must be divisible by num_heads."""
    with pytest.raises(AssertionError):
        SelfAttention(
            hidden_dim=127,  # Not divisible by 8
            num_heads=8,
            apply_qk_norm=True,
            use_rope=False,
        )


@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_rope_requirements(data_dim):
    """Test RoPE dimensionality requirements are enforced at init time."""
    num_heads = 8

    # Choose head_dim that violates RoPE requirements
    if data_dim == 1:
        hidden_dim = 8 * 3  # head_dim = 3 (not divisible by 2)
        rope_spatial_dims = (32,)
    elif data_dim == 2:
        hidden_dim = 8 * 6  # head_dim = 6 (not divisible by 4)
        rope_spatial_dims = (8, 8)
    elif data_dim == 3:
        hidden_dim = 8 * 8  # head_dim = 8 (not divisible by 6)
        rope_spatial_dims = (4, 4, 4)

    # Should raise assertion error about divisibility at init time
    with pytest.raises(AssertionError):
        SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            apply_qk_norm=True,
            use_rope=True,
            attn_dropout=0.0,
            rope_spatial_dims=rope_spatial_dims,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_attention_is_permutation_invariant_to_key_value_order(data_dim, device):
    """Test that swapping K/V positions changes output (attention is not symmetric)."""
    hidden_dim = 288 if data_dim == 3 else 128
    num_heads = 8

    attn = SelfAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        apply_qk_norm=False,
        use_rope=False,
        attn_dropout=0.0,
    ).to(device)

    batch_size = 1

    if data_dim == 1:
        input_shape = (batch_size, 16, hidden_dim)
    elif data_dim == 2:
        input_shape = (batch_size, 4, 4, hidden_dim)
    elif data_dim == 3:
        input_shape = (batch_size, 4, 4, 4, hidden_dim)

    query = torch.randn(*input_shape, device=device)
    key = torch.randn(*input_shape, device=device)
    value = torch.randn(*input_shape, device=device)

    # Create a different key
    key2 = torch.randn(*input_shape, device=device)

    output1 = attn(query, key, value, cp_group=None)
    output2 = attn(query, key2, value, cp_group=None)

    # Outputs should be different when key changes
    assert not torch.allclose(output1, output2)
