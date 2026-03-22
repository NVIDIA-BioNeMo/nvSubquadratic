# TODO: Add license header here

"""Tests for causality in Attention and Mamba modules.

Causality Test Principle
------------------------
A causal model ensures that output at position i only depends on inputs at positions 0, 1, ..., i.
Modifying input at position j > i should NOT change output at position i.

These tests verify causality for:
- **Attention**: `is_causal=True` enables causal masking
- **Mamba**: `bidirectional=False` ensures unidirectional (causal) SSM processing

Test Summary
------------
+----+---------------------------------------------+-------------------+------------------------------------------+
| #  | Test                                        | Module            | What it verifies                         |
+----+---------------------------------------------+-------------------+------------------------------------------+
| **Operator Tests** - Test raw Attention/Mamba operators directly                                                |
+----+---------------------------------------------+-------------------+------------------------------------------+
| 1  | test_causal_attention_future_independence   | Attention         | Last position doesn't affect earlier     |
| 2  | test_causal_attention_middle_position       | Attention         | Position N doesn't affect 0..N-1         |
| 3  | test_non_causal_attention_sees_future       | Attention         | Position 0 IS affected by future         |
| 4  | test_causal_attention_triangular_dependency | Attention         | Full triangular dependency structure     |
| 5  | test_causal_mamba_future_independence       | MambaNDMixer      | Last position doesn't affect earlier     |
| 6  | test_causal_mamba_middle_position           | MambaNDMixer      | Position N doesn't affect 0..N-1         |
| 7  | test_bidirectional_mamba_sees_future        | MambaNDMixer      | Position 0 IS affected by future         |
| 8  | test_causal_mamba_triangular_dependency     | MambaNDMixer      | Triangular for positions 4, 8, 12        |
+----+---------------------------------------------+-------------------+------------------------------------------+
| **Operator Gradient Tests** - Verify gradients don't backprop from past to future                               |
+----+---------------------------------------------+-------------------+------------------------------------------+
| 9  | test_attention_causal_gradient_flow         | Attention         | No grad flow from pos 0 -> future        |
| 10 | test_attention_non_causal_gradient_flow     | Attention         | Grad DOES flow to all positions          |
| 11 | test_mamba_causal_gradient_flow             | MambaNDMixer      | No grad flow from pos 0 -> future        |
| 12 | test_mamba_bidirectional_gradient_flow      | MambaNDMixer      | Grad DOES flow to all positions          |
+----+---------------------------------------------+-------------------+------------------------------------------+
| **Mixer Tests** - Test full QKVSequenceMixer/MambaNDMixer stack (includes projections)                          |
+----+---------------------------------------------+-------------------+------------------------------------------+
| 13 | test_causal_mixer_future_independence       | QKVSequenceMixer  | Last position doesn't affect earlier     |
| 14 | test_causal_mixer_middle_position           | QKVSequenceMixer  | Position N doesn't affect 0..N-1         |
| 15 | test_non_causal_mixer_sees_future           | QKVSequenceMixer  | Position 0 IS affected by future         |
| 16 | test_causal_mixer_gradient_flow             | QKVSequenceMixer  | No grad flow from pos 0 -> future        |
| 17 | test_causal_mixer_future_independence       | MambaNDMixer      | Last position doesn't affect earlier     |
| 18 | test_causal_mixer_middle_position           | MambaNDMixer      | Position N doesn't affect 0..N-1         |
| 19 | test_bidirectional_mixer_sees_future        | MambaNDMixer      | Position 0 IS affected by future         |
| 20 | test_causal_mixer_gradient_flow             | MambaNDMixer      | No grad flow from pos 0 -> future        |
+----+---------------------------------------------+-------------------+------------------------------------------+

Usage
-----
Run all tests (requires GPU for Mamba tests):
    pytest tests/test_causality_attn_mamba.py -v -o "addopts="

Run only Attention tests (no GPU required):
    pytest tests/test_causality_attn_mamba.py -v -o "addopts=" -k "Attention"

Notes
-----
- Mamba tests are skipped automatically if CUDA is not available
- The `mamba_ssm` import warnings are expected (optional GPU dependency)
"""

import pytest
import torch


# Check if mamba_ssm is available and functional
try:
    from mamba_ssm import Mamba2  # noqa: F401

    # Test that the CUDA kernels actually work (not just import)
    _test_tensor = torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu")
    MAMBA_AVAILABLE = torch.cuda.is_available()
except (ImportError, Exception):
    MAMBA_AVAILABLE = False

requires_mamba = pytest.mark.skipif(
    not MAMBA_AVAILABLE,
    reason="mamba_ssm not installed or CUDA not available",
)


################################################################################
# Attention Causality Tests (Forward + Gradient Flow)
################################################################################


class TestAttentionCausality:
    """Tests for Attention causality."""

    @pytest.fixture
    def attention_causal(self):
        """Create a causal Attention module."""
        from nvsubquadratic.modules.attention import Attention

        return Attention(
            hidden_dim=64,
            num_heads=4,
            apply_qk_norm=False,
            use_rope=True,
            is_causal=True,
            attn_dropout=0.0,
        )

    @pytest.fixture
    def attention_non_causal(self):
        """Create a non-causal Attention module."""
        from nvsubquadratic.modules.attention import Attention

        return Attention(
            hidden_dim=64,
            num_heads=4,
            apply_qk_norm=False,
            use_rope=True,
            is_causal=False,
            attn_dropout=0.0,
        )

    def test_causal_attention_future_independence(self, attention_causal):
        """Test that causal attention output at position i is independent of input at position j > i."""
        attention_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 32
        hidden_dim = 64

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Forward pass with original input
        with torch.no_grad():
            out_original = attention_causal(x, x, x)

        # Modify input at the LAST position (should not affect earlier positions)
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            out_modified = attention_causal(x_modified, x_modified, x_modified)

        # All positions EXCEPT the last should be identical
        # (causal: position i doesn't see position i+1, ..., seq_len-1)
        assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-5), (
            "Causal attention: modifying future input should not affect past outputs"
        )

    def test_causal_attention_middle_position(self, attention_causal):
        """Test causality by modifying a middle position."""
        attention_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 32
        hidden_dim = 64
        modify_pos = 16  # Modify position 16

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            out_original = attention_causal(x, x, x)

        # Modify input at position 16
        x_modified = x.clone()
        x_modified[:, modify_pos, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            out_modified = attention_causal(x_modified, x_modified, x_modified)

        # Positions 0 to modify_pos-1 should be identical (they don't see position modify_pos)
        assert torch.allclose(out_original[:, :modify_pos, :], out_modified[:, :modify_pos, :], atol=1e-5), (
            f"Causal attention: positions before {modify_pos} should not be affected by modifying position {modify_pos}"
        )

        # Positions modify_pos onwards MAY be different (they see the modified position)
        # We don't assert they're different, just that earlier positions are unchanged

    def test_non_causal_attention_sees_future(self, attention_non_causal):
        """Test that non-causal attention IS affected by future inputs."""
        attention_non_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 32
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            out_original = attention_non_causal(x, x, x)

        # Modify input at the LAST position
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            out_modified = attention_non_causal(x_modified, x_modified, x_modified)

        # For non-causal attention, earlier positions SHOULD be affected
        # (they can see the modified future position)
        assert not torch.allclose(out_original[:, 0, :], out_modified[:, 0, :], atol=1e-5), (
            "Non-causal attention: first position should be affected by modifying last position"
        )

    def test_causal_attention_triangular_dependency(self, attention_causal):
        """Test that causal attention has triangular dependency structure."""
        attention_causal.eval()
        torch.manual_seed(42)

        batch_size = 1
        seq_len = 8
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Test each position: modifying position j should not affect positions < j
        for j in range(1, seq_len):
            x_modified = x.clone()
            x_modified[:, j, :] = torch.randn(batch_size, hidden_dim)

            with torch.no_grad():
                out_original = attention_causal(x, x, x)
                out_modified = attention_causal(x_modified, x_modified, x_modified)

            # Positions 0 to j-1 should be identical
            assert torch.allclose(out_original[:, :j, :], out_modified[:, :j, :], atol=1e-5), (
                f"Causal attention: modifying position {j} affected positions before it"
            )


class TestAttentionCausalityGradients:
    """Test causality through gradient flow."""

    def test_attention_causal_gradient_flow(self):
        """Test that gradients don't flow from future to past in causal attention."""
        from nvsubquadratic.modules.attention import Attention

        attention = Attention(
            hidden_dim=64,
            num_heads=4,
            apply_qk_norm=False,
            use_rope=True,
            is_causal=True,
            attn_dropout=0.0,
        )
        attention.train()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        # Forward pass
        out = attention(x, x, x)

        # Compute gradient w.r.t. output at position 0
        loss = out[:, 0, :].sum()
        loss.backward()

        # For causal attention, gradient at position 0 should NOT depend on positions > 0
        # This means d(loss)/d(x[position]) should be zero for positions > 0
        grad = x.grad

        # Positions 1 to seq_len-1 should have zero (or near-zero) gradients
        # because they don't affect output at position 0
        future_grad_norm = grad[:, 1:, :].abs().max().item()
        assert future_grad_norm < 1e-5, (
            f"Causal attention: gradients flow from position 0 to future positions (max grad: {future_grad_norm})"
        )

    def test_attention_non_causal_gradient_flow(self):
        """Test that gradients DO flow from future to past in non-causal attention."""
        from nvsubquadratic.modules.attention import Attention

        attention = Attention(
            hidden_dim=64,
            num_heads=4,
            apply_qk_norm=False,
            use_rope=True,
            is_causal=False,
            attn_dropout=0.0,
        )
        attention.train()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        # Forward pass
        out = attention(x, x, x)

        # Compute gradient w.r.t. output at position 0
        loss = out[:, 0, :].sum()
        loss.backward()

        # For non-causal attention, gradients SHOULD flow to all positions
        grad = x.grad
        future_grad_norm = grad[:, 1:, :].abs().max().item()
        assert future_grad_norm > 1e-5, (
            f"Non-causal attention: gradients should flow to future positions (max grad: {future_grad_norm})"
        )


################################################################################
# Mamba Causality Tests (Forward + Gradient Flow)
################################################################################


@requires_mamba
class TestMambaCausality:
    """Tests for Mamba causality."""

    @pytest.fixture
    def mamba_causal(self):
        """Create a causal (unidirectional) Mamba module."""
        from mamba_ssm import Mamba2

        from nvsubquadratic.lazy_config import LazyConfig
        from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

        mamba_layer_cfg = LazyConfig(Mamba2)(
            d_model=64,
            headdim=32,
            expand=2,
        )
        return MambaNDMixer(
            mamba_layer_cfg=mamba_layer_cfg,
            bidirectional=False,  # Causal!
        ).cuda()

    @pytest.fixture
    def mamba_bidirectional(self):
        """Create a bidirectional (non-causal) Mamba module."""
        from mamba_ssm import Mamba2

        from nvsubquadratic.lazy_config import LazyConfig
        from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

        mamba_layer_cfg = LazyConfig(Mamba2)(
            d_model=64,
            headdim=32,
            expand=2,
        )
        return MambaNDMixer(
            mamba_layer_cfg=mamba_layer_cfg,
            bidirectional=True,  # Non-causal
        ).cuda()

    def test_causal_mamba_future_independence(self, mamba_causal):
        """Test that causal Mamba output at position i is independent of input at position j > i."""
        mamba_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        with torch.no_grad():
            out_original = mamba_causal(x)

        # Modify input at the LAST position
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(batch_size, hidden_dim).cuda()

        with torch.no_grad():
            out_modified = mamba_causal(x_modified)

        # All positions EXCEPT the last should be identical
        assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-4), (
            "Causal Mamba: modifying future input should not affect past outputs"
        )

    def test_causal_mamba_middle_position(self, mamba_causal):
        """Test causality by modifying a middle position."""
        mamba_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
        hidden_dim = 64
        modify_pos = 32  # Modify position 32

        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        with torch.no_grad():
            out_original = mamba_causal(x)

        # Modify input at position 32
        x_modified = x.clone()
        x_modified[:, modify_pos, :] = torch.randn(batch_size, hidden_dim).cuda()

        with torch.no_grad():
            out_modified = mamba_causal(x_modified)

        # Positions 0 to modify_pos-1 should be identical
        assert torch.allclose(out_original[:, :modify_pos, :], out_modified[:, :modify_pos, :], atol=1e-4), (
            f"Causal Mamba: positions before {modify_pos} should not be affected by modifying position {modify_pos}"
        )

    def test_bidirectional_mamba_sees_future(self, mamba_bidirectional):
        """Test that bidirectional Mamba IS affected by future inputs.

        Note: Mamba has finite memory, so we use a shorter sequence and modify
        the second half to ensure the effect is measurable.
        """
        mamba_bidirectional.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 32  # Shorter sequence for stronger effect
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        with torch.no_grad():
            out_original = mamba_bidirectional(x)

        # Modify the SECOND HALF of the sequence (stronger signal)
        x_modified = x.clone()
        x_modified[:, seq_len // 2 :, :] = torch.randn(batch_size, seq_len // 2, hidden_dim).cuda() * 10

        with torch.no_grad():
            out_modified = mamba_bidirectional(x_modified)

        # For bidirectional Mamba, earlier positions SHOULD be affected by later positions
        # Check that the difference is non-trivial at the first position
        diff = (out_original[:, 0, :] - out_modified[:, 0, :]).abs().max().item()
        assert diff > 0.001, (
            f"Bidirectional Mamba: first position should be affected by modifying second half (diff={diff})"
        )

    def test_causal_mamba_triangular_dependency(self, mamba_causal):
        """Test that causal Mamba has triangular dependency structure."""
        mamba_causal.eval()
        torch.manual_seed(42)

        batch_size = 1
        seq_len = 16
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        # Test several positions
        for j in [4, 8, 12]:
            x_modified = x.clone()
            x_modified[:, j, :] = torch.randn(batch_size, hidden_dim).cuda()

            with torch.no_grad():
                out_original = mamba_causal(x)
                out_modified = mamba_causal(x_modified)

            # Positions 0 to j-1 should be identical
            assert torch.allclose(out_original[:, :j, :], out_modified[:, :j, :], atol=1e-4), (
                f"Causal Mamba: modifying position {j} affected positions before it"
            )


@requires_mamba
class TestMambaCausalityGradients:
    """Test causality through gradient flow for Mamba."""

    def test_mamba_causal_gradient_flow(self):
        """Test that gradients don't flow from future to past in causal Mamba."""
        from mamba_ssm import Mamba2

        from nvsubquadratic.lazy_config import LazyConfig
        from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

        mamba_layer_cfg = LazyConfig(Mamba2)(
            d_model=64,
            headdim=32,
            expand=2,
        )
        mamba = MambaNDMixer(
            mamba_layer_cfg=mamba_layer_cfg,
            bidirectional=False,  # Causal
        ).cuda()
        mamba.train()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True, device="cuda")

        # Forward pass
        out = mamba(x)

        # Compute gradient w.r.t. output at position 0
        loss = out[:, 0, :].sum()
        loss.backward()

        # For causal Mamba, gradient at position 0 should NOT depend on positions > 0
        grad = x.grad

        # Positions 1 to seq_len-1 should have zero (or near-zero) gradients
        future_grad_norm = grad[:, 1:, :].abs().max().item()
        assert future_grad_norm < 1e-5, (
            f"Causal Mamba: gradients flow from position 0 to future positions (max grad: {future_grad_norm})"
        )

    def test_mamba_bidirectional_gradient_flow(self):
        """Test that gradients DO flow from future to past in bidirectional Mamba."""
        from mamba_ssm import Mamba2

        from nvsubquadratic.lazy_config import LazyConfig
        from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

        mamba_layer_cfg = LazyConfig(Mamba2)(
            d_model=64,
            headdim=32,
            expand=2,
        )
        mamba = MambaNDMixer(
            mamba_layer_cfg=mamba_layer_cfg,
            bidirectional=True,  # Non-causal
        ).cuda()
        mamba.train()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True, device="cuda")

        # Forward pass
        out = mamba(x)

        # Compute gradient w.r.t. output at position 0
        loss = out[:, 0, :].sum()
        loss.backward()

        # For bidirectional Mamba, gradients SHOULD flow to all positions
        grad = x.grad
        future_grad_norm = grad[:, 1:, :].abs().max().item()
        assert future_grad_norm > 1e-5, (
            f"Bidirectional Mamba: gradients should flow to future positions (max grad: {future_grad_norm})"
        )


################################################################################
# Mixer-Level Causality Tests (QKVSequenceMixer wrapping operators)
################################################################################


class TestQKVSequenceMixerAttentionCausality:
    """Tests for QKVSequenceMixer with Attention - the full mixer stack.

    QKVSequenceMixer wraps the Attention operator with:
    - QKV projection (Linear) - point-wise, causality-safe
    - Attention operator - causal if is_causal=True
    - Output projection (Linear) - point-wise, causality-safe

    These tests verify the full mixer maintains causality.
    """

    @pytest.fixture
    def mixer_causal(self):
        """Create a causal QKVSequenceMixer with Attention."""
        from nvsubquadratic.lazy_config import LazyConfig, instantiate
        from nvsubquadratic.modules.attention import Attention
        from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer

        hidden_dim = 64
        mixer_cfg = LazyConfig(QKVSequenceMixer)(
            hidden_dim=hidden_dim,
            mixer_cfg=LazyConfig(Attention)(
                hidden_dim=hidden_dim,
                num_heads=4,
                apply_qk_norm=False,
                use_rope=True,
                is_causal=True,  # Causal!
                attn_dropout=0.0,
            ),
        )
        return instantiate(mixer_cfg)

    @pytest.fixture
    def mixer_non_causal(self):
        """Create a non-causal QKVSequenceMixer with Attention."""
        from nvsubquadratic.lazy_config import LazyConfig, instantiate
        from nvsubquadratic.modules.attention import Attention
        from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer

        hidden_dim = 64
        mixer_cfg = LazyConfig(QKVSequenceMixer)(
            hidden_dim=hidden_dim,
            mixer_cfg=LazyConfig(Attention)(
                hidden_dim=hidden_dim,
                num_heads=4,
                apply_qk_norm=False,
                use_rope=True,
                is_causal=False,  # Non-causal
                attn_dropout=0.0,
            ),
        )
        return instantiate(mixer_cfg)

    def test_causal_mixer_future_independence(self, mixer_causal):
        """Test that causal QKVSequenceMixer output at position i is independent of input at j > i."""
        mixer_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 32
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            out_original = mixer_causal(x)

        # Modify input at the LAST position
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            out_modified = mixer_causal(x_modified)

        # All positions EXCEPT the last should be identical
        assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-5), (
            "Causal QKVSequenceMixer: modifying future input should not affect past outputs"
        )

    def test_causal_mixer_middle_position(self, mixer_causal):
        """Test causality by modifying a middle position."""
        mixer_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 32
        hidden_dim = 64
        modify_pos = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            out_original = mixer_causal(x)

        x_modified = x.clone()
        x_modified[:, modify_pos, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            out_modified = mixer_causal(x_modified)

        # Positions 0 to modify_pos-1 should be identical
        assert torch.allclose(out_original[:, :modify_pos, :], out_modified[:, :modify_pos, :], atol=1e-5), (
            f"Causal QKVSequenceMixer: positions before {modify_pos} should not be affected"
        )

    def test_non_causal_mixer_sees_future(self, mixer_non_causal):
        """Test that non-causal QKVSequenceMixer IS affected by future inputs."""
        mixer_non_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 32
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            out_original = mixer_non_causal(x)

        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            out_modified = mixer_non_causal(x_modified)

        # For non-causal mixer, first position SHOULD be affected
        assert not torch.allclose(out_original[:, 0, :], out_modified[:, 0, :], atol=1e-5), (
            "Non-causal QKVSequenceMixer: first position should be affected by modifying last position"
        )

    def test_causal_mixer_gradient_flow(self, mixer_causal):
        """Test that gradients don't flow from future to past in causal mixer."""
        mixer_causal.train()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        out = mixer_causal(x)
        loss = out[:, 0, :].sum()
        loss.backward()

        # Future positions should have zero gradients
        future_grad_norm = x.grad[:, 1:, :].abs().max().item()
        assert future_grad_norm < 1e-5, (
            f"Causal QKVSequenceMixer: gradients flow to future positions (max grad: {future_grad_norm})"
        )


@requires_mamba
class TestMambaNDMixerCausality:
    """Tests for MambaNDMixer - the full Mamba mixer stack.

    MambaNDMixer wraps Mamba2 and handles:
    - Flattening ND inputs to 1D sequences
    - Optional bidirectional processing
    - Reshaping back to ND

    These tests verify the full mixer maintains causality when bidirectional=False.
    """

    @pytest.fixture
    def mixer_causal(self):
        """Create a causal MambaNDMixer (bidirectional=False)."""
        from mamba_ssm import Mamba2

        from nvsubquadratic.lazy_config import LazyConfig, instantiate
        from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

        hidden_dim = 64
        mixer_cfg = LazyConfig(MambaNDMixer)(
            mamba_layer_cfg=LazyConfig(Mamba2)(
                d_model=hidden_dim,
                headdim=32,
                expand=2,
            ),
            bidirectional=False,  # Causal!
        )
        return instantiate(mixer_cfg).cuda()

    @pytest.fixture
    def mixer_bidirectional(self):
        """Create a bidirectional MambaNDMixer (non-causal)."""
        from mamba_ssm import Mamba2

        from nvsubquadratic.lazy_config import LazyConfig, instantiate
        from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

        hidden_dim = 64
        mixer_cfg = LazyConfig(MambaNDMixer)(
            mamba_layer_cfg=LazyConfig(Mamba2)(
                d_model=hidden_dim,
                headdim=32,
                expand=2,
            ),
            bidirectional=True,  # Non-causal
        )
        return instantiate(mixer_cfg).cuda()

    def test_causal_mixer_future_independence(self, mixer_causal):
        """Test that causal MambaNDMixer output at position i is independent of input at j > i."""
        mixer_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        with torch.no_grad():
            out_original = mixer_causal(x)

        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(batch_size, hidden_dim).cuda()

        with torch.no_grad():
            out_modified = mixer_causal(x_modified)

        assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-4), (
            "Causal MambaNDMixer: modifying future input should not affect past outputs"
        )

    def test_causal_mixer_middle_position(self, mixer_causal):
        """Test causality by modifying a middle position."""
        mixer_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
        hidden_dim = 64
        modify_pos = 32

        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        with torch.no_grad():
            out_original = mixer_causal(x)

        x_modified = x.clone()
        x_modified[:, modify_pos, :] = torch.randn(batch_size, hidden_dim).cuda()

        with torch.no_grad():
            out_modified = mixer_causal(x_modified)

        assert torch.allclose(out_original[:, :modify_pos, :], out_modified[:, :modify_pos, :], atol=1e-4), (
            f"Causal MambaNDMixer: positions before {modify_pos} should not be affected"
        )

    def test_bidirectional_mixer_sees_future(self, mixer_bidirectional):
        """Test that bidirectional MambaNDMixer IS affected by future inputs."""
        mixer_bidirectional.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 32
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        with torch.no_grad():
            out_original = mixer_bidirectional(x)

        # Modify second half with large values for measurable effect
        x_modified = x.clone()
        x_modified[:, seq_len // 2 :, :] = torch.randn(batch_size, seq_len // 2, hidden_dim).cuda() * 10

        with torch.no_grad():
            out_modified = mixer_bidirectional(x_modified)

        diff = (out_original[:, 0, :] - out_modified[:, 0, :]).abs().max().item()
        assert diff > 0.001, (
            f"Bidirectional MambaNDMixer: first position should be affected by modifying second half (diff={diff})"
        )

    def test_causal_mixer_gradient_flow(self, mixer_causal):
        """Test that gradients don't flow from future to past in causal mixer."""
        mixer_causal.train()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True, device="cuda")

        out = mixer_causal(x)
        loss = out[:, 0, :].sum()
        loss.backward()

        future_grad_norm = x.grad[:, 1:, :].abs().max().item()
        assert future_grad_norm < 1e-5, (
            f"Causal MambaNDMixer: gradients flow to future positions (max grad: {future_grad_norm})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
