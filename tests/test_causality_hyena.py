# TODO: Add license header here

"""Tests for causality in Hyena modules.

Causality Test Principle
------------------------
A causal model ensures that output at position i only depends on inputs at positions 0, 1, ..., i.
Modifying input at position j > i should NOT change output at position i.

These tests verify causality for:
- **CKConvND**: `is_causal=True` enables causal FFT convolution
- **CausalConv1D**: Left-only padding ensures causality
- **Hyena**: Full Hyena mixer with causal global convolution
- **QKVSequenceMixer+Hyena**: Full mixer stack with projections

Test Summary
------------
+----+---------------------------------------------+---------------------+------------------------------------------+
| #  | Test                                        | Module              | What it verifies                         |
+----+---------------------------------------------+---------------------+------------------------------------------+
| **CausalConv1D Tests** - Test causal short convolution                                                                |
+----+---------------------------------------------+---------------------+------------------------------------------+
| 1  | test_causal_conv1d_future_independence      | CausalConv1D        | Last position doesn't affect earlier     |
| 2  | test_causal_conv1d_middle_position          | CausalConv1D        | Position N doesn't affect 0..N-1         |
| 3  | test_causal_conv1d_gradient_flow            | CausalConv1D        | No grad flow from pos 0 -> future        |
+----+---------------------------------------------+---------------------+------------------------------------------+
| **CKConvND Tests** - Test causal global convolution                                                                   |
+----+---------------------------------------------+---------------------+------------------------------------------+
| 4  | test_causal_ckconv_future_independence      | CKConvND            | Last position doesn't affect earlier     |
| 5  | test_causal_ckconv_middle_position          | CKConvND            | Position N doesn't affect 0..N-1         |
| 6  | test_non_causal_ckconv_sees_future          | CKConvND            | Position 0 IS affected by future         |
| 7  | test_causal_ckconv_triangular_dependency    | CKConvND            | Full triangular dependency structure     |
| 8  | test_causal_ckconv_gradient_flow            | CKConvND            | No grad flow from pos 0 -> future        |
| 9  | test_non_causal_ckconv_gradient_flow        | CKConvND            | Grad DOES flow to all positions          |
+----+---------------------------------------------+---------------------+------------------------------------------+
| **Hyena Mixer Tests** - Test full Hyena stack (short conv + global conv + gates)                                      |
+----+---------------------------------------------+---------------------+------------------------------------------+
| 10 | test_causal_hyena_future_independence       | Hyena               | Last position doesn't affect earlier     |
| 11 | test_causal_hyena_middle_position           | Hyena               | Position N doesn't affect 0..N-1         |
| 12 | test_non_causal_hyena_sees_future           | Hyena               | Position 0 IS affected by future         |
| 13 | test_causal_hyena_triangular_dependency     | Hyena               | Full triangular dependency structure     |
| 14 | test_causal_hyena_gradient_flow             | Hyena               | No grad flow from pos 0 -> future        |
| 15 | test_non_causal_hyena_gradient_flow         | Hyena               | Grad DOES flow to all positions          |
+----+---------------------------------------------+---------------------+------------------------------------------+
| **QKVSequenceMixer+Hyena Tests** - Test full mixer stack with projections                                             |
+----+---------------------------------------------+---------------------+------------------------------------------+
| 16 | test_causal_mixer_future_independence       | QKVSequenceMixer    | Last position doesn't affect earlier     |
| 17 | test_causal_mixer_middle_position           | QKVSequenceMixer    | Position N doesn't affect 0..N-1         |
| 18 | test_non_causal_mixer_sees_future           | QKVSequenceMixer    | Position 0 IS affected by future         |
| 19 | test_causal_mixer_gradient_flow             | QKVSequenceMixer    | No grad flow from pos 0 -> future        |
| 20 | test_non_causal_mixer_gradient_flow         | QKVSequenceMixer    | Grad DOES flow to all positions          |
+----+---------------------------------------------+---------------------+------------------------------------------+
| **Integration Tests**                                                                                                 |
+----+---------------------------------------------+---------------------+------------------------------------------+
| 21 | test_hyena_1d_is_causal_integration         | Hyena               | Full stack causality (matches ccnn_v2)   |
+----+---------------------------------------------+---------------------+------------------------------------------+

Usage
-----
Run all tests:
    pytest tests/test_causality_hyena.py -v -o "addopts="

Run only CausalConv1D tests:
    pytest tests/test_causality_hyena.py -v -o "addopts=" -k "CausalConv1D"

Run only Hyena tests:
    pytest tests/test_causality_hyena.py -v -o "addopts=" -k "Hyena"

Run only QKVSequenceMixer tests:
    pytest tests/test_causality_hyena.py -v -o "addopts=" -k "QKVSequenceMixer"
"""

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.causal_conv1d import CausalConv1D
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND


################################################################################
# CausalConv1D Tests
################################################################################


class TestCausalConv1DCausality:
    """Tests for CausalConv1D causality."""

    @pytest.fixture
    def causal_conv(self):
        """Create a CausalConv1D module."""
        return CausalConv1D(
            in_channels=16,
            out_channels=16,
            kernel_size=7,
            groups=16,  # depthwise
        )

    def test_causal_conv1d_future_independence(self, causal_conv):
        """Test that modifying the last position doesn't affect earlier outputs."""
        torch.manual_seed(42)
        causal_conv.eval()

        batch_size, channels, seq_len = 2, 16, 64
        x1 = torch.randn(batch_size, channels, seq_len)
        x2 = x1.clone()

        # Modify only the last position
        x2[:, :, -1] = torch.randn(batch_size, channels)

        with torch.no_grad():
            y1 = causal_conv(x1)
            y2 = causal_conv(x2)

        # All positions except the last should be identical
        assert torch.allclose(y1[:, :, :-1], y2[:, :, :-1], atol=1e-6), (
            "CausalConv1D violated causality: modifying last position affected earlier outputs"
        )

        # Last position should differ
        diff_last = (y1[:, :, -1] - y2[:, :, -1]).abs().max().item()
        assert diff_last > 1e-6, "Last position should differ after modification"

    def test_causal_conv1d_middle_position(self, causal_conv):
        """Test that modifying position N doesn't affect positions 0..N-1."""
        torch.manual_seed(42)
        causal_conv.eval()

        batch_size, channels, seq_len = 2, 16, 64
        x1 = torch.randn(batch_size, channels, seq_len)
        x2 = x1.clone()

        # Modify position 32 and everything after
        t = 32
        x2[:, :, t:] = torch.randn(batch_size, channels, seq_len - t)

        with torch.no_grad():
            y1 = causal_conv(x1)
            y2 = causal_conv(x2)

        # Positions 0..t-1 should be identical
        assert torch.allclose(y1[:, :, :t], y2[:, :, :t], atol=1e-6), (
            f"CausalConv1D violated causality: modifying position {t}+ affected positions 0..{t - 1}"
        )

    def test_causal_conv1d_gradient_flow(self, causal_conv):
        """Test that gradients don't flow from position 0 to future positions."""
        torch.manual_seed(42)

        batch_size, channels, seq_len = 2, 16, 64
        x = torch.randn(batch_size, channels, seq_len, requires_grad=True)

        y = causal_conv(x)

        # Backprop from position 0 only
        loss = y[:, :, 0].sum()
        loss.backward()

        # Gradients should only exist for positions 0..kernel_size-1 (due to receptive field)
        # But for causal, gradients at position 0 should NOT flow to future inputs
        grad = x.grad
        assert grad is not None

        # Position 0 output depends only on position 0 input (and earlier, but there are none)
        # So gradient should be zero for positions > 0
        grad_future = grad[:, :, 1:].abs().max().item()
        assert grad_future < 1e-6, f"Gradient leaked to future positions: max grad at pos > 0 = {grad_future}"


################################################################################
# CKConvND Causality Tests
################################################################################


class TestCKConvNDCausality:
    """Tests for CKConvND causality with is_causal=True."""

    @pytest.fixture
    def ckconv_causal(self):
        """Create a causal CKConvND module."""
        hidden_dim = 16
        seq_len = 64

        kernel_cfg = LazyConfig(SIRENKernelND)(
            out_dim=hidden_dim,
            data_dim=1,
            mlp_hidden_dim=32,
            num_layers=2,
            embedding_dim=32,
            omega_0=10.0,
            L_cache=seq_len,
            use_bias=True,
            hidden_omega_0=1.0,
        )

        return CKConvND(
            data_dim=1,
            hidden_dim=hidden_dim,
            kernel_cfg=kernel_cfg,
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="double",
            fft_padding="zero",
            is_causal=True,
        )

    @pytest.fixture
    def ckconv_non_causal(self):
        """Create a non-causal CKConvND module."""
        hidden_dim = 16
        seq_len = 64

        kernel_cfg = LazyConfig(SIRENKernelND)(
            out_dim=hidden_dim,
            data_dim=1,
            mlp_hidden_dim=32,
            num_layers=2,
            embedding_dim=32,
            omega_0=10.0,
            L_cache=seq_len,
            use_bias=True,
            hidden_omega_0=1.0,
        )

        return CKConvND(
            data_dim=1,
            hidden_dim=hidden_dim,
            kernel_cfg=kernel_cfg,
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="double",
            fft_padding="zero",
            is_causal=False,
        )

    def test_causal_ckconv_future_independence(self, ckconv_causal):
        """Test that modifying the last position doesn't affect earlier outputs."""
        torch.manual_seed(42)
        ckconv_causal.eval()

        batch_size, seq_len, hidden_dim = 2, 64, 16
        # BLH format: [B, L, H]
        x1 = torch.randn(batch_size, seq_len, hidden_dim)
        x2 = x1.clone()

        # Modify only the last position
        x2[:, -1, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            y1 = ckconv_causal(x1, is_bhl_input=False)
            y2 = ckconv_causal(x2, is_bhl_input=False)

        # All positions except the last should be identical
        assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-5), (
            "Causal CKConvND violated causality: modifying last position affected earlier outputs"
        )

        # Last position should differ
        diff_last = (y1[:, -1, :] - y2[:, -1, :]).abs().max().item()
        assert diff_last > 1e-6, "Last position should differ after modification"

    def test_causal_ckconv_middle_position(self, ckconv_causal):
        """Test that modifying position N doesn't affect positions 0..N-1."""
        torch.manual_seed(42)
        ckconv_causal.eval()

        batch_size, seq_len, hidden_dim = 2, 64, 16
        x1 = torch.randn(batch_size, seq_len, hidden_dim)
        x2 = x1.clone()

        # Modify position 32 and everything after
        t = 32
        x2[:, t:, :] = torch.randn(batch_size, seq_len - t, hidden_dim)

        with torch.no_grad():
            y1 = ckconv_causal(x1, is_bhl_input=False)
            y2 = ckconv_causal(x2, is_bhl_input=False)

        # Positions 0..t-1 should be identical
        assert torch.allclose(y1[:, :t, :], y2[:, :t, :], atol=1e-5), (
            f"Causal CKConvND violated causality: modifying position {t}+ affected positions 0..{t - 1}"
        )

    def test_non_causal_ckconv_sees_future(self, ckconv_non_causal):
        """Test that non-causal CKConvND DOES see future positions."""
        torch.manual_seed(42)
        ckconv_non_causal.eval()

        batch_size, seq_len, hidden_dim = 2, 64, 16
        x1 = torch.randn(batch_size, seq_len, hidden_dim)
        x2 = x1.clone()

        # Modify only the last position
        x2[:, -1, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            y1 = ckconv_non_causal(x1, is_bhl_input=False)
            y2 = ckconv_non_causal(x2, is_bhl_input=False)

        # Position 0 SHOULD be affected by the last position (non-causal)
        diff_first = (y1[:, 0, :] - y2[:, 0, :]).abs().max().item()
        assert diff_first > 1e-6, "Non-causal CKConvND should see future: position 0 was not affected by last position"

    def test_causal_ckconv_triangular_dependency(self, ckconv_causal):
        """Test full triangular dependency structure for causal CKConvND."""
        torch.manual_seed(42)
        ckconv_causal.eval()

        batch_size, seq_len, hidden_dim = 1, 32, 16
        x_base = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            y_base = ckconv_causal(x_base, is_bhl_input=False)

        # Test multiple positions
        for t in [4, 8, 16, 24]:
            x_mod = x_base.clone()
            x_mod[:, t:, :] = torch.randn(batch_size, seq_len - t, hidden_dim)

            with torch.no_grad():
                y_mod = ckconv_causal(x_mod, is_bhl_input=False)

            # Positions 0..t-1 should be identical
            assert torch.allclose(y_base[:, :t, :], y_mod[:, :t, :], atol=1e-5), (
                f"Causal CKConvND violated triangular dependency at position {t}"
            )

    def test_causal_ckconv_gradient_flow(self, ckconv_causal):
        """Test that gradients don't flow from position 0 to future positions."""
        torch.manual_seed(42)

        batch_size, seq_len, hidden_dim = 2, 64, 16
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        y = ckconv_causal(x, is_bhl_input=False)

        # Backprop from position 0 only
        loss = y[:, 0, :].sum()
        loss.backward()

        grad = x.grad
        assert grad is not None

        # For causal conv, gradient at position 0 should NOT flow to future inputs
        grad_future = grad[:, 1:, :].abs().max().item()
        assert grad_future < 1e-5, f"Gradient leaked to future positions: max grad at pos > 0 = {grad_future}"

    def test_non_causal_ckconv_gradient_flow(self, ckconv_non_causal):
        """Test that gradients DO flow to all positions in non-causal CKConvND."""
        torch.manual_seed(42)

        batch_size, seq_len, hidden_dim = 2, 64, 16
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        y = ckconv_non_causal(x, is_bhl_input=False)

        # Backprop from position 0 only
        loss = y[:, 0, :].sum()
        loss.backward()

        grad = x.grad
        assert grad is not None

        # For non-causal conv, gradients SHOULD flow to all positions
        grad_future = grad[:, 1:, :].abs().max().item()
        assert grad_future > 1e-5, (
            f"Non-causal CKConvND: gradients should flow to future positions (max grad: {grad_future})"
        )


################################################################################
# Hyena Mixer Causality Tests
################################################################################


class TestHyenaCausality:
    """Tests for Hyena mixer causality."""

    @pytest.fixture
    def hyena_causal(self):
        """Create a causal Hyena mixer."""
        hidden_dim = 16
        seq_len = 64

        global_conv_cfg = LazyConfig(CKConvND)(
            data_dim=1,
            hidden_dim=hidden_dim,
            kernel_cfg=LazyConfig(SIRENKernelND)(
                out_dim=hidden_dim,
                data_dim=1,
                mlp_hidden_dim=32,
                num_layers=2,
                embedding_dim=32,
                omega_0=10.0,
                L_cache=seq_len,
                use_bias=True,
                hidden_omega_0=1.0,
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="double",
            fft_padding="zero",
            is_causal=True,
        )

        short_conv_cfg = LazyConfig(CausalConv1D)(
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim * 3,
            kernel_size=7,
            groups=hidden_dim * 3,
        )

        return Hyena(
            global_conv_cfg=global_conv_cfg,
            short_conv_cfg=short_conv_cfg,
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            apply_qk_norm=True,
            use_rope=True,
            output_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
        )

    @pytest.fixture
    def hyena_non_causal(self):
        """Create a non-causal Hyena mixer."""
        hidden_dim = 16
        seq_len = 64

        global_conv_cfg = LazyConfig(CKConvND)(
            data_dim=1,
            hidden_dim=hidden_dim,
            kernel_cfg=LazyConfig(SIRENKernelND)(
                out_dim=hidden_dim,
                data_dim=1,
                mlp_hidden_dim=32,
                num_layers=2,
                embedding_dim=32,
                omega_0=10.0,
                L_cache=seq_len,
                use_bias=True,
                hidden_omega_0=1.0,
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="double",
            fft_padding="zero",
            is_causal=False,
        )

        # Non-causal short conv (standard Conv1d with padding)
        short_conv_cfg = LazyConfig(torch.nn.Conv1d)(
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim * 3,
            kernel_size=7,
            padding=3,  # same padding
            groups=hidden_dim * 3,
        )

        return Hyena(
            global_conv_cfg=global_conv_cfg,
            short_conv_cfg=short_conv_cfg,
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            apply_qk_norm=True,
            use_rope=True,
            output_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
        )

    def test_causal_hyena_future_independence(self, hyena_causal):
        """Test that modifying the last position doesn't affect earlier outputs."""
        torch.manual_seed(42)
        hyena_causal.eval()

        batch_size, seq_len, hidden_dim = 2, 64, 16
        # Hyena expects [B, L, H] format (channels-last)
        x1 = torch.randn(batch_size, seq_len, hidden_dim)
        x2 = x1.clone()

        # Modify only the last position
        x2[:, -1, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            # Hyena takes q, k, v - use same input for self-mixing
            y1 = hyena_causal(x1, x1, x1)
            y2 = hyena_causal(x2, x2, x2)

        # All positions except the last should be identical
        assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-4), (
            "Causal Hyena violated causality: modifying last position affected earlier outputs"
        )

        # Last position should differ
        diff_last = (y1[:, -1, :] - y2[:, -1, :]).abs().max().item()
        assert diff_last > 1e-6, "Last position should differ after modification"

    def test_causal_hyena_middle_position(self, hyena_causal):
        """Test that modifying position N doesn't affect positions 0..N-1."""
        torch.manual_seed(42)
        hyena_causal.eval()

        batch_size, seq_len, hidden_dim = 2, 64, 16
        x1 = torch.randn(batch_size, seq_len, hidden_dim)
        x2 = x1.clone()

        # Modify position 32 and everything after
        t = 32
        x2[:, t:, :] = torch.randn(batch_size, seq_len - t, hidden_dim)

        with torch.no_grad():
            y1 = hyena_causal(x1, x1, x1)
            y2 = hyena_causal(x2, x2, x2)

        # Positions 0..t-1 should be identical
        assert torch.allclose(y1[:, :t, :], y2[:, :t, :], atol=1e-4), (
            f"Causal Hyena violated causality: modifying position {t}+ affected positions 0..{t - 1}"
        )

    def test_non_causal_hyena_sees_future(self, hyena_non_causal):
        """Test that non-causal Hyena DOES see future positions."""
        torch.manual_seed(42)
        hyena_non_causal.eval()

        batch_size, seq_len, hidden_dim = 2, 64, 16
        x1 = torch.randn(batch_size, seq_len, hidden_dim)
        x2 = x1.clone()

        # Modify only the last position
        x2[:, -1, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            y1 = hyena_non_causal(x1, x1, x1)
            y2 = hyena_non_causal(x2, x2, x2)

        # Position 0 SHOULD be affected by the last position (non-causal)
        diff_first = (y1[:, 0, :] - y2[:, 0, :]).abs().max().item()
        assert diff_first > 1e-6, "Non-causal Hyena should see future: position 0 was not affected by last position"

    def test_causal_hyena_triangular_dependency(self, hyena_causal):
        """Test full triangular dependency structure for causal Hyena."""
        torch.manual_seed(42)
        hyena_causal.eval()

        batch_size, seq_len, hidden_dim = 1, 32, 16
        x_base = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            y_base = hyena_causal(x_base, x_base, x_base)

        # Test multiple positions
        for t in [4, 8, 16, 24]:
            x_mod = x_base.clone()
            x_mod[:, t:, :] = torch.randn(batch_size, seq_len - t, hidden_dim)

            with torch.no_grad():
                y_mod = hyena_causal(x_mod, x_mod, x_mod)

            # Positions 0..t-1 should be identical
            assert torch.allclose(y_base[:, :t, :], y_mod[:, :t, :], atol=1e-4), (
                f"Causal Hyena violated triangular dependency at position {t}"
            )

    def test_causal_hyena_gradient_flow(self, hyena_causal):
        """Test that gradients don't flow from position 0 to future positions."""
        torch.manual_seed(42)

        batch_size, seq_len, hidden_dim = 2, 64, 16
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        y = hyena_causal(x, x, x)

        # Backprop from position 0 only
        loss = y[:, 0, :].sum()
        loss.backward()

        grad = x.grad
        assert grad is not None

        # For causal Hyena, gradient at position 0 should NOT flow to future inputs
        grad_future = grad[:, 1:, :].abs().max().item()
        assert grad_future < 1e-4, f"Gradient leaked to future positions: max grad at pos > 0 = {grad_future}"

    def test_non_causal_hyena_gradient_flow(self):
        """Test that gradients DO flow to future positions in non-causal Hyena.

        Uses simplified Hyena (Identity gate, no norms) for cleaner gradient signal.
        """
        torch.manual_seed(42)

        hidden_dim = 16
        seq_len = 64

        # Create simplified non-causal Hyena with Identity gate (no gradient dampening)
        global_conv_cfg = LazyConfig(CKConvND)(
            data_dim=1,
            hidden_dim=hidden_dim,
            kernel_cfg=LazyConfig(SIRENKernelND)(
                out_dim=hidden_dim,
                data_dim=1,
                mlp_hidden_dim=32,
                num_layers=2,
                embedding_dim=32,
                omega_0=10.0,
                L_cache=seq_len,
                use_bias=True,
                hidden_omega_0=1.0,
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="double",
            fft_padding="zero",
            is_causal=False,  # Non-causal
        )

        # Standard Conv1d (non-causal) for short conv
        short_conv_cfg = LazyConfig(torch.nn.Conv1d)(
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim * 3,
            kernel_size=7,
            padding=3,
            groups=hidden_dim * 3,
        )

        # Simplified Hyena: Identity gate (no gating), Identity norms
        hyena_simple = Hyena(
            global_conv_cfg=global_conv_cfg,
            short_conv_cfg=short_conv_cfg,
            gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),  # No gate!
            pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),  # No norm
            apply_qk_norm=False,  # No QK norm
            use_rope=False,  # No RoPE
            output_norm_cfg=LazyConfig(torch.nn.Identity)(),  # No output norm
        )
        hyena_simple.train()

        batch_size = 2
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        y = hyena_simple(x, x, x)
        loss = y[:, 0, :].sum()
        loss.backward()

        grad = x.grad
        assert grad is not None

        # For non-causal Hyena, gradient SHOULD flow to future positions
        grad_future = grad[:, 1:, :].abs().max().item()
        assert grad_future > 1e-5, (
            f"Non-causal Hyena: gradients should flow to future positions (max grad: {grad_future})"
        )


################################################################################
# QKVSequenceMixer with Hyena Causality Tests
################################################################################


class TestQKVSequenceMixerHyenaCausality:
    """Tests for QKVSequenceMixer with Hyena - the full mixer stack.

    QKVSequenceMixer wraps the Hyena operator with:
    - QKV projection (Linear) - point-wise, causality-safe
    - Hyena operator - causal if configured with causal CKConvND and CausalConv1D
    - Output projection (Linear) - point-wise, causality-safe

    These tests verify the full mixer maintains causality.
    """

    @pytest.fixture
    def mixer_causal(self):
        """Create a causal QKVSequenceMixer with Hyena."""
        from nvsubquadratic.lazy_config import LazyConfig, instantiate
        from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer

        hidden_dim = 64
        seq_len = 64

        global_conv_cfg = LazyConfig(CKConvND)(
            data_dim=1,
            hidden_dim=hidden_dim,
            kernel_cfg=LazyConfig(SIRENKernelND)(
                out_dim=hidden_dim,
                data_dim=1,
                mlp_hidden_dim=32,
                num_layers=2,
                embedding_dim=32,
                omega_0=10.0,
                L_cache=seq_len,
                use_bias=True,
                hidden_omega_0=1.0,
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="double",
            fft_padding="zero",
            is_causal=True,  # Causal!
        )

        short_conv_cfg = LazyConfig(CausalConv1D)(  # Causal short conv!
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim * 3,
            kernel_size=7,
            groups=hidden_dim * 3,
        )

        mixer_cfg = LazyConfig(QKVSequenceMixer)(
            hidden_dim=hidden_dim,
            mixer_cfg=LazyConfig(Hyena)(
                global_conv_cfg=global_conv_cfg,
                short_conv_cfg=short_conv_cfg,
                gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
                apply_qk_norm=True,
                use_rope=True,
                output_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            ),
        )
        return instantiate(mixer_cfg)

    @pytest.fixture
    def mixer_non_causal(self):
        """Create a non-causal QKVSequenceMixer with Hyena."""
        from nvsubquadratic.lazy_config import LazyConfig, instantiate
        from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer

        hidden_dim = 64
        seq_len = 64

        global_conv_cfg = LazyConfig(CKConvND)(
            data_dim=1,
            hidden_dim=hidden_dim,
            kernel_cfg=LazyConfig(SIRENKernelND)(
                out_dim=hidden_dim,
                data_dim=1,
                mlp_hidden_dim=32,
                num_layers=2,
                embedding_dim=32,
                omega_0=10.0,
                L_cache=seq_len,
                use_bias=True,
                hidden_omega_0=1.0,
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="double",
            fft_padding="zero",
            is_causal=False,  # Non-causal
        )

        # Non-causal short conv (standard Conv1d with padding)
        short_conv_cfg = LazyConfig(torch.nn.Conv1d)(
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim * 3,
            kernel_size=7,
            padding=3,  # same padding
            groups=hidden_dim * 3,
        )

        mixer_cfg = LazyConfig(QKVSequenceMixer)(
            hidden_dim=hidden_dim,
            mixer_cfg=LazyConfig(Hyena)(
                global_conv_cfg=global_conv_cfg,
                short_conv_cfg=short_conv_cfg,
                gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
                apply_qk_norm=True,
                use_rope=True,
                output_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            ),
        )
        return instantiate(mixer_cfg)

    def test_causal_mixer_future_independence(self, mixer_causal):
        """Test that causal QKVSequenceMixer+Hyena output at position i is independent of input at j > i."""
        mixer_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
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
        assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-4), (
            "Causal QKVSequenceMixer+Hyena: modifying future input should not affect past outputs"
        )

    def test_causal_mixer_middle_position(self, mixer_causal):
        """Test causality by modifying a middle position."""
        mixer_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
        hidden_dim = 64
        modify_pos = 32

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            out_original = mixer_causal(x)

        x_modified = x.clone()
        x_modified[:, modify_pos, :] = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            out_modified = mixer_causal(x_modified)

        # Positions 0 to modify_pos-1 should be identical
        assert torch.allclose(out_original[:, :modify_pos, :], out_modified[:, :modify_pos, :], atol=1e-4), (
            f"Causal QKVSequenceMixer+Hyena: positions before {modify_pos} should not be affected"
        )

    def test_non_causal_mixer_sees_future(self, mixer_non_causal):
        """Test that non-causal QKVSequenceMixer+Hyena IS affected by future inputs."""
        mixer_non_causal.eval()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
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
            "Non-causal QKVSequenceMixer+Hyena: first position should be affected by modifying last position"
        )

    def test_causal_mixer_gradient_flow(self, mixer_causal):
        """Test that gradients don't flow from future to past in causal mixer."""
        mixer_causal.train()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        out = mixer_causal(x)
        loss = out[:, 0, :].sum()
        loss.backward()

        # Future positions should have zero gradients
        future_grad_norm = x.grad[:, 1:, :].abs().max().item()
        assert future_grad_norm < 1e-4, (
            f"Causal QKVSequenceMixer+Hyena: gradients flow to future positions (max grad: {future_grad_norm})"
        )

    def test_non_causal_mixer_gradient_flow(self, mixer_non_causal):
        """Test that gradients DO flow to all positions in non-causal mixer."""
        mixer_non_causal.train()
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 64
        hidden_dim = 64

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        out = mixer_non_causal(x)
        loss = out[:, 0, :].sum()
        loss.backward()

        # For non-causal mixer, gradients SHOULD flow to all positions
        future_grad_norm = x.grad[:, 1:, :].abs().max().item()
        assert future_grad_norm > 1e-5, (
            f"Non-causal QKVSequenceMixer+Hyena: gradients should flow to future positions (max grad: {future_grad_norm})"
        )


################################################################################
# Integration Test - Matches ccnn_v2 test
################################################################################


@torch.no_grad()
def test_hyena_1d_is_causal_integration():
    """Integration test matching ccnn_v2's hyena_nd_1d_causality_test.py.

    This test verifies that the full Hyena stack with causal settings
    properly maintains causality.
    """
    torch.manual_seed(0)

    # Shapes
    batch_size = 2
    seq_len = 64
    hidden_dim = 16

    # Build a Hyena mixer with causal settings
    global_conv_cfg = LazyConfig(CKConvND)(
        data_dim=1,
        hidden_dim=hidden_dim,
        kernel_cfg=LazyConfig(SIRENKernelND)(
            out_dim=hidden_dim,
            data_dim=1,
            mlp_hidden_dim=32,
            num_layers=3,
            embedding_dim=32,
            omega_0=10.0,
            L_cache=seq_len,
            use_bias=True,
            hidden_omega_0=1.0,
        ),
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type="double",
        fft_padding="zero",
        is_causal=True,
    )

    short_conv_cfg = LazyConfig(CausalConv1D)(
        in_channels=hidden_dim * 3,
        out_channels=hidden_dim * 3,
        groups=hidden_dim * 3,
        kernel_size=7,
    )

    model = Hyena(
        global_conv_cfg=global_conv_cfg,
        short_conv_cfg=short_conv_cfg,
        gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
        pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim, eps=1e-5),
        apply_qk_norm=True,
        use_rope=True,
        output_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim, eps=1e-5),
    )
    model.eval()

    # Inputs (channels-last): [B, T, C]
    x = torch.randn(batch_size, seq_len, hidden_dim)
    x2 = x.clone()

    # Modify the future after time t; keep prefix identical
    t = seq_len // 2
    x2[:, t + 1 :, :] = torch.randn_like(x2[:, t + 1 :, :])

    # Feed q=k=v for a self-mixing check
    y1 = model(x, x, x)
    y2 = model(x2, x2, x2)

    # Assert causality: outputs up to and including t must be identical
    assert torch.allclose(y1[:, : t + 1], y2[:, : t + 1], atol=1e-5, rtol=1e-4), (
        "Hyena with causal CKConvND violated causality: early outputs changed when only future inputs were modified."
    )

    # Sanity: outputs should differ somewhere after t (most of the time)
    diff_future = (y1[:, t + 1 :] - y2[:, t + 1 :]).abs().max().item()
    assert diff_future > 1e-7, "Future outputs did not change; test may be degenerate."


if __name__ == "__main__":
    test_hyena_1d_is_causal_integration()
    print("✅ Hyena 1D causal test passed.")
