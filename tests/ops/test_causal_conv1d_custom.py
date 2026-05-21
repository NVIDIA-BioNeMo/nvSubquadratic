# TODO: Add license header here


"""Tests for the direct (non-FFT) 1D causal conv wrappers.

Validates :mod:`nvsubquadratic.ops.causal_conv1d_custom`:

- :func:`causal_conv1d` — depthwise short causal conv (also exercised
  indirectly via :class:`SubqOpsCausalConv1d`, but tested here standalone for
  silu and bias paths).
- :func:`b2b_causal_conv1d` — fused back-to-back causal conv. Not wired into
  any module yet; this test pins the output shape contract so a future
  fused-Hyena variant can rely on it.

Usage (requires GPU):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/ops/test_causal_conv1d_custom.py -v -o addopts=""
"""

import pytest
import torch
import torch.nn.functional as F


ATOL = 1e-3
RTOL = 1e-4


def _has_subq_ops() -> bool:
    try:
        from subquadratic_ops_torch.b2b_causal_conv1d import b2b_causal_conv1d  # noqa: F401
        from subquadratic_ops_torch.causal_conv1d import causal_conv1d  # noqa: F401

        return True
    except ImportError:
        return False


requires_subq_ops = pytest.mark.skipif(
    not _has_subq_ops(),
    reason="subquadratic_ops_torch direct 1D kernels not available",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

pytestmark = [requires_subq_ops, requires_cuda]


class TestCausalConv1d:
    """Direct tests for nvsubquadratic.ops.causal_conv1d_custom.causal_conv1d."""

    def _torch_ref(self, x, weight, bias, activation):
        """Reference: causal pad + F.conv1d + activation."""
        K = weight.shape[-1]
        x_padded = F.pad(x, (K - 1, 0))
        w_3d = weight.unsqueeze(1)  # [C, K] -> [C, 1, K] for nn.Conv1d depthwise
        y = F.conv1d(x_padded, w_3d, bias, groups=weight.shape[0])
        return F.silu(y) if activation == "silu" else y

    @pytest.mark.parametrize("C, L, K", [(16, 64, 3), (36, 256, 3), (64, 128, 7)])
    def test_matches_reference_no_bias(self, C, L, K):
        from nvsubquadratic.ops.causal_conv1d_custom import causal_conv1d

        torch.manual_seed(42)
        x = torch.randn(2, C, L, device="cuda", dtype=torch.float32)
        w = torch.randn(C, K, device="cuda", dtype=torch.float32)

        y = causal_conv1d(x, w, None, "identity")
        y_ref = self._torch_ref(x, w, None, "identity")
        torch.testing.assert_close(y, y_ref, atol=ATOL, rtol=RTOL)

    def test_matches_reference_with_bias(self):
        from nvsubquadratic.ops.causal_conv1d_custom import causal_conv1d

        torch.manual_seed(42)
        C, L, K = 36, 256, 3
        x = torch.randn(2, C, L, device="cuda", dtype=torch.float32)
        w = torch.randn(C, K, device="cuda", dtype=torch.float32)
        b = torch.randn(C, device="cuda", dtype=torch.float32)

        y = causal_conv1d(x, w, b, "identity")
        y_ref = self._torch_ref(x, w, b, "identity")
        torch.testing.assert_close(y, y_ref, atol=ATOL, rtol=RTOL)

    def test_silu_activation(self):
        from nvsubquadratic.ops.causal_conv1d_custom import causal_conv1d

        torch.manual_seed(42)
        C, L, K = 36, 256, 3
        x = torch.randn(2, C, L, device="cuda", dtype=torch.float32)
        w = torch.randn(C, K, device="cuda", dtype=torch.float32)

        y = causal_conv1d(x, w, None, "silu")
        y_ref = self._torch_ref(x, w, None, "silu")
        torch.testing.assert_close(y, y_ref, atol=ATOL, rtol=RTOL)


class TestB2BCausalConv1d:
    """Shape + finiteness contract for the b2b fused kernel.

    This is the only test for b2b_causal_conv1d in the suite — the wrapper is
    a thin pass-through and the kernel itself is upstream's responsibility.
    A future fused-Hyena module will add semantic tests.
    """

    # b2b kernel supports specific projection kernel sizes only.
    @pytest.mark.parametrize("B, C, L, K", [(2, 16, 64, 3), (4, 32, 128, 4), (2, 16, 128, 8)])
    def test_output_shape_and_finite(self, B, C, L, K):
        from nvsubquadratic.ops.causal_conv1d_custom import b2b_causal_conv1d

        torch.manual_seed(42)
        x = torch.randn(B, 3 * C, L, device="cuda", dtype=torch.float32)
        w_proj = torch.randn(3 * C, K, device="cuda", dtype=torch.float32)
        w_mixer = torch.randn(C, K, device="cuda", dtype=torch.float32)
        skip_bias = torch.randn(C, device="cuda", dtype=torch.float32)

        y = b2b_causal_conv1d(x, w_proj, w_mixer, skip_bias)
        assert y.shape == (B, C, L)
        assert torch.isfinite(y).all()
