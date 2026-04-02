# TODO: Add license header here


"""Integration tests for CKConvND with fft_backend='subq_ops'.

Validates that CKConvND produces correct forward and backward results when
using the ``subquadratic_ops_torch`` CUDA FFT backend, by comparing against
the default ``torch_fft`` backend.

Also tests that invalid configurations are properly rejected with assertions.

Usage (requires GPU — run inside SLURM):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/modules/test_ckconv_nd_subq.py -v -o addopts=""
"""

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND


ATOL = 1e-3
RTOL = 1e-4
ATOL_GRAD = 1e-2
RTOL_GRAD = 1e-3


def _has_subq_ops() -> bool:
    try:
        from subquadratic_ops_torch.fft_conv2d import fft_conv2d  # noqa: F401

        return True
    except ImportError:
        return False


requires_subq_ops = pytest.mark.skipif(
    not _has_subq_ops(),
    reason="subquadratic_ops_torch not installed",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

pytestmark = [requires_subq_ops, requires_cuda]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
SPATIAL = 8
L_CACHE = SPATIAL


def _make_ckconv(grid_type, fft_backend, use_chunked=False):
    """Build a CKConvND instance with a small SIREN kernel for testing."""
    kernel_cfg = LazyConfig(SIRENKernelND)(
        data_dim=2,
        out_dim=HIDDEN_DIM,
        mlp_hidden_dim=16,
        num_layers=2,
        embedding_dim=16,
        omega_0=10.0,
        L_cache=L_CACHE,
        use_bias=True,
    )
    return CKConvND(
        data_dim=2,
        hidden_dim=HIDDEN_DIM,
        kernel_cfg=kernel_cfg,
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type=grid_type,
        fft_padding="zero",
        fft_backend=fft_backend,
        use_chunked_fftconv=use_chunked,
    )


def _sync_weights(src: CKConvND, dst: CKConvND):
    """Copy all parameters and buffers from src to dst (in-place)."""
    dst.load_state_dict(src.state_dict())


# ---------------------------------------------------------------------------
# Forward + backward correctness: subq_ops vs torch_fft
# ---------------------------------------------------------------------------


# TODO(@moradza): remove skip when subquadratic-ops produces correct results
@pytest.mark.skip(
    reason="subquadratic_ops_torch does not produce correct results with the currently installed version; pending next release"
)
class TestForwardBackward:
    """CKConvND with subq_ops produces same output as torch_fft."""

    @pytest.mark.parametrize("grid_type", ["single", "double"])
    def test_forward_matches(self, grid_type):
        """Forward output matches between backends."""
        torch.manual_seed(42)
        model_ref = _make_ckconv(grid_type, "torch_fft").cuda()
        model_subq = _make_ckconv(grid_type, "subq_ops").cuda()
        _sync_weights(model_ref, model_subq)

        x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda", dtype=torch.float32)

        y_ref = model_ref(x)
        y_subq = model_subq(x)

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("grid_type", ["single", "double"])
    def test_backward_matches(self, grid_type):
        """Gradients match between backends."""
        torch.manual_seed(42)
        model_ref = _make_ckconv(grid_type, "torch_fft").cuda()
        model_subq = _make_ckconv(grid_type, "subq_ops").cuda()
        _sync_weights(model_ref, model_subq)

        x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda", dtype=torch.float32)

        # Reference backward
        y_ref = model_ref(x.clone().requires_grad_(True))
        y_ref.sum().backward()
        ref_grads = {n: p.grad.clone() for n, p in model_ref.named_parameters() if p.grad is not None}

        # subq_ops backward
        y_subq = model_subq(x.clone().requires_grad_(True))
        y_subq.sum().backward()
        subq_grads = {n: p.grad.clone() for n, p in model_subq.named_parameters() if p.grad is not None}

        for name in ref_grads:
            torch.testing.assert_close(
                subq_grads[name],
                ref_grads[name],
                atol=ATOL_GRAD,
                rtol=RTOL_GRAD,
                msg=f"Gradient mismatch for {name}",
            )

    def test_forward_bhl_input(self):
        """Forward matches when using is_bhl_input=True."""
        torch.manual_seed(42)
        model_ref = _make_ckconv("double", "torch_fft").cuda()
        model_subq = _make_ckconv("double", "subq_ops").cuda()
        _sync_weights(model_ref, model_subq)

        x = torch.randn(2, HIDDEN_DIM, SPATIAL, SPATIAL, device="cuda", dtype=torch.float32)

        y_ref = model_ref(x, is_bhl_input=True)
        y_subq = model_subq(x, is_bhl_input=True)

        torch.testing.assert_close(y_subq, y_ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Chunked integration
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="subquadratic_ops_torch does not produce correct results with the currently installed version; pending next release"
)
class TestChunked:
    """CKConvND with subq_ops + use_chunked_fftconv matches non-chunked."""

    @pytest.mark.parametrize("grid_type", ["single", "double"])
    def test_chunked_matches_non_chunked(self, grid_type):
        """Chunked subq_ops produces identical output to non-chunked subq_ops."""
        torch.manual_seed(42)
        model = _make_ckconv(grid_type, "subq_ops", use_chunked=False).cuda()
        model_chunked = _make_ckconv(grid_type, "subq_ops", use_chunked=True).cuda()
        _sync_weights(model, model_chunked)

        x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda", dtype=torch.float32)

        y = model(x)
        y_chunked = model_chunked(x)

        torch.testing.assert_close(y_chunked, y, atol=0, rtol=0)

    def test_chunked_matches_torch_fft(self):
        """Chunked subq_ops matches torch_fft reference."""
        torch.manual_seed(42)
        model_ref = _make_ckconv("double", "torch_fft").cuda()
        model_chunked = _make_ckconv("double", "subq_ops", use_chunked=True).cuda()
        _sync_weights(model_ref, model_chunked)

        x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda", dtype=torch.float32)

        y_ref = model_ref(x)
        y_chunked = model_chunked(x)

        torch.testing.assert_close(y_chunked, y_ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Assertion tests — invalid configurations must be rejected
# ---------------------------------------------------------------------------


class TestAssertions:
    """CKConvND raises AssertionError for invalid subq_ops configurations."""

    def test_rejects_data_dim_1(self):
        """data_dim=1 is rejected."""
        with pytest.raises(AssertionError, match="only supports 2D"):
            CKConvND(
                data_dim=1,
                hidden_dim=32,
                kernel_cfg=LazyConfig(torch.nn.Identity)(),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                fft_backend="subq_ops",
            )

    def test_rejects_data_dim_3(self):
        """data_dim=3 is rejected."""
        with pytest.raises(AssertionError, match="only supports 2D"):
            CKConvND(
                data_dim=3,
                hidden_dim=32,
                kernel_cfg=LazyConfig(torch.nn.Identity)(),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                fft_backend="subq_ops",
            )

    def test_rejects_circular_padding(self):
        """fft_padding='circular' is rejected."""
        with pytest.raises(AssertionError, match="only supports zero-padded"):
            CKConvND(
                data_dim=2,
                hidden_dim=32,
                kernel_cfg=LazyConfig(torch.nn.Identity)(),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="single",
                fft_padding="circular",
                fft_backend="subq_ops",
            )

    def test_rejects_causal_with_2d(self):
        """is_causal=True + data_dim=2 is rejected (causal requires 1D)."""
        with pytest.raises(AssertionError, match="Causal CKConvND only supports 1D"):
            CKConvND(
                data_dim=2,
                hidden_dim=32,
                kernel_cfg=LazyConfig(torch.nn.Identity)(),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                is_causal=True,
                fft_backend="subq_ops",
            )

    def test_rejects_causal_with_1d(self):
        """is_causal=True + data_dim=1 hits the subq_ops data_dim=2 constraint."""
        with pytest.raises(AssertionError, match="only supports 2D"):
            CKConvND(
                data_dim=1,
                hidden_dim=32,
                kernel_cfg=LazyConfig(torch.nn.Identity)(),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                is_causal=True,
                fft_backend="subq_ops",
            )

    def test_rejects_fp16_fft(self):
        """use_fp16_fft=True is rejected."""
        with pytest.raises(AssertionError, match="does not support fp16 FFT"):
            CKConvND(
                data_dim=2,
                hidden_dim=32,
                kernel_cfg=LazyConfig(torch.nn.Identity)(),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                use_fp16_fft=True,
                fft_backend="subq_ops",
            )

    def test_rejects_invalid_backend(self):
        """Invalid fft_backend string is rejected."""
        with pytest.raises(AssertionError, match="Invalid fft_backend"):
            CKConvND(
                data_dim=2,
                hidden_dim=32,
                kernel_cfg=LazyConfig(torch.nn.Identity)(),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                fft_backend="invalid",
            )
