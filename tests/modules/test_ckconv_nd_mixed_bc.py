# TODO: Add license header here


"""Integration tests for CKConvND with mixed boundary-condition FFT padding.

Validates the new per-axis ``fft_padding`` API (comma-separated string or
sequence of mode strings) of :class:`nvsubquadratic.modules.ckconv_nd.CKConvND`
against the existing single-mode API and against analytical / op-level
references:

1. **Equivalence with legacy single-mode strings**:
   - ``fft_padding="zero, zero, ..."`` (or ``["zero", ...]``) produces
     bit-identical output to ``fft_padding='zero', grid_type='double'``.
   - ``fft_padding="circular, circular, ..."`` produces bit-identical
     output to ``fft_padding='circular', grid_type='single'``.

   These prove the legacy code paths are untouched.

2. **Mixed-BC correctness**: forward output of CKConvND with a per-axis
   ``periodic`` tuple matches the corresponding call to the underlying
   :mod:`nvsubquadratic.ops.mixed_fftconv` op directly (with the same kernel
   produced by ``CKConvND.kernel``).

3. **Per-axis kernel size**: the SIREN kernel is constructed with
   ``(s+1)//2`` grid points on periodic axes and ``s`` on non-periodic
   axes — verified via the generated kernel's spatial shape.

4. **FLOP accounting**: ``flop_count`` uses per-axis padded sizes (no
   padding on periodic axes, ``min(N + (K+1)//2, 2N)`` on non-periodic).

5. **Validation errors**: incompatible argument combinations raise the
   expected exceptions.

All numerical tests require CUDA.
"""

from __future__ import annotations

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import (
    CKConvND,
    _grid_is_single_per_axis,
    _resolve_periodic,
)
from nvsubquadratic.modules.kernels_nd import SIRENKernelND


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CKConvND FFT path")


# ---------------------------------------------------------------------------
# Common builder
# ---------------------------------------------------------------------------

HIDDEN_DIM = 16
SPATIAL = 16


def _periodic_to_fft_padding(periodic: tuple[bool, ...]) -> list[str]:
    """Convert a bool-tuple periodicity spec to the public per-axis form.

    Tests use ``periodic`` bool tuples for parametrisation (concise + easy
    to use in assertions on ``_periodic_per_axis``), but the public CKConvND
    API only accepts mode-name strings — this helper bridges the two.

    Returns the **list form** (always unambiguously per-axis, even with one
    axis: ``["zero"]`` rather than ``"zero"`` which would be parsed as the
    legacy single-mode form requiring ``grid_type``). The comma-separated
    string form is exercised directly by ``test_comma_string_and_list_forms_are_equivalent``.
    """
    return ["circular" if p else "zero" for p in periodic]


def _make_kernel_cfg(data_dim: int, L_cache: int = SPATIAL):
    return LazyConfig(SIRENKernelND)(
        data_dim=data_dim,
        out_dim=HIDDEN_DIM,
        mlp_hidden_dim=16,
        num_layers=2,
        embedding_dim=16,
        omega_0=10.0,
        L_cache=L_cache,
        use_bias=True,
    )


def _make_ckconv(
    *,
    data_dim: int,
    fft_padding,
    grid_type=None,
    use_chunked_fftconv: bool = False,
    L_cache: int = SPATIAL,
):
    return CKConvND(
        data_dim=data_dim,
        hidden_dim=HIDDEN_DIM,
        kernel_cfg=_make_kernel_cfg(data_dim, L_cache=L_cache),
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type=grid_type,
        fft_padding=fft_padding,
        use_chunked_fftconv=use_chunked_fftconv,
    )


def _sync_weights(src: CKConvND, dst: CKConvND) -> None:
    dst.load_state_dict(src.state_dict())


# ---------------------------------------------------------------------------
# 1. Resolver / helper unit tests (CPU-only, fast)
# ---------------------------------------------------------------------------


class TestResolverHelpers:
    """Unit tests for ``_resolve_periodic`` and ``_grid_is_single_per_axis``."""

    # ── single-mode strings (legacy form) ────────────────────────────────────
    def test_resolve_zero_string(self):
        assert _resolve_periodic("zero", 1) == (False,)
        assert _resolve_periodic("zero", 3) == (False, False, False)

    def test_resolve_circular_string(self):
        assert _resolve_periodic("circular", 1) == (True,)
        assert _resolve_periodic("circular", 2) == (True, True)

    # ── comma-separated strings (per-axis form, preferred for YAML/CLI) ──────
    def test_resolve_comma_separated_2d(self):
        assert _resolve_periodic("circular, zero", 2) == (True, False)
        assert _resolve_periodic("zero, circular", 2) == (False, True)

    def test_resolve_comma_separated_3d(self):
        assert _resolve_periodic("circular, circular, zero", 3) == (True, True, False)
        assert _resolve_periodic("zero, circular, zero", 3) == (False, True, False)

    def test_resolve_comma_separated_normalises_whitespace_and_case(self):
        assert _resolve_periodic("Circular,Zero", 2) == (True, False)
        assert _resolve_periodic("  CIRCULAR  ,  zero  ", 2) == (True, False)

    # ── sequence-of-strings form (equivalent to comma-separated) ─────────────
    def test_resolve_sequence_of_strings_list(self):
        assert _resolve_periodic(["circular", "zero"], 2) == (True, False)
        assert _resolve_periodic(["zero", "zero", "circular"], 3) == (False, False, True)

    def test_resolve_sequence_of_strings_tuple(self):
        assert _resolve_periodic(("circular", "zero"), 2) == (True, False)

    # ── error cases ───────────────────────────────────────────────────────────
    def test_resolve_unknown_mode_raises(self):
        with pytest.raises(ValueError, match=r"Invalid padding mode 'nonsense'"):
            _resolve_periodic("nonsense", 2)

    def test_resolve_unknown_mode_in_comma_form_raises(self):
        with pytest.raises(ValueError, match=r"Invalid padding mode 'wall'"):
            _resolve_periodic("circular, wall", 2)

    def test_resolve_comma_wrong_length(self):
        with pytest.raises(ValueError, match=r"exactly data_dim=2"):
            _resolve_periodic("zero, circular, zero", 2)

    def test_resolve_sequence_wrong_length(self):
        with pytest.raises(ValueError, match=r"length data_dim=2"):
            _resolve_periodic(["zero", "circular", "zero"], 2)

    def test_resolve_bool_sequence_redirects_to_string_api(self):
        # Boolean sequences used to be a supported input briefly during
        # development and were replaced with mode-name strings for
        # readability. The error message must guide users to the new form.
        with pytest.raises(ValueError, match=r"no longer accepts a sequence of booleans"):
            _resolve_periodic((True, False), 2)

    def test_resolve_lone_bool_redirects_to_string_api(self):
        with pytest.raises(ValueError, match=r"True/False is not a valid input"):
            _resolve_periodic(True, 2)

    def test_resolve_wrong_type(self):
        with pytest.raises(ValueError, match=r"fft_padding must be a padding-mode string"):
            _resolve_periodic(42, 2)

    # ── grid-type derivation ──────────────────────────────────────────────────
    def test_grid_per_axis_string_mode(self):
        assert _grid_is_single_per_axis("single", (False, False)) == (True, True)
        assert _grid_is_single_per_axis("double", (True, True)) == (False, False)

    def test_grid_per_axis_mixed_mode(self):
        assert _grid_is_single_per_axis(None, (True, False)) == (True, False)
        assert _grid_is_single_per_axis(None, (False, True, True)) == (False, True, True)


# ---------------------------------------------------------------------------
# 2. Validation tests (CPU-only)
# ---------------------------------------------------------------------------


class TestValidation:
    """Bad argument combinations must raise loudly at __init__."""

    def test_per_axis_padding_requires_no_grid_type(self):
        with pytest.raises(ValueError, match=r"grid_type must be None"):
            _make_ckconv(data_dim=2, fft_padding="circular, zero", grid_type="single")

    def test_string_padding_requires_grid_type(self):
        with pytest.raises(AssertionError, match=r"Invalid grid type"):
            _make_ckconv(data_dim=2, fft_padding="zero", grid_type=None)

    def test_comma_form_wrong_length(self):
        with pytest.raises(ValueError, match=r"exactly data_dim=2"):
            _make_ckconv(data_dim=2, fft_padding="circular, zero, circular")

    def test_sequence_form_wrong_length(self):
        with pytest.raises(ValueError, match=r"length data_dim=2"):
            _make_ckconv(data_dim=2, fft_padding=["circular", "zero", "circular"])

    def test_invalid_mode_string(self):
        with pytest.raises(ValueError, match=r"Invalid padding mode 'weird'"):
            _make_ckconv(data_dim=2, fft_padding="weird", grid_type="single")

    def test_bool_sequence_rejected_with_redirect(self):
        # The old (True, False) form is no longer accepted — the error must
        # redirect to the string API.
        with pytest.raises(ValueError, match=r"no longer accepts a sequence of booleans"):
            _make_ckconv(data_dim=2, fft_padding=(True, False))

    def test_causal_with_periodic_axis_raises(self):
        # 1D causal cannot combine with a periodic axis.
        kernel_cfg = _make_kernel_cfg(data_dim=1)
        with pytest.raises(ValueError, match=r"is_causal=True is incompatible"):
            CKConvND(
                data_dim=1,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=kernel_cfg,
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type=None,
                fft_padding=["circular"],
                is_causal=True,
            )

    def test_subq_ops_with_per_axis_padding_raises(self):
        with pytest.raises(ValueError, match=r"does not support a per-axis fft_padding"):
            CKConvND(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=_make_kernel_cfg(data_dim=2),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type=None,
                fft_padding="circular, zero",
                fft_backend="subq_ops",
            )

    def test_fp16_with_per_axis_padding_raises(self):
        with pytest.raises(NotImplementedError, match=r"use_fp16_fft is not supported"):
            CKConvND(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=_make_kernel_cfg(data_dim=2),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type=None,
                fft_padding="circular, zero",
                use_fp16_fft=True,
            )

    def test_existing_circular_with_double_grid_still_rejected(self):
        # Legacy single-mode constraint untouched.
        with pytest.raises(AssertionError, match=r"requires grid_type='single'"):
            _make_ckconv(data_dim=2, fft_padding="circular", grid_type="double")


# ---------------------------------------------------------------------------
# 3. Per-axis kernel size (no CUDA needed: we just construct the kernel)
# ---------------------------------------------------------------------------


class TestPerAxisKernelSize:
    """The auto-derived per-axis grid produces the expected SIREN kernel shape.

    The SIREN positional embedding produces ``2*L - 1`` points per axis given
    ``L = grid_lens[d]``. With CKConvND:

    - **Periodic axis** (single grid): ``L_d = (N_d + 1) // 2`` ⇒ kernel size
      ``2 * ((N_d + 1) // 2) - 1`` = ``N_d`` for odd ``N_d``, ``N_d - 1`` for even.
    - **Non-periodic axis** (double grid): ``L_d = N_d`` ⇒ kernel size ``2 * N_d - 1``.

    Spatial ``16`` is even, so periodic axes produce ``2 * 8 - 1 = 15``.
    Spatial ``8`` is even, so periodic axes produce ``2 * 4 - 1 = 7``.
    """

    @pytest.mark.parametrize(
        "periodic,spatial,expected_kernel_shape",
        [
            # 2D, mixed: periodic on x → K_x = 15; linear on y → K_y = 31.
            ((True, False), (16, 16), (1, 15, 31, HIDDEN_DIM)),
            # 2D, mixed: linear on x → K_x = 31; periodic on y → K_y = 15.
            ((False, True), (16, 16), (1, 31, 15, HIDDEN_DIM)),
            # 2D, all-False: both axes double-grid → K = 31 each.
            ((False, False), (16, 16), (1, 31, 31, HIDDEN_DIM)),
            # 2D, all-True: both axes single-grid → K = 15 each (even N).
            ((True, True), (16, 16), (1, 15, 15, HIDDEN_DIM)),
            # 3D, mixed.
            ((True, True, False), (8, 8, 8), (1, 7, 7, 15, HIDDEN_DIM)),
            ((False, True, True), (8, 8, 8), (1, 15, 7, 7, HIDDEN_DIM)),
        ],
    )
    def test_kernel_shape(self, periodic, spatial, expected_kernel_shape):
        data_dim = len(periodic)
        # CPU is sufficient — the kernel is a small SIREN MLP.
        conv = CKConvND(
            data_dim=data_dim,
            hidden_dim=HIDDEN_DIM,
            kernel_cfg=_make_kernel_cfg(data_dim=data_dim, L_cache=max(spatial)),
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type=None,
            fft_padding=_periodic_to_fft_padding(periodic),
        )

        # Compute the kernel via the same path forward() uses.
        is_single = _grid_is_single_per_axis(None, periodic)
        grid_lens = tuple((s + 1) // 2 if g_single else s for s, g_single in zip(spatial, is_single))
        with torch.no_grad():
            kernel, _ = conv.kernel(list(grid_lens))
        assert tuple(kernel.shape) == expected_kernel_shape, (
            f"kernel shape mismatch: got {tuple(kernel.shape)}, expected {expected_kernel_shape}"
        )


# ---------------------------------------------------------------------------
# 4. Bit-identical equivalence: tuple "all-False"/"all-True" vs legacy strings
# ---------------------------------------------------------------------------


@requires_cuda
class TestEquivalenceWithLegacy:
    """Tuple ``(False,...,False)`` / ``(True,...,True)`` produces bit-identical
    output to the corresponding legacy string mode. This proves the legacy
    code path is untouched."""

    @pytest.mark.parametrize("data_dim", [1, 2, 3])
    def test_uniform_zero_per_axis_matches_zero_single_mode(self, data_dim):
        # A per-axis "zero, zero, ..." form must produce bit-identical output
        # to the legacy single-mode fft_padding="zero" with grid_type="double".
        torch.manual_seed(0)
        legacy = _make_ckconv(data_dim=data_dim, fft_padding="zero", grid_type="double").cuda()
        per_axis = _make_ckconv(
            data_dim=data_dim,
            fft_padding=_periodic_to_fft_padding((False,) * data_dim),
        ).cuda()
        _sync_weights(legacy, per_axis)

        x_shape = (2, *([SPATIAL] * data_dim), HIDDEN_DIM)
        x = torch.randn(*x_shape, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(per_axis(x), legacy(x), rtol=0, atol=0)

    @pytest.mark.parametrize("data_dim", [1, 2, 3])
    def test_uniform_circular_per_axis_matches_circular_single_mode(self, data_dim):
        # Same equivalence for the all-circular case.
        torch.manual_seed(0)
        legacy = _make_ckconv(data_dim=data_dim, fft_padding="circular", grid_type="single").cuda()
        per_axis = _make_ckconv(
            data_dim=data_dim,
            fft_padding=_periodic_to_fft_padding((True,) * data_dim),
        ).cuda()
        _sync_weights(legacy, per_axis)

        x_shape = (2, *([SPATIAL] * data_dim), HIDDEN_DIM)
        x = torch.randn(*x_shape, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(per_axis(x), legacy(x), rtol=0, atol=0)

    def test_comma_string_and_list_forms_are_equivalent(self):
        # The two per-axis input forms (comma-separated string vs sequence of
        # mode strings) must produce identical results.
        torch.manual_seed(0)
        as_string = _make_ckconv(data_dim=2, fft_padding="circular, zero").cuda()
        as_list = _make_ckconv(data_dim=2, fft_padding=["circular", "zero"]).cuda()
        _sync_weights(as_string, as_list)

        x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(as_list(x), as_string(x), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 5. Mixed-mode forward correctness
# ---------------------------------------------------------------------------


@requires_cuda
class TestMixedForward:
    """In mixed mode the CKConvND output equals the direct mixed_fftconv op
    applied to the same kernel that CKConvND constructs.

    This separates 'is the dispatch wired correctly?' (this test) from
    'is the mixed op itself correct?' (covered by tests/ops/test_mixed_fftconv.py).
    """

    @pytest.mark.parametrize(
        "periodic",
        [(True, False), (False, True)],
    )
    def test_mixed_2d_matches_direct_op(self, periodic):
        """CKConvND mixed dispatch produces bit-identical output to calling the
        underlying op with the same kernel.

        We call the **same** op (``_w_reshape`` BLH wrapper) the module uses;
        cuFFT can choose different plans for permuted vs contiguous data, so
        using the BHL op directly would introduce ULP-level diffs that aren't
        a wiring bug — they're an FFT-plan artifact.
        """
        from nvsubquadratic.ops.mixed_fftconv import mixed_fftconv2d_fp32_bhl_w_reshape

        torch.manual_seed(0)
        conv = _make_ckconv(data_dim=2, fft_padding=_periodic_to_fft_padding(periodic)).cuda()

        x = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda", dtype=torch.float32)
        y_module = conv(x)

        # Re-derive what the module sees: kernel + shortcut, then call the same
        # BLH op the module wrapped via _wrap_mixed_op.
        is_single = _grid_is_single_per_axis(None, periodic)
        grid_lens = [(s + 1) // 2 if g else s for s, g in zip(x.shape[1:-1], is_single)]
        with torch.no_grad():
            kernel_blh, _ = conv.kernel(grid_lens)
        y_direct = mixed_fftconv2d_fp32_bhl_w_reshape(x, kernel_blh, periodic, conv.shortcut)
        torch.testing.assert_close(y_module, y_direct, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "periodic",
        [(True, True, False), (True, False, True), (False, True, True)],
    )
    def test_mixed_3d_matches_direct_op(self, periodic):
        from nvsubquadratic.ops.mixed_fftconv import mixed_fftconv3d_fp32_bhl_w_reshape

        torch.manual_seed(0)
        conv = _make_ckconv(data_dim=3, fft_padding=_periodic_to_fft_padding(periodic), L_cache=8).cuda()

        x = torch.randn(1, 8, 8, 8, HIDDEN_DIM, device="cuda", dtype=torch.float32)
        y_module = conv(x)

        is_single = _grid_is_single_per_axis(None, periodic)
        grid_lens = [(s + 1) // 2 if g else s for s, g in zip(x.shape[1:-1], is_single)]
        with torch.no_grad():
            kernel_blh, _ = conv.kernel(grid_lens)
        y_direct = mixed_fftconv3d_fp32_bhl_w_reshape(x, kernel_blh, periodic, conv.shortcut)
        torch.testing.assert_close(y_module, y_direct, rtol=0, atol=0)

    @pytest.mark.parametrize("periodic", [(True, False), (False, True)])
    def test_mixed_bhl_input(self, periodic):
        torch.manual_seed(0)
        conv = _make_ckconv(data_dim=2, fft_padding=_periodic_to_fft_padding(periodic)).cuda()
        x_blh = torch.randn(2, SPATIAL, SPATIAL, HIDDEN_DIM, device="cuda", dtype=torch.float32)
        x_bhl = x_blh.permute(0, 3, 1, 2).contiguous()
        y_blh = conv(x_blh, is_bhl_input=False)
        y_bhl = conv(x_bhl, is_bhl_input=True)
        torch.testing.assert_close(y_blh.permute(0, 3, 1, 2).contiguous(), y_bhl, rtol=0, atol=0)

    @pytest.mark.parametrize("periodic", [(True, False), (False, True), (True, True), (False, False)])
    def test_mixed_chunked_matches_non_chunked(self, periodic):
        torch.manual_seed(0)
        # Make sure chunking is exercised: hidden_dim > chunk_size.
        # The default chunk_size for the mixed-chunked op is 128, so increase
        # hidden_dim above it to force chunking.
        hidden = 192
        kernel_cfg = LazyConfig(SIRENKernelND)(
            data_dim=2,
            out_dim=hidden,
            mlp_hidden_dim=16,
            num_layers=2,
            embedding_dim=16,
            omega_0=10.0,
            L_cache=SPATIAL,
            use_bias=True,
        )

        def _make(use_chunked: bool):
            return CKConvND(
                data_dim=2,
                hidden_dim=hidden,
                kernel_cfg=kernel_cfg,
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type=None,
                fft_padding=_periodic_to_fft_padding(periodic),
                use_chunked_fftconv=use_chunked,
            )

        std = _make(False).cuda()
        chunked = _make(True).cuda()
        _sync_weights(std, chunked)

        x = torch.randn(2, SPATIAL, SPATIAL, hidden, device="cuda", dtype=torch.float32)
        y_std = std(x)
        y_chunked = chunked(x)
        torch.testing.assert_close(y_chunked, y_std, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 6. flop_count uses per-axis padded sizes
# ---------------------------------------------------------------------------


class TestFLOPAccounting:
    """``flop_count`` reflects per-axis FFT padded sizes in mixed mode.

    For a mixed config with ``fft_padding="circular, zero"`` (periodic on
    x, zero-padded on y) on a square input, the padded sizes should be
    ``(N, min(N + (K+1)//2, 2N))`` — strictly less than the all-zero
    ``(min(N + ...), min(N + ...))`` and strictly more than the
    all-circular ``(N, N)``.
    """

    def test_mixed_is_between_circular_and_zero(self):
        spatial = (SPATIAL, SPATIAL)
        zero = _make_ckconv(data_dim=2, fft_padding="zero", grid_type="double")
        circ = _make_ckconv(data_dim=2, fft_padding="circular", grid_type="single")
        mixed = _make_ckconv(data_dim=2, fft_padding="circular, zero")

        flops_zero = zero.flop_count(spatial)
        flops_circ = circ.flop_count(spatial)
        flops_mixed = mixed.flop_count(spatial)

        # Mixed (one periodic, one zero-padded) sits strictly between the
        # two uniform extremes: more FFT work than all-periodic (because the
        # non-periodic axis is padded up), less than all-zero-padded.
        assert flops_circ < flops_mixed < flops_zero, (
            f"Expected circ < mixed < zero, got circ={flops_circ}, mixed={flops_mixed}, zero={flops_zero}"
        )

    def test_uniform_zero_per_axis_matches_zero_single_mode_flops(self):
        spatial = (SPATIAL, SPATIAL)
        legacy = _make_ckconv(data_dim=2, fft_padding="zero", grid_type="double")
        per_axis = _make_ckconv(data_dim=2, fft_padding="zero, zero")
        assert per_axis.flop_count(spatial) == legacy.flop_count(spatial)

    def test_uniform_circular_per_axis_matches_circular_single_mode_flops(self):
        spatial = (SPATIAL, SPATIAL)
        legacy = _make_ckconv(data_dim=2, fft_padding="circular", grid_type="single")
        per_axis = _make_ckconv(data_dim=2, fft_padding="circular, circular")
        assert per_axis.flop_count(spatial) == legacy.flop_count(spatial)
