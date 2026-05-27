# TODO: Add license header here


"""CKConv (long-convolution) implementation for ND signals."""

import copy
import inspect
import math
import warnings
from collections.abc import Sequence
from typing import Literal

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, _resolve_target, instantiate
from nvsubquadratic.modules.kernels_nd import _normalize_l_cache

# Standard FFT convolutions
from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_fp32_bhl,
    circular_fftconv1d_fp32_bhl_w_reshape,
    circular_fftconv2d_fp32_bhl,
    circular_fftconv2d_fp32_bhl_w_reshape,
    circular_fftconv3d_fp32_bhl,
    circular_fftconv3d_fp32_bhl_w_reshape,
)

# FP16 circular FFT convolutions (requires power-of-2 spatial dimensions)
from nvsubquadratic.ops.circular_fftconv_fp16 import (
    circular_fftconv1d_fp16_bhl,
    circular_fftconv1d_fp16_bhl_w_reshape,
    circular_fftconv2d_fp16_bhl,
    circular_fftconv2d_fp16_bhl_w_reshape,
    circular_fftconv3d_fp16_bhl,
    circular_fftconv3d_fp16_bhl_w_reshape,
)
from nvsubquadratic.ops.fftconv import (
    causal_fftconv1d_fp32_bhl,
    causal_fftconv1d_fp32_bhl_w_reshape,
    fftconv1d_fp32_bhl,
    fftconv1d_fp32_bhl_w_reshape,
    fftconv2d_fp32_bhl,
    fftconv2d_fp32_bhl_w_reshape,
    fftconv3d_fp32_bhl,
    fftconv3d_fp32_bhl_w_reshape,
)

# Chunked (memory-efficient) variants for zero-padded and causal convolutions
# Note: circular convolutions don't have chunked variants (lower memory overhead already)
from nvsubquadratic.ops.fftconv_chunked import (
    causal_fftconv1d_fp32_bhl as causal_fftconv1d_fp32_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    causal_fftconv1d_fp32_bhl_w_reshape as causal_fftconv1d_fp32_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv1d_fp32_bhl as fftconv1d_fp32_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv1d_fp32_bhl_w_reshape as fftconv1d_fp32_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv2d_fp32_bhl as fftconv2d_fp32_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv2d_fp32_bhl_w_reshape as fftconv2d_fp32_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv3d_fp32_bhl as fftconv3d_fp32_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv3d_fp32_bhl_w_reshape as fftconv3d_fp32_bhl_w_reshape_chunked,
)

# FP16 FFT convolutions (power-of-2 padding + ortho normalization)
from nvsubquadratic.ops.fftconv_fp16 import (
    causal_fftconv1d_fp16_bhl,
    causal_fftconv1d_fp16_bhl_chunked,
    causal_fftconv1d_fp16_bhl_w_reshape,
    causal_fftconv1d_fp16_bhl_w_reshape_chunked,
    fftconv1d_fp16_bhl,
    fftconv1d_fp16_bhl_chunked,
    fftconv1d_fp16_bhl_w_reshape,
    fftconv1d_fp16_bhl_w_reshape_chunked,
    fftconv2d_fp16_bhl,
    fftconv2d_fp16_bhl_chunked,
    fftconv2d_fp16_bhl_w_reshape,
    fftconv2d_fp16_bhl_w_reshape_chunked,
    fftconv3d_fp16_bhl,
    fftconv3d_fp16_bhl_chunked,
    fftconv3d_fp16_bhl_w_reshape,
    fftconv3d_fp16_bhl_w_reshape_chunked,
)

# Mixed boundary-condition FFT convolutions (per-axis periodic / non-periodic).
# Used when ``fft_padding`` is given as a list of mode strings (e.g.
# ``["circular", "zero"]``) rather than a single mode; the all-zero /
# all-circular corners dispatch internally to the legacy ops below to
# preserve bit-identical behavior.
from nvsubquadratic.ops.mixed_fftconv import (
    mixed_fftconv1d_fp32_bhl,
    mixed_fftconv1d_fp32_bhl_chunked,
    mixed_fftconv1d_fp32_bhl_w_reshape,
    mixed_fftconv1d_fp32_bhl_w_reshape_chunked,
    mixed_fftconv2d_fp32_bhl,
    mixed_fftconv2d_fp32_bhl_chunked,
    mixed_fftconv2d_fp32_bhl_w_reshape,
    mixed_fftconv2d_fp32_bhl_w_reshape_chunked,
    mixed_fftconv3d_fp32_bhl,
    mixed_fftconv3d_fp32_bhl_chunked,
    mixed_fftconv3d_fp32_bhl_w_reshape,
    mixed_fftconv3d_fp32_bhl_w_reshape_chunked,
)


# Mapping from padding mode and data dimensionality to FFT convolution functions.
# Each entry is a tuple: (fn_for_BLH_input (bhl + reshape), fn_for_BHL_input)
FFT_FUNCTIONS = {
    "circular": {
        1: (circular_fftconv1d_fp32_bhl_w_reshape, circular_fftconv1d_fp32_bhl),
        2: (circular_fftconv2d_fp32_bhl_w_reshape, circular_fftconv2d_fp32_bhl),
        3: (circular_fftconv3d_fp32_bhl_w_reshape, circular_fftconv3d_fp32_bhl),
    },
    "zero": {
        1: (fftconv1d_fp32_bhl_w_reshape, fftconv1d_fp32_bhl),
        2: (fftconv2d_fp32_bhl_w_reshape, fftconv2d_fp32_bhl),
        3: (fftconv3d_fp32_bhl_w_reshape, fftconv3d_fp32_bhl),
    },
    "causal": {
        1: (causal_fftconv1d_fp32_bhl_w_reshape, causal_fftconv1d_fp32_bhl),
        # Causal is only supported for 1D (sequences)
    },
}

# Chunked versions (memory-efficient, trades compute for lower peak memory)
# Note: circular convolutions don't have chunked variants - they already have lower
# memory overhead since they don't require padding.
FFT_FUNCTIONS_CHUNKED = {
    "zero": {
        1: (fftconv1d_fp32_bhl_w_reshape_chunked, fftconv1d_fp32_bhl_chunked),
        2: (fftconv2d_fp32_bhl_w_reshape_chunked, fftconv2d_fp32_bhl_chunked),
        3: (fftconv3d_fp32_bhl_w_reshape_chunked, fftconv3d_fp32_bhl_chunked),
    },
    "causal": {
        1: (causal_fftconv1d_fp32_bhl_w_reshape_chunked, causal_fftconv1d_fp32_bhl_chunked),
        # Causal is only supported for 1D (sequences)
    },
}

# FP16 versions (power-of-2 padding + ortho normalization to prevent overflow)
# Note: circular fp16 requires power-of-2 spatial dimensions (cuFFT constraint).
FFT_FUNCTIONS_FP16 = {
    "circular": {
        1: (circular_fftconv1d_fp16_bhl_w_reshape, circular_fftconv1d_fp16_bhl),
        2: (circular_fftconv2d_fp16_bhl_w_reshape, circular_fftconv2d_fp16_bhl),
        3: (circular_fftconv3d_fp16_bhl_w_reshape, circular_fftconv3d_fp16_bhl),
    },
    "zero": {
        1: (fftconv1d_fp16_bhl_w_reshape, fftconv1d_fp16_bhl),
        2: (fftconv2d_fp16_bhl_w_reshape, fftconv2d_fp16_bhl),
        3: (fftconv3d_fp16_bhl_w_reshape, fftconv3d_fp16_bhl),
    },
    "causal": {
        1: (causal_fftconv1d_fp16_bhl_w_reshape, causal_fftconv1d_fp16_bhl),
        # Causal is only supported for 1D (sequences)
    },
}

# FP16 + chunked: combines fp16 memory savings with channel-chunking savings
FFT_FUNCTIONS_FP16_CHUNKED = {
    "zero": {
        1: (fftconv1d_fp16_bhl_w_reshape_chunked, fftconv1d_fp16_bhl_chunked),
        2: (fftconv2d_fp16_bhl_w_reshape_chunked, fftconv2d_fp16_bhl_chunked),
        3: (fftconv3d_fp16_bhl_w_reshape_chunked, fftconv3d_fp16_bhl_chunked),
    },
    "causal": {
        1: (causal_fftconv1d_fp16_bhl_w_reshape_chunked, causal_fftconv1d_fp16_bhl_chunked),
        # Causal is only supported for 1D (sequences)
    },
}

# Mixed-BC FFT convolutions: only fp32 in v1 (see docs/ops/MIXED_BC_PLAN.md).
# Each entry is ``(fn_for_BLH_input (bhl_w_reshape), fn_for_BHL_input)`` and
# takes an additional ``periodic`` argument compared to the legacy ops; the
# wrapper ``_wrap_mixed_op`` below adapts the call signature.
MIXED_FFT_FUNCTIONS = {
    1: (mixed_fftconv1d_fp32_bhl_w_reshape, mixed_fftconv1d_fp32_bhl),
    2: (mixed_fftconv2d_fp32_bhl_w_reshape, mixed_fftconv2d_fp32_bhl),
    3: (mixed_fftconv3d_fp32_bhl_w_reshape, mixed_fftconv3d_fp32_bhl),
}

MIXED_FFT_FUNCTIONS_CHUNKED = {
    1: (mixed_fftconv1d_fp32_bhl_w_reshape_chunked, mixed_fftconv1d_fp32_bhl_chunked),
    2: (mixed_fftconv2d_fp32_bhl_w_reshape_chunked, mixed_fftconv2d_fp32_bhl_chunked),
    3: (mixed_fftconv3d_fp32_bhl_w_reshape_chunked, mixed_fftconv3d_fp32_bhl_chunked),
}


# Padding-mode strings accepted by ``fft_padding``. Map name → per-axis
# periodic flag (``True`` ⇒ circular conv on that axis).
_PADDING_MODE_TO_PERIODIC: dict[str, bool] = {"zero": False, "circular": True}


def _parse_padding_mode(mode: str) -> bool:
    """Map a single padding-mode string to its per-axis periodic flag."""
    normalised = mode.strip().lower()
    if normalised not in _PADDING_MODE_TO_PERIODIC:
        valid = sorted(_PADDING_MODE_TO_PERIODIC)
        raise ValueError(f"Invalid padding mode {mode!r}. Must be one of {valid}.")
    return _PADDING_MODE_TO_PERIODIC[normalised]


def _resolve_periodic(
    fft_padding: "str | Sequence[str]",
    data_dim: int,
) -> tuple[bool, ...]:
    """Normalise ``fft_padding`` to a per-axis tuple of booleans.

    Accepted forms:

    - **Single mode string** — applies to every axis:

      - ``"zero"``     → ``(False, ..., False)`` (length ``data_dim``).
      - ``"circular"`` → ``(True,  ..., True)``.

    - **Sequence of mode strings** — one mode per spatial axis, in order:

      - ``["circular", "zero"]`` → ``(True, False)`` for ``data_dim=2``.
      - ``["zero", "circular", "zero"]`` → ``(False, True, False)`` for ``data_dim=3``.
      - ``("circular", "zero")`` (tuple form) is equivalent.
      - Mode names are case-insensitive and whitespace-stripped, so
        ``[" Circular ", "ZERO"]`` works.

    Two input shapes that are deliberately **rejected**:

    - **Booleans** (e.g. ``True``, ``(True, False)``): the per-axis intent
      is not obvious from the boolean values; the error message redirects
      to the list-of-strings form.
    - **Comma-separated strings** (e.g. ``"circular, zero"``): the list
      form is unambiguous and reads the same in Python and OmegaConf, so
      we keep a single canonical per-axis form to avoid two ways of saying
      the same thing.

    Raises:
        ValueError: on invalid mode strings, wrong number of axes, or
            disallowed input types.
    """
    if isinstance(fft_padding, bool):
        raise ValueError(
            "fft_padding=True/False is not a valid input. Use 'zero' (all axes "
            "zero-padded), 'circular' (all axes periodic), or a per-axis list "
            "of mode strings such as ['circular', 'zero']."
        )

    if isinstance(fft_padding, str):
        if "," in fft_padding:
            raise ValueError(
                f"fft_padding does not accept comma-separated strings (got "
                f"{fft_padding!r}). For per-axis modes use a list, e.g. "
                f"['circular', 'zero']."
            )
        return (_parse_padding_mode(fft_padding),) * data_dim

    if isinstance(fft_padding, Sequence) and not isinstance(fft_padding, (str, bytes)):
        items = list(fft_padding)
        if any(isinstance(item, bool) for item in items):
            raise ValueError(
                "fft_padding no longer accepts a sequence of booleans (e.g. "
                "(True, False)) because the per-axis intent is not obvious "
                "from the boolean values. Use mode strings instead: "
                "['circular', 'zero'] for a 2D config with periodic x and "
                "zero-padded y."
            )
        if not all(isinstance(item, str) for item in items):
            raise ValueError(
                f"fft_padding sequence must contain only padding-mode strings "
                f"('zero' / 'circular'). Got: {fft_padding!r}."
            )
        if len(items) != data_dim:
            raise ValueError(
                f"fft_padding sequence must have length data_dim={data_dim}, got length {len(items)}: {fft_padding!r}."
            )
        return tuple(_parse_padding_mode(item) for item in items)

    raise ValueError(
        f"fft_padding must be a single mode string ('zero' / 'circular') or a "
        f"sequence of mode strings (one per spatial axis, e.g. "
        f"['circular', 'zero']). Got {fft_padding!r} "
        f"(type {type(fft_padding).__name__})."
    )


def _wrap_mixed_op(op_fn, periodic: tuple[bool, ...]):
    """Adapt a ``mixed_fftconv*`` op to the standard ``(x, kernel, shortcut)`` signature.

    The mixed ops take an extra positional ``periodic`` argument between
    ``kernel`` and ``shortcut``. ``CKConvND.apply_convolution`` calls the FFT
    function as ``fn(x, kernel, shortcut)``; this wrapper binds ``periodic``
    so the rest of the module is unchanged from the legacy ops.
    """

    def _wrapped(x, kernel, shortcut):
        return op_fn(x, kernel, periodic, shortcut)

    return _wrapped


def _grid_is_single_per_axis(
    grid_type: "Literal['double', 'single'] | None",
    periodic: tuple[bool, ...],
) -> tuple[bool, ...]:
    """Return per-axis 'use single grid' flags for the SIREN kernel.

    In CKConvND, ``grid_type='single'`` means the SIREN kernel grid spans
    ``(N+1)//2`` per axis so the produced kernel size equals the input size
    on that axis (paired with periodic/circular FFT conv). ``'double'``
    means the kernel grid spans ``N`` so the kernel covers twice the input
    (paired with zero-padded FFT conv).

    - **String mode** (``grid_type`` is ``'single'`` / ``'double'``): the same
      choice applies to every axis.
    - **Tuple / mixed mode** (``grid_type is None``): the per-axis flag is
      auto-derived as ``periodic[d]`` (periodic axis ⇒ single grid; non-
      periodic ⇒ double grid). This matches the recipe in
      :func:`nvsubquadratic.ops.mixed_fftconv._mixed_recipe`.
    """
    if grid_type is None:
        return tuple(periodic)
    return (grid_type == "single",) * len(periodic)


class CKConvND(torch.nn.Module):
    """CKConv (long-convolution) implementation for ND signals."""

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        kernel_cfg: LazyConfig,
        mask_cfg: LazyConfig,
        grid_type: "Literal['double', 'single'] | None",
        fft_padding: "Literal['zero', 'circular'] | str | Sequence[str]",
        is_causal: bool = False,
        use_chunked_fftconv: bool = False,
        use_fp16_fft: bool = False,
        fft_backend: Literal["torch_fft", "subq_ops"] = "torch_fft",
    ):
        """Initialize the CKConvND.

        Args:
            data_dim: Dimension of input data (1D for sequences, 2D for images, 3D for videos, etc.).
            hidden_dim: Hidden dimension.
            kernel_cfg: LazyConfig for the kernel.
            mask_cfg: LazyConfig for the mask.
            grid_type: How the SIREN kernel grid relates to the input size.
                ``"single"`` ⇒ kernel size == input size (paired with periodic
                FFT convolution). ``"double"`` ⇒ kernel size == 2*input size
                (paired with zero-padded FFT convolution).
                **Must be ``None`` (or omitted)** when ``fft_padding`` is in
                per-axis form — in that case the grid type is auto-derived per
                axis (``"single"`` on periodic axes, ``"double"`` on non-periodic
                axes). Required when ``fft_padding`` is a single mode string.
            fft_padding: Boundary behavior of the FFT convolution. Accepts:

                - ``"zero"``     — every axis zero-padded ("same" linear conv).
                - ``"circular"`` — every axis periodic (wrap-around conv).
                - **List/tuple of mode strings** — one per spatial axis, in
                  order, e.g. ``["circular", "zero"]`` for a 2D config that is
                  periodic on x and zero-padded on y. Mode names are
                  case-insensitive and whitespace-stripped.

                Required for datasets with mixed periodic/non-periodic
                boundary conditions (e.g. Well's ``rayleigh_benard``,
                ``viscoelastic_instability``). When supplied as a per-axis
                list, the ``grid_type`` argument must be ``None``.
                Must be ``"zero"`` (or an all-``"zero"`` list) when
                ``is_causal=True``.
            is_causal: If True, use causal (left-only) convolution where output at position i
                only depends on inputs at positions 0, 1, ..., i. Only supported for 1D data.
            use_chunked_fftconv: If True, use memory-efficient chunked FFT convolutions.
                Processes channels in chunks to reduce peak memory from complex FFT
                intermediates. Typical savings: ~26% memory with ~11% compute overhead.
                Useful for memory-constrained training with large spatial dimensions
                in 2D/3D. Default is False.
            use_fp16_fft: If True, use fp16 FFT convolutions. Uses ortho
                normalization to prevent overflow. Saves ~36% peak memory per
                convolution with ~0.8% mean relative error vs f32. For zero/causal
                padding, sizes are auto-padded to power-of-2. For circular padding,
                the input spatial dimensions must already be powers of 2 (a runtime
                assertion will fire otherwise). Default is False.
                Not supported with a per-axis ``fft_padding`` in v1 (the fp16
                mixed op is a planned follow-up — see ``docs/ops/MIXED_BC_PLAN.md``).
            fft_backend: FFT convolution backend to use. ``'torch_fft'`` (default)
                uses the torch.fft-based implementations. ``'subq_ops'`` uses the
                optimized CUDA kernels from ``subquadratic_ops_torch``. The subq_ops
                backend currently supports:
                  - 2D, zero-padded, non-causal convolutions
                  - 1D causal convolutions (``data_dim=1`` + ``is_causal=True``)
                It does not support fp16 FFT. It supports chunked convolutions via
                channel-wise chunking.  Per-sample (FiLM) weights are supported on
                the 2D path only; the 1D causal CUDA kernel does not accept batched
                weights.
        """
        assert fft_backend in ["torch_fft", "subq_ops"], (
            f"Invalid fft_backend: {fft_backend!r}. Must be 'torch_fft' or 'subq_ops'."
        )

        # ---- Normalise fft_padding & grid_type --------------------------------
        # The per-axis form is a sequence of mode strings (e.g.
        # ["circular", "zero"]). NOTE: we deliberately use ``Sequence`` rather
        # than ``(list, tuple)`` because OmegaConf wraps Python lists as
        # ``ListConfig``, which is *not* a ``list`` subclass; configs flowing
        # through LazyConfig would otherwise hit the legacy single-mode path
        # and trip the ``grid_type`` assertion. The legacy single-mode string
        # form ("zero" / "circular") still requires the user to supply
        # ``grid_type``.
        _periodic = _resolve_periodic(fft_padding, data_dim)
        _is_tuple_mode = isinstance(fft_padding, Sequence) and not isinstance(fft_padding, (str, bytes))

        if _is_tuple_mode:
            if grid_type is not None:
                raise ValueError(
                    "grid_type must be None (or omitted) when fft_padding is a "
                    "per-axis list of mode strings. The per-axis grid is "
                    "auto-derived ('single' on periodic axes, 'double' on "
                    "non-periodic axes). "
                    f"Got grid_type={grid_type!r}, fft_padding={fft_padding!r}."
                )
        else:
            assert grid_type in ["double", "single"], (
                f"Invalid grid type: {grid_type}. Must be 'double' or 'single' "
                f"when fft_padding is a single mode string."
            )

        # Stash the per-axis tuple (single source of truth for forward + flop_count).
        # In legacy string mode this is still a uniform all-True / all-False tuple,
        # but the dispatch below picks legacy ops directly (``_is_tuple_mode=False``).
        # In tuple mode (any uniformity), the mixed op handles the dispatch — it
        # internally calls the legacy linear/circular ops bit-identically for the
        # uniform corners and the mixed core path for everything else.
        self_periodic_per_axis = _periodic

        # ---- Causal / mixed-BC compatibility ---------------------------------
        if is_causal:
            assert data_dim == 1, f"Causal CKConvND only supports 1D inputs. Got {data_dim}D."
            if _is_tuple_mode:
                # The mixed_fftconv* ops implement non-causal linear/circular
                # convolution; there is no causal mixed path. Falling through
                # silently would dispatch to the non-causal op and produce
                # output that leaks future positions. Use fft_padding="zero"
                # (single-mode string) for 1D causal.
                raise ValueError(
                    "is_causal=True is not supported with a per-axis fft_padding "
                    "list. Use fft_padding='zero' (single-mode string) for 1D "
                    f"causal. Got fft_padding={fft_padding!r}."
                )
            if any(_periodic):
                raise ValueError(
                    "is_causal=True is incompatible with periodic FFT padding. "
                    f"Got periodic={_periodic} (from fft_padding={fft_padding!r})."
                )

        # ---- Circular / chunked legacy constraints ---------------------------
        # The legacy circular path requires single grid; check that here (only
        # applies in string mode; the mixed path auto-handles per-axis grids).
        if not _is_tuple_mode and fft_padding == "circular":
            assert grid_type == "single", (
                "fft_padding='circular' requires grid_type='single' (kernel size equals input size)."
            )
            assert not use_chunked_fftconv, (
                "use_chunked_fftconv=True is not supported with fft_padding='circular'. "
                "Chunked FFT convolutions are only implemented for 'zero' padding (and 'causal' 1D). "
                "Circular convolutions already have lower memory overhead due to no padding."
            )

        # ---- fp16 + mixed-BC: not supported in v1 -----------------------------
        if use_fp16_fft and _is_tuple_mode:
            raise NotImplementedError(
                "use_fp16_fft is not supported with a per-axis fft_padding in v1. "
                "Either drop the fp16 flag or use a uniform 'zero'/'circular' fft_padding. "
                "See docs/ops/MIXED_BC_PLAN.md (§4.2) for the planned fp16 mixed op."
            )

        if use_fp16_fft and not _is_tuple_mode and fft_padding == "circular":
            warnings.warn(
                "use_fp16_fft with circular padding requires power-of-2 spatial "
                "dimensions (cuFFT fp16 constraint). A runtime assertion will fire "
                "if the input is not power-of-2.",
                stacklevel=2,
            )

        # subq_ops backend constraints
        if fft_backend == "subq_ops":
            if _is_tuple_mode:
                raise ValueError(
                    "fft_backend='subq_ops' does not support a per-axis fft_padding. "
                    "The CUDA kernel implements zero-padded conv only. "
                    "Use fft_backend='torch_fft' for mixed boundary conditions."
                )
            if data_dim == 1:
                assert is_causal, (
                    "fft_backend='subq_ops' on 1D requires is_causal=True "
                    "(no non-causal 1D CUDA kernel is wired). Got is_causal=False."
                )
            elif data_dim == 2:
                assert not is_causal, (
                    "fft_backend='subq_ops' on 2D does not support causal convolutions (causal is 1D only)."
                )
                assert fft_padding == "zero", (
                    "fft_backend='subq_ops' on 2D only supports zero-padded convolutions. "
                    f"Got fft_padding='{fft_padding}'."
                )
            else:
                raise AssertionError(
                    f"fft_backend='subq_ops' only supports data_dim in (1, 2). Got data_dim={data_dim}."
                )
            assert not use_fp16_fft, (
                "fft_backend='subq_ops' does not support fp16 FFT — the CUDA kernel "
                "manages its own precision internally. Use use_fp16_fft=False."
            )

        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.fft_padding = fft_padding
        self.is_causal = is_causal
        self.use_chunked_fftconv = use_chunked_fftconv
        self.use_fp16_fft = use_fp16_fft
        self.fft_backend = fft_backend
        # Per-axis BC: single source of truth used by forward() and flop_count().
        # Always present (length == data_dim), even in legacy single-mode form.
        self._periodic_per_axis: tuple[bool, ...] = self_periodic_per_axis
        # When the user supplies fft_padding as a per-axis list of mode strings
        # (e.g. ["circular", "zero"]), we dispatch through the unified
        # mixed_fftconv* ops for every combination of per-axis BCs. The mixed op
        # auto-routes to the legacy linear/circular ops internally for the
        # uniform corners, preserving bit-identical results for those cases.
        self._is_tuple_mode: bool = _is_tuple_mode

        # When the SIREN kernel grid is "single" on an axis, ``grid_lens`` is
        # halved on that axis relative to ``spatial_dims`` (see forward()).
        # We pre-adjust ``L_cache`` so that the positional-embedding grid_cache
        # spans [-1, 1] for the actual kernel size on each axis instead of a
        # truncated subrange.  ``L_cache`` may be a scalar int (isotropic) or a
        # sequence of length ``data_dim`` (anisotropic).
        L_cache_raw = getattr(kernel_cfg, "L_cache", None)
        effective_L_per_axis: tuple[int, ...] | None = None
        if L_cache_raw is not None:
            effective_L_per_axis = _normalize_l_cache(L_cache_raw, data_dim)
            is_single_per_axis = _grid_is_single_per_axis(grid_type, self._periodic_per_axis)
            if any(is_single_per_axis):
                # Deepcopy before mutating so shared config objects aren't corrupted.
                kernel_cfg = copy.deepcopy(kernel_cfg)
                effective_L_per_axis = tuple(
                    (L + 1) // 2 if is_single else L for L, is_single in zip(effective_L_per_axis, is_single_per_axis)
                )
                # Pass the new L_cache back in the same form the user supplied
                # (scalar in / scalar out, sequence in / sequence out) so config
                # serialization round-trips cleanly. In the mixed-mode case the
                # per-axis L_cache may be anisotropic, so a scalar input must
                # be promoted to a list.
                effective_is_anisotropic = len(set(effective_L_per_axis)) > 1
                if (
                    isinstance(L_cache_raw, Sequence) and not isinstance(L_cache_raw, (str, bytes))
                ) or effective_is_anisotropic:
                    kernel_cfg.L_cache = list(effective_L_per_axis)
                else:
                    kernel_cfg.L_cache = int(effective_L_per_axis[0])

        # Inject the actual kernel size into mask_cfg so that attenuation-based
        # initialization (GaussianModulationND) uses the correct grid geometry.
        # The mask is intentionally isotropic here (one ``grid_size`` shared
        # across axes); we feed it the *largest* per-axis kernel size so the
        # narrowest reachable Gaussian bandwidth (``min_std`` from
        # ``min_attenuation_at_step``) stays achievable on the highest-resolution
        # axis.  Per-axis bandwidth differences should be expressed via the
        # mask's per-axis ``init_extent`` instead.
        if effective_L_per_axis is not None:
            mask_target = _resolve_target(mask_cfg["__target__"]) if "__target__" in mask_cfg else None
            if mask_target is not None and "grid_size" in inspect.signature(mask_target).parameters:
                # Deepcopy before mutating so shared config objects aren't corrupted.
                mask_cfg = copy.deepcopy(mask_cfg)
                mask_cfg.grid_size = 2 * max(effective_L_per_axis) - 1

        # Construct kernel and mask
        self.kernel = instantiate(kernel_cfg)
        self.mask = instantiate(mask_cfg)

        # Construct shortcut projection
        self.shortcut = torch.nn.Parameter(torch.empty(hidden_dim))
        bounds = math.sqrt(1.0 / hidden_dim)
        self.shortcut.data.uniform_(-bounds, bounds)

        # Select FFT convolution functions based on backend
        if fft_backend == "subq_ops":
            if data_dim == 1:
                # 1D causal path (gated by the constraint block above).
                from nvsubquadratic.ops.fftconv_custom import (
                    causal_fftconv1d_bhl,
                    causal_fftconv1d_bhl_chunked,
                    causal_fftconv1d_bhl_w_reshape,
                    causal_fftconv1d_bhl_w_reshape_chunked,
                )

                if use_chunked_fftconv:
                    self.fftconv_fn = causal_fftconv1d_bhl_w_reshape_chunked
                    self.fftconv_fn_bhl_input = causal_fftconv1d_bhl_chunked
                else:
                    self.fftconv_fn = causal_fftconv1d_bhl_w_reshape
                    self.fftconv_fn_bhl_input = causal_fftconv1d_bhl
            elif data_dim == 2:
                from nvsubquadratic.ops.fftconv_custom import (
                    fftconv2d_bhl,
                    fftconv2d_bhl_chunked,
                    fftconv2d_bhl_w_reshape,
                    fftconv2d_bhl_w_reshape_chunked,
                )

                if use_chunked_fftconv:
                    self.fftconv_fn = fftconv2d_bhl_w_reshape_chunked
                    self.fftconv_fn_bhl_input = fftconv2d_bhl_chunked
                else:
                    self.fftconv_fn = fftconv2d_bhl_w_reshape
                    self.fftconv_fn_bhl_input = fftconv2d_bhl
            else:
                raise AssertionError(
                    f"fft_backend='subq_ops' dispatch reached unexpected data_dim={data_dim}; "
                    "the constraint block above should have rejected this."
                )
        elif self._is_tuple_mode:
            # Per-axis ``fft_padding`` (list of mode strings, e.g.
            # ["circular", "zero"]): route through the unified
            # mixed_fftconv* ops with ``periodic`` bound via _wrap_mixed_op so
            # the rest of CKConvND can keep calling the FFT function with the
            # (x, kernel, shortcut) signature used by every legacy op. The
            # all-zero / all-circular corners are dispatched internally to
            # the legacy linear / circular ops bit-identically (see
            # _dispatch_legacy_if_uniform in mixed_fftconv.py).
            mixed_table = MIXED_FFT_FUNCTIONS_CHUNKED if use_chunked_fftconv else MIXED_FFT_FUNCTIONS
            try:
                fn_w_reshape, fn_bhl = mixed_table[self.data_dim]
            except KeyError:
                valid_dims = sorted(mixed_table.keys())
                raise ValueError(
                    f"Mixed-BC FFT conv not implemented for data_dim={self.data_dim}. Valid: {valid_dims}"
                )
            self.fftconv_fn = _wrap_mixed_op(fn_w_reshape, self._periodic_per_axis)
            self.fftconv_fn_bhl_input = _wrap_mixed_op(fn_bhl, self._periodic_per_axis)
        else:
            # torch_fft backend, legacy single-mode string ("zero" / "circular";
            # uniform per-axis forms are taken care of in the branch above by
            # the mixed op's internal dispatch).
            # Causal mode overrides fft_padding for 1D.
            if is_causal:
                effective_padding = "causal"
            elif all(self._periodic_per_axis):
                effective_padding = "circular"
            else:
                effective_padding = "zero"

            # Choose FFT functions: fp16+chunked > fp16 > chunked > standard
            if use_fp16_fft and use_chunked_fftconv:
                fft_fn_table = FFT_FUNCTIONS_FP16_CHUNKED
            elif use_fp16_fft:
                fft_fn_table = FFT_FUNCTIONS_FP16
            elif use_chunked_fftconv:
                fft_fn_table = FFT_FUNCTIONS_CHUNKED
            else:
                fft_fn_table = FFT_FUNCTIONS
            try:
                self.fftconv_fn, self.fftconv_fn_bhl_input = fft_fn_table[effective_padding][self.data_dim]
            except KeyError:
                valid_dims = sorted(fft_fn_table.get(effective_padding, {}).keys())
                raise ValueError(
                    f"Unsupported configuration: fft_padding='{effective_padding}', data_dim={self.data_dim}. "
                    f"Valid dimensions for '{effective_padding}': {valid_dims}"
                )

        # Remember grid_type for forward() / flop_count() (None in mixed mode —
        # the per-axis grid is computed from ``self._periodic_per_axis`` via
        # ``_grid_is_single_per_axis``).
        self.grid_type = grid_type

    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        bc_repr = f"fft_padding={self.fft_padding!r}"
        if self._is_tuple_mode:
            bc_repr += f", periodic_per_axis={self._periodic_per_axis}"
        return (
            f"data_dim={self.data_dim}, hidden_dim={self.hidden_dim}, "
            f"{bc_repr}, grid_type={self.grid_type!r}, is_causal={self.is_causal}, "
            f"use_chunked_fftconv={self.use_chunked_fftconv}, use_fp16_fft={self.use_fp16_fft}, "
            f"fft_backend={self.fft_backend!r}"
        )

    def flop_count(self, spatial_dims: tuple[int, ...], inference: bool = False) -> int:
        """Count FLOPs for CKConv: kernel generation + FFT convolution.

        Two phases:

        **Phase 1 — Kernel generation** (via SIREN MLP):
          Delegated to ``self.kernel.flop_count(grid_lens, inference)``.
          At ``inference=True`` without FiLM, the kernel is input-independent
          and can be precomputed, so this returns 0.

        **Phase 2 — FFT-based depthwise convolution** (C = ``self.hidden_dim``):
          The convolution is computed in the frequency domain.  Padded signal
          sizes Np_i depend on the padding mode:
            - ``"zero"`` non-causal ("same"-mode):
                Np_i = min(s_i + (k_i + 1) // 2,  2 * s_i)
              Only half the kernel width of extra padding is needed beyond
              the input size, because the output is cropped back to input
              size (centered crop).  Matches ``fftconv.py`` line 624-628.
            - ``"zero"`` causal (1D only):
                Np_i = min(s_i + k_i,  2 * s_i)
              Full linear convolution length; output is tail-cropped.
            - ``"circular"``: Np_i = s_i  (wrap-around, no extra padding)

          A separable N-D FFT on a grid of size (Np_1, ..., Np_d) costs:
            5 * prod(Np) * sum(log2(Np_i))  real FLOPs per channel,
          based on the radix-2 Cooley-Tukey decomposition where each butterfly
          operation costs ~5 real FLOPs (1 complex multiply ≈ 4 real muls +
          2 real adds, minus shared twiddle-factor optimizations → ~5 ops).
          Note: the implementation uses ``rfft`` (real-to-complex), which is
          ~2x cheaper than a full complex FFT; the 5N log N formula is a
          conservative (upper-bound) estimate consistent with standard
          vision-paper conventions.

          Three FFTs are needed: forward FFT of input, forward FFT of kernel,
          and inverse FFT of the product.  At ``inference=True`` without FiLM,
          the kernel FFT is precomputed and cached, reducing to 2 FFTs.

          Pointwise complex multiply in the frequency domain:
            6 * C * prod(Np)  (4 real muls + 2 real adds for (a+bi)(c+di)).

          Shortcut (skip connection): C * prod(spatial_dims)  (elementwise).

        Args:
            spatial_dims: Spatial dimensions of the input signal, e.g. (H, W).
            inference: If True and kernel has no FiLM, skip kernel generation
                and kernel FFT (both are precomputable and cached).

        Returns:
            Total FLOPs as an integer.
        """
        C = self.hidden_dim
        has_film = getattr(self.kernel, "film_generator", None) is not None

        # Determine per-axis kernel grid_lens (same logic as forward).
        # In legacy string mode this is uniform; in mixed mode it's per-axis.
        is_single_per_axis = _grid_is_single_per_axis(self.grid_type, self._periodic_per_axis)
        grid_lens = tuple((s + 1) // 2 if is_single else s for s, is_single in zip(spatial_dims, is_single_per_axis))

        # Kernel spatial sizes: the SIREN generates on a (2*L - 1) grid per dim
        kernel_sizes = tuple(2 * gl - 1 for gl in grid_lens)

        # For causal 1D, kernel is cropped to second half
        if self.is_causal:
            kernel_sizes = tuple(ks // 2 + 1 for ks in kernel_sizes)

        flops = 0

        # Phase 1: Kernel generation
        flops += self.kernel.flop_count(grid_lens, inference=inference)

        # Phase 2: FFT convolution
        # Per-axis padded sizes match the actual fftconv implementations:
        #   periodic axis:                s  (no extra padding)
        #   non-periodic non-causal:      min(s + (k+1)//2, 2*s)
        #   causal (1D only, all axes):   min(s + k, 2*s)
        if self.is_causal:
            padded_dims = tuple(min(s + k, 2 * s) for s, k in zip(spatial_dims, kernel_sizes))
        else:
            padded_dims = tuple(
                s if is_periodic else min(s + (k + 1) // 2, 2 * s)
                for s, k, is_periodic in zip(spatial_dims, kernel_sizes, self._periodic_per_axis)
            )

        prod_padded = 1
        for p in padded_dims:
            prod_padded *= p
        log2_sum = sum(math.log2(max(p, 1)) for p in padded_dims)

        # 3 FFTs (input, kernel, inverse) normally;
        # 2 FFTs (input, inverse) at inference without FiLM (kernel FFT cached).
        num_ffts = 2 if (inference and not has_film) else 3
        fft_flops = num_ffts * 5 * C * prod_padded * log2_sum

        # Pointwise complex multiply in frequency domain
        cmul_flops = 6 * C * prod_padded

        # Shortcut (elementwise multiply: input * shortcut_weight)
        prod_spatial = 1
        for s in spatial_dims:
            prod_spatial *= s
        shortcut_flops = C * prod_spatial

        flops += int(fft_flops) + cmul_flops + shortcut_flops

        return flops

    def apply_convolution(
        self, x: torch.Tensor, conv_kernel: torch.Tensor, shortcut: torch.Tensor, is_bhl_input: bool
    ) -> torch.Tensor:
        """Apply the convolution operation using the FFT-based convolution function.

        Args:
            x (torch.Tensor): Input tensor.
            conv_kernel (torch.Tensor): Convolution kernel tensor.
            shortcut (torch.Tensor): Shortcut tensor.
            is_bhl_input (bool): Whether the input is in BHL format.

        Returns:
            torch.Tensor: Output tensor after applying convolution.
        """
        if is_bhl_input:
            conv_kernel = rearrange(
                conv_kernel, "b ... c -> b c ..."
            )  # Reshape kernel to [B, C, * spatial_dims] (Kernels are always in BLH format)
            _conv_fn = self.fftconv_fn_bhl_input
        else:
            _conv_fn = self.fftconv_fn

        return _conv_fn(x, conv_kernel, shortcut)

    def forward(
        self,
        x: torch.Tensor,
        is_bhl_input: bool = False,
        cp_group: torch.distributed.ProcessGroup = None,
        **mixer_kwargs,
    ) -> torch.Tensor:
        """Forward pass of the CKConvND.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, * spatial_dims, hidden_dim) or (batch_size, hidden_dim, * spatial_dims)
            is_bhl_input (bool): Whether the input is in BHL format, i.e., (batch_size, hidden_dim, * spatial_dims).
                Default is False.
            cp_group (torch.distributed.ProcessGroup): Context parallel process group.
                Default is None.
            **mixer_kwargs: Additional keyword arguments forwarded to the kernel generator
                (e.g. ``conditioning`` for FiLM-enabled SIRENKernelND).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, * spatial_dims, hidden_dim) or (batch_size, hidden_dim, * spatial_dims)
        """
        # Get the spatial dimensions from the input tensor
        if is_bhl_input:
            spatial_dims = x.shape[2:]  # [* spatial_dims]
        else:
            spatial_dims = x.shape[1:-1]  # [* spatial_dims]

        # Compute per-axis grid_lens. In legacy string mode the same choice
        # applies to every axis (uniform halving or no halving); in mixed mode
        # the choice is per-axis (single grid → halved on periodic axes, double
        # grid → full on non-periodic axes), matching the per-axis FFT recipe.
        is_single_per_axis = _grid_is_single_per_axis(self.grid_type, self._periodic_per_axis)
        grid_lens = [
            (seq_len + 1) // 2 if is_single else seq_len
            for seq_len, is_single in zip(spatial_dims, is_single_per_axis)
        ]

        # Compute kernel (pass conditioning if available for FiLM-enabled kernels)
        conditioning = mixer_kwargs.get("conditioning", None)
        conv_kernel, grid = self.kernel(grid_lens, conditioning=conditioning)

        # Apply mask to kernel
        if not isinstance(self.mask, torch.nn.Identity):
            conv_kernel = self.mask(grid=grid, x=conv_kernel)

        # For causal convolution, crop the kernel to use only the "positive" half
        # (i.e., the part that looks backward in time). The kernel is in BLH format: [1, L, H].
        # We keep positions from L//2 to L-1, which after the FFT flip becomes causal.
        if self.is_causal:
            # Kernel shape is [1, kernel_len, hidden_dim] for 1D
            # Crop to [1, kernel_len // 2, hidden_dim] keeping the second half
            kernel_len = conv_kernel.shape[-2]
            conv_kernel = conv_kernel[..., kernel_len // 2 :, :]

        # Handle context parallelism by slicing the kernel to match input channel dimensions
        if cp_group is not None and cp_group.size() > 1:
            if self.is_causal:
                raise ValueError("Causal CKConvND has not been verified to work with context parallelism.")
            cp_world_size = cp_group.size()
            cp_rank = cp_group.rank()

            # Get the channel dimension (last dimension in BLH format)
            kernel_channels = conv_kernel.shape[-1]
            channels_per_rank = kernel_channels // cp_world_size

            # Slice the kernel along the channel dimension for this CP rank
            start_idx = cp_rank * channels_per_rank
            end_idx = start_idx + channels_per_rank
            conv_kernel = conv_kernel[..., start_idx:end_idx]

            # Also slice the shortcut parameter
            shortcut = self.shortcut[start_idx:end_idx]
        else:
            shortcut = self.shortcut

        # Apply convolution
        out = self.apply_convolution(x, conv_kernel, shortcut, is_bhl_input)

        return out
