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

# TODO: Add license header here


r"""Continuous Kernel Convolution (CKConv) for N-dimensional signals.

Background
----------
CKConv (Romero et al., "CKConv: Continuous Kernel Convolution With Arbitrary
Resolution", ICLR 2022, arXiv:2102.02611) is a long-range convolutional
operator whose filter is **not** a learnable lookup table but instead an
*implicit neural representation* of the kernel: a small MLP maps continuous
spatial coordinates to kernel values on the fly.

Formally, for a D-dimensional input signal x ∈ R^{C × *spatial}, the CKConv
output is:

.. code-block:: none

    y = x * k_θ

where ``*`` denotes (multi-dimensional) convolution, and the kernel

.. code-block:: none

    k_θ(p) = MLP_θ(pos_enc(p))

is evaluated at every position p on a continuous spatial grid normalised to
``[-1, 1]^D``.  Because the kernel is defined by a fixed-size MLP rather than
an explicit filter bank, **the parameter count is independent of the signal
length**, and the same model can be evaluated at any resolution without weight
surgery (resolution-independence).  Long-range dependencies are captured
because the MLP can, in principle, place non-zero weight at any relative
offset spanning the full input — there is no fixed receptive field.

The convolution itself is computed in the frequency domain via FFT:

.. code-block:: none

    y = IFFT( FFT(x) ⊙ FFT(k_θ) )

which costs O(N log N) per channel rather than O(N·K) for a spatial
convolution with kernel size K = N.  This is the same subquadratic trick that
powers Hyena (see ``nvsubquadratic.modules.hyena_nd``).

ND generalisation
-----------------
This module extends the original 1D CKConv formulation to **arbitrary spatial
rank** (1D sequences, 2D images, 3D volumes) by routing the global convolution
through the FFT primitives in ``nvsubquadratic.ops.fftconv``.  The kernel
generator from ``nvsubquadratic.modules.kernels_nd`` evaluates the MLP on a
D-dimensional coordinate grid; the resulting filter is passed directly to the
FFT convolution ops.

Boundary conditions
-------------------
Two boundary conditions are supported per spatial axis:

* **Zero-padding** (``fft_padding="zero"``): standard linear convolution with
  "same" output size.  The kernel size is ``2*N`` (double-grid), covering the
  full receptive field without wrap-around.  Matches ``torch.nn.ConvNd(padding='same')``
  semantics.

* **Circular / periodic** (``fft_padding="circular"``): wrap-around convolution
  for signals with periodic boundary conditions (e.g. longitude in climate
  models, ARC grid problems).  The kernel size equals the input size
  (single-grid, ``(N+1)//2`` grid points evaluated per axis).

* **Per-axis mixed** (``fft_padding=["circular", "zero"]`` etc.): one mode
  string per spatial axis.  Routes through ``nvsubquadratic.ops.mixed_fftconv``.
  Used for datasets such as Well's ``rayleigh_benard`` and
  ``viscoelastic_instability``, where some axes are periodic and others are
  not.

* **Causal** (``is_causal=True``, 1D only): output at position ``n`` only
  depends on inputs at positions ``0, …, n``.  The kernel is evaluated on the
  double-grid and then cropped to its causal (positive-lag) half before the
  FFT convolution.

Shortcut (skip connection)
--------------------------
Every forward pass adds a per-channel residual term:

.. code-block:: none

    y ← y + shortcut ⊙ x

where ``shortcut`` is a learnable ``[hidden_dim]`` parameter vector.  This
algebraic shortcut is fused into the FFT convolution op (no extra kernel
launch) and matches the design used throughout Hyena-style operators.

Related modules
---------------
* ``nvsubquadratic.modules.kernels_nd`` — implicit kernel parametrisation
  (``SIRENKernelND``, ``RandomFourierKernelND``, FiLM-conditioned variants)
* ``nvsubquadratic.ops.fftconv`` — FFT convolution primitives consumed here
* ``nvsubquadratic.ops.mixed_fftconv`` — per-axis mixed-BC FFT convolution
* ``nvsubquadratic.modules.hyena_nd`` — Hyena operator that wraps CKConvND
  as its global conv

References:
----------
* Romero et al. (2022). *CKConv: Continuous Kernel Convolution With Arbitrary
  Resolution*. ICLR 2022. https://arxiv.org/abs/2102.02611
* Sitzmann et al. (2020). *Implicit Neural Representations with Periodic
  Activation Functions*. NeurIPS 2020. (SIREN kernel)
"""

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
    """Map a single padding-mode string to its per-axis periodic flag.

    Args:
        mode: A single padding-mode string, one of ``"zero"`` or ``"circular"``
            (case-insensitive, whitespace-stripped).

    Returns:
        ``True`` if the mode is ``"circular"`` (the axis uses circular /
        wrap-around FFT convolution, matching ``_PADDING_MODE_TO_PERIODIC``),
        ``False`` if it is ``"zero"`` (the axis uses zero-padded linear FFT
        convolution).  The boolean is the ``periodic`` flag used by the
        downstream FFT dispatch tables.

    Raises:
        ValueError: If ``mode`` is not one of the recognised padding modes.
    """
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

    Args:
        fft_padding: Boundary-condition specification, either a single mode
            string (``"zero"`` / ``"circular"``) or a sequence of mode strings
            of length ``data_dim`` (one per spatial axis).  OmegaConf
            ``ListConfig`` objects are accepted as sequences because ``Sequence``
            matching is used internally rather than ``isinstance(…, list)`` —
            this matters when configs flow through ``LazyConfig``.
        data_dim: Number of spatial dimensions in the input signal.  Used to
            validate the length of a sequence-form ``fft_padding`` and to
            broadcast a scalar mode string.

    Returns:
        A tuple of ``data_dim`` booleans, one per spatial axis.  ``True``
        means that axis uses circular (wrap-around / periodic) FFT convolution;
        ``False`` means zero-padded (linear) FFT convolution.  This tuple is
        the canonical ``periodic`` flag consumed by the FFT dispatch tables and
        the ``mixed_fftconv*`` ops.

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

    Args:
        op_fn: A ``mixed_fftconv*`` function from
            ``nvsubquadratic.ops.mixed_fftconv`` with signature
            ``(x, kernel, periodic, shortcut)``.
        periodic: Per-axis periodicity flags of length ``data_dim``.  ``True``
            on an axis means circular (wrap-around) convolution; ``False``
            means zero-padded (linear) convolution.

    Returns:
        A callable with signature ``(x, kernel, shortcut)`` that internally
        forwards ``periodic`` to ``op_fn``.
    """

    def _wrapped(x, kernel, shortcut):
        """Call ``op_fn(x, kernel, periodic, shortcut)`` with bound ``periodic``."""
        return op_fn(x, kernel, periodic, shortcut)

    return _wrapped


def _grid_is_single_per_axis(
    grid_type: "Literal['double', 'single'] | None",
    periodic: tuple[bool, ...],
) -> tuple[bool, ...]:
    """Return per-axis 'use single grid' flags for the SIREN kernel.

    In CKConvND, ``grid_type='single'`` means the SIREN kernel grid spans
    ``L = (N+1)//2`` grid points per axis.  ``SIRENKernelND`` evaluates on a
    ``(2*L - 1)``-point grid, so the produced kernel has size
    ``2*(N+1)//2 - 1 ≈ N`` — i.e. the kernel size equals the input size on
    that axis (paired with periodic/circular FFT conv).  ``'double'`` means
    the kernel grid spans ``L = N`` points, giving kernel size ``2*N - 1 ≈ 2*N``
    (paired with zero-padded FFT conv), so the kernel covers twice the input.

    - **String mode** (``grid_type`` is ``'single'`` / ``'double'``): the same
      choice applies to every axis.
    - **Tuple / mixed mode** (``grid_type is None``): the per-axis flag is
      auto-derived as ``periodic[d]`` (periodic axis ⇒ single grid; non-
      periodic ⇒ double grid). This matches the recipe in
      :func:`nvsubquadratic.ops.mixed_fftconv._mixed_recipe`.

    Args:
        grid_type: Either ``"single"`` (kernel size == input size, for circular
            conv), ``"double"`` (kernel size == 2 × input size, for zero-padded
            conv), or ``None`` to auto-derive per axis from ``periodic``.
        periodic: Per-axis periodicity flags of length ``data_dim``.

    Returns:
        A tuple of booleans of length ``len(periodic)``.  ``True`` on axis
        ``d`` means that axis uses the single grid (kernel size == input size);
        ``False`` means double grid (kernel size == 2 × input size).
    """
    if grid_type is None:
        return tuple(periodic)
    return (grid_type == "single",) * len(periodic)


class CKConvND(torch.nn.Module):
    """N-dimensional Continuous Kernel Convolution (CKConv) operator.

    CKConvND implements the CKConv operator (Romero et al., arXiv:2102.02611)
    generalised to arbitrary spatial rank D ∈ {1, 2, 3}.  The convolutional
    kernel is **not** stored as an explicit lookup table; instead it is
    produced on the fly by a small MLP evaluated on a continuous positional
    grid:

    .. code-block:: none

        k_θ(p) = MLP_θ(pos_enc(p)),   p ∈ [-1, 1]^D

    The convolution with the input signal x is then computed in the frequency
    domain:

    .. code-block:: none

        y = IFFT( FFT(x) ⊙ FFT(k_θ) )  +  shortcut ⊙ x

    at O(N log N) cost per channel, where N = prod(spatial_dims).

    The MLP and its positional encoding are provided through ``kernel_cfg``
    (typically a ``SIRENKernelND`` or ``RandomFourierKernelND`` lazy config
    from ``nvsubquadratic.modules.kernels_nd``).  An optional attenuation mask
    ``mask_cfg`` (e.g. ``GaussianModulationND``) is applied to the kernel
    values after the MLP forward pass to restrict the effective receptive
    field at initialisation.

    **Boundary conditions** are controlled jointly by ``fft_padding`` and
    ``grid_type``:

    * ``fft_padding="zero", grid_type="double"``: standard linear convolution
      with "same" output size.  The kernel spans the full input (double-grid:
      ``2*N`` points) and is zero-padded before the FFT.
    * ``fft_padding="circular", grid_type="single"``: periodic convolution,
      kernel size == input size (single-grid: ``(N+1)//2`` grid points → kernel
      of length N after the MLP).
    * ``fft_padding=["circular", "zero"], grid_type=None``: per-axis mixed
      boundary conditions; ``grid_type`` is auto-derived per axis.

    **Context parallelism**: when ``cp_group`` is supplied in ``forward``, the
    kernel is sliced along the channel dimension to match the local slice of
    the input, and the ``shortcut`` parameter is sliced accordingly.  Causal
    mode is not verified to be correct under CP.

    Attributes:
        data_dim (int): Spatial rank of the input (1 for sequences, 2 for
            images, 3 for volumes).
        hidden_dim (int): Number of channels C processed by this operator.
        fft_padding (str or Sequence[str]): Boundary condition specification
            as supplied by the caller.  The normalised per-axis representation
            is in ``_periodic_per_axis``.
        is_causal (bool): Whether the operator enforces causal (past-only)
            convolution.  Only valid when ``data_dim=1``.
        use_chunked_fftconv (bool): Whether to process channels in chunks to
            reduce peak GPU memory.
        use_fp16_fft (bool): Whether to use fp16 FFT convolution ops.
        fft_backend (str): FFT backend identifier, ``"torch_fft"`` or
            ``"subq_ops"``.
        grid_type (str or None): Kernel grid size mode (``"single"``,
            ``"double"``, or ``None`` for per-axis auto-derivation).
        kernel (nn.Module): Implicit kernel generator (produces
            ``(kernel_values, grid)`` on each forward call).
        mask (nn.Module): Attenuation mask applied to kernel values after
            generation.  ``nn.Identity`` when no mask is configured.
        shortcut (nn.Parameter): Learnable per-channel skip-connection scale
            of shape ``(hidden_dim,)``.  Fused into the FFT convolution op.
            Initialised with ``uniform(-1/√hidden_dim, 1/√hidden_dim)``
            (Kaiming-uniform scale).
        fftconv_fn (callable): Selected FFT convolution function for
            channels-last (BLH) input with internal reshape.  Signature:
            ``(x, kernel, shortcut) → output``.
        fftconv_fn_bhl_input (callable): Selected FFT convolution function for
            channels-first (BHL) input.  Signature:
            ``(x, kernel, shortcut) → output``.
        _periodic_per_axis (tuple[bool, ...]): Per-axis periodicity flags
            of length ``data_dim``, derived from ``fft_padding``.
        _is_tuple_mode (bool): ``True`` when ``fft_padding`` was supplied as
            a sequence of mode strings (mixed-BC path).
    """

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
        """Construct a CKConvND operator.

        Validates the combination of ``fft_padding``, ``grid_type``,
        ``is_causal``, ``use_fp16_fft``, and ``fft_backend``, normalises the
        per-axis boundary-condition representation, adjusts ``kernel_cfg``
        and ``mask_cfg`` to match the resolved kernel grid geometry, and
        selects the appropriate FFT convolution function pair.

        Args:
            data_dim: Spatial rank of the input signal.  ``1`` for 1D
                sequences, ``2`` for 2D images (H, W), ``3`` for 3D
                volumes (D, H, W).
            hidden_dim: Number of channels C.  Determines the size of the
                learnable ``shortcut`` parameter and the channel dimension of
                every intermediate tensor.
            kernel_cfg: Lazy config (``LazyConfig``) that instantiates the
                implicit kernel generator when resolved.  Typically points to
                ``SIRENKernelND`` or ``RandomFourierKernelND``.  The kernel's
                ``out_dim`` must equal ``hidden_dim``; a mismatch will produce
                incorrect tensor shapes at runtime.  The ``L_cache`` field, if
                present, is adjusted to match the resolved kernel grid size
                (single or double grid) before instantiation.
            mask_cfg: Lazy config for the attenuation mask applied to the
                generated kernel values.  Use ``torch.nn.Identity`` (or an
                empty identity config) for no masking.  If the mask class
                accepts a ``grid_size`` parameter, it is set automatically
                based on the largest per-axis kernel size.
            grid_type: Relationship between the SIREN coordinate grid and the
                input spatial size on each axis.

                * ``"single"``: grid spans ``(N+1)//2`` points → kernel size
                  equals input size N (for periodic / circular conv).
                * ``"double"``: grid spans ``N`` points → kernel size is
                  ``2*N - 1 ≈ 2*N`` (for zero-padded conv).
                * ``None``: **required** when ``fft_padding`` is a per-axis
                  list.  The grid type is auto-derived per axis: ``"single"``
                  on periodic axes, ``"double"`` on non-periodic axes.

                Must not be ``None`` when ``fft_padding`` is a single mode
                string (``"zero"`` or ``"circular"``).
            fft_padding: Boundary-condition mode.  Accepted forms:

                * ``"zero"``: all axes zero-padded (linear "same" conv).
                * ``"circular"``: all axes periodic (wrap-around conv).
                  Requires ``grid_type="single"`` and
                  ``use_chunked_fftconv=False``.
                * ``["circular", "zero"]`` (list/tuple of mode strings, one
                  per spatial axis, length must equal ``data_dim``): per-axis
                  mixed boundary conditions.  Requires ``grid_type=None`` and
                  ``fft_backend="torch_fft"`` and ``use_fp16_fft=False``.
                  Mode names are case-insensitive and whitespace-stripped.

                Must be ``"zero"`` (or an all-``"zero"`` list) when
                ``is_causal=True``.
            is_causal: If ``True``, enforce causal (past-only) convolution so
                that the output at position ``n`` only depends on inputs at
                positions ``0, …, n``.  Only valid for ``data_dim=1``.
                Incompatible with periodic ``fft_padding`` and with the
                per-axis list form of ``fft_padding``.  Default: ``False``.
            use_chunked_fftconv: If ``True``, process channels in groups to
                reduce peak GPU memory from complex FFT intermediates.
                Typical savings: ~26% memory at ~11% compute overhead.
                Not supported with ``fft_padding="circular"``.
                Default: ``False``.
            use_fp16_fft: If ``True``, use fp16 FFT convolution ops.
                Uses ``norm="ortho"`` internally to prevent overflow.  Saves
                ~36% peak memory per convolution at ~0.8% mean relative error
                vs fp32.  For zero/causal padding, spatial dims are
                auto-padded to the next power of two.  For circular padding,
                the input dims must already be powers of two (a runtime
                assertion fires otherwise).  Not supported with a per-axis
                ``fft_padding`` list (see ``docs/ops/MIXED_BC_PLAN.md``).
                Not supported with ``fft_backend="subq_ops"``.
                Default: ``False``.
            fft_backend: Which FFT convolution backend to use.

                * ``"torch_fft"`` (default): torch.fft-based implementations
                  in ``nvsubquadratic.ops.fftconv`` and related modules.
                * ``"subq_ops"``: optimised CUDA kernels from
                  ``subquadratic_ops_torch``.  Supported configurations:

                  - ``data_dim=2``, ``is_causal=False``, ``fft_padding="zero"``
                    (2D non-causal zero-padded conv).  Per-sample (FiLM)
                    batched kernel weights are supported on this path.
                  - ``data_dim=1``, ``is_causal=True`` (1D causal conv).
                    The 1D causal CUDA kernel does not accept batched per-sample
                    weights; FiLM conditioning is unsupported on this path.

                  Does not support fp16 FFT, per-axis ``fft_padding``, or
                  ``data_dim=3``.

        Raises:
            AssertionError: If ``fft_backend`` is not one of the recognised
                values, or if a constraint between ``grid_type``,
                ``fft_padding``, ``is_causal``, ``use_fp16_fft``, and
                ``fft_backend`` is violated.
            ValueError: If ``fft_padding`` is invalid (wrong type, wrong
                length, comma-separated string, boolean), if ``is_causal``
                is combined with a per-axis padding list or periodic padding,
                or if the resolved ``(fft_padding, data_dim)`` combination
                has no registered FFT function.
            NotImplementedError: If ``use_fp16_fft=True`` is requested
                together with a per-axis ``fft_padding`` list (fp16 mixed
                ops are not yet implemented).
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
        """Return a concise summary string for ``print(module)`` and ``repr(module)``.

        Returns:
            A human-readable string listing the key hyperparameters:
            ``data_dim``, ``hidden_dim``, ``fft_padding``,
            ``periodic_per_axis`` (only when in per-axis list mode),
            ``grid_type``, ``is_causal``, ``use_chunked_fftconv``,
            ``use_fp16_fft``, and ``fft_backend``.
        """
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

        Two phases.

        **Phase 1 — kernel generation (via SIREN MLP).**  Delegated to
        ``self.kernel.flop_count(grid_lens, inference)``.  At
        ``inference=True`` without FiLM, the kernel is input-independent and
        can be precomputed, so this phase returns 0.

        **Phase 2 — FFT-based depthwise convolution** with ``C =
        self.hidden_dim``.  The convolution runs in the frequency domain.
        Padded signal sizes ``Np_i`` depend on the padding mode:

        - ``"zero"`` non-causal ("same" mode):
          ``Np_i = min(s_i + (k_i + 1) // 2, 2 * s_i)``.  Only half the
          kernel width of extra padding is needed because the output is
          centre-cropped back to the input size.  Matches ``fftconv.py``
          lines 624-628.
        - ``"zero"`` causal (1D only): ``Np_i = min(s_i + k_i, 2 * s_i)``.
          Full linear convolution length; the output is tail-cropped.
        - ``"circular"``: ``Np_i = s_i``.  Wrap-around, no extra padding.

        A separable N-D FFT on a grid of size ``(Np_1, ..., Np_d)`` costs
        ``5 * prod(Np) * sum(log2(Np_i))`` real FLOPs per channel, from the
        radix-2 Cooley-Tukey decomposition (each butterfly ≈ 5 real FLOPs:
        1 complex multiply = 4 real muls + 2 real adds, minus shared
        twiddle-factor optimisations).  The implementation uses ``rfft``
        (real-to-complex), which is ~2x cheaper than a full complex FFT;
        the ``5N log N`` formula is a conservative upper bound consistent
        with vision-paper conventions.

        Three FFTs are needed (forward of input, forward of kernel, inverse
        of the product).  At ``inference=True`` without FiLM the kernel FFT
        is precomputed and cached, reducing to two FFTs.

        Pointwise complex multiply in the frequency domain costs
        ``6 * C * prod(Np)`` (4 real muls + 2 real adds for ``(a + bi)(c + di)``).
        The shortcut (skip connection) costs ``C * prod(spatial_dims)``
        elementwise multiplies.

        Args:
            spatial_dims: Spatial dimensions of the input signal, e.g.
                ``(H, W)`` for a 2D image or ``(L,)`` for a 1D sequence.
                Must have length equal to ``self.data_dim``.
            inference: If ``True`` and the kernel has no FiLM conditioning,
                skip the kernel generation and kernel FFT FLOPs (both can be
                precomputed and cached at inference time).

        Returns:
            Total estimated FLOPs as an integer.
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
        """Apply the FFT-based depthwise convolution.

        Dispatches to the pre-selected ``fftconv_fn`` or
        ``fftconv_fn_bhl_input`` depending on the memory layout of ``x``.
        When ``is_bhl_input=True`` the kernel is first transposed from
        channels-last ``(B, *spatial, C)`` to channels-first ``(B, C, *spatial)``
        to match the BHL-native FFT op.

        The output y is computed as:

        .. code-block:: none

            y = IFFT( FFT(x) ⊙ FFT(conv_kernel) ) + shortcut ⊙ x

        The ``shortcut`` term is fused inside the FFT op (no extra kernel
        launch).

        Args:
            x: Input signal.

                * BLH layout (``is_bhl_input=False``): shape
                  ``(B, *spatial, C)`` where ``C = self.hidden_dim``.
                * BHL layout (``is_bhl_input=True``): shape
                  ``(B, C, *spatial)``.

            conv_kernel: Kernel values produced by ``self.kernel`` and
                optionally masked by ``self.mask``.  Always in channels-last
                (BLH) format on entry: shape ``(1_or_B, *kernel_spatial, C)``.
                ``kernel_spatial`` equals ``spatial`` on single-grid (circular)
                axes and ``2*N - 1`` on double-grid (zero-padded) axes, where
                ``N`` is the corresponding input spatial size.  Transposed
                internally when ``is_bhl_input=True``.
            shortcut: Per-channel skip-connection scale, shape ``(C,)``.
                Typically ``self.shortcut`` or a CP-sliced view thereof.
            is_bhl_input: If ``True``, treat ``x`` as channels-first
                ``(B, C, *spatial)`` and use ``self.fftconv_fn_bhl_input``.
                If ``False``, treat ``x`` as channels-last ``(B, *spatial, C)``
                and use ``self.fftconv_fn`` (which handles the reshape
                internally).

        Returns:
            Output tensor in the same memory layout as the input ``x``:
            ``(B, *spatial, C)`` when ``is_bhl_input=False``, or
            ``(B, C, *spatial)`` when ``is_bhl_input=True``.
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
        """Run the CKConv forward pass.

        Generates the implicit kernel from the positional grid, optionally
        applies the attenuation mask, crops the kernel for causal mode,
        handles context-parallel channel slicing, and applies the FFT
        convolution with the shortcut term.

        Computation (non-causal, no CP):

        .. code-block:: none

            grid_lens = [(s+1)//2  if single-grid axis  else  s  for s in spatial_dims]
            k_θ, grid = self.kernel(grid_lens, conditioning=conditioning)  # (1, *grid_lens, C)
            k_θ       = self.mask(grid=grid, x=k_θ)                        # attenuation
            y         = IFFT(FFT(x) ⊙ FFT(k_θ)) + shortcut ⊙ x           # FFT conv

        For causal mode (1D only), ``k_θ`` is cropped to its causal (positive-
        lag) half before the FFT convolution:

        .. code-block:: none

            k_θ = k_θ[..., kernel_len // 2 :, :]   # keep second half

        Args:
            x: Input signal tensor.  Two supported layouts:

                * **Channels-last** (``is_bhl_input=False``, default):
                  shape ``(B, *spatial, hidden_dim)`` where ``spatial`` has
                  length ``self.data_dim``.
                * **Channels-first** (``is_bhl_input=True``):
                  shape ``(B, hidden_dim, *spatial)``.

            is_bhl_input: If ``True``, ``x`` is in channels-first
                ``(B, C, *spatial)`` layout.  Default: ``False`` (channels-last).
            cp_group: Context-parallel process group.  When provided and
                ``cp_group.size() > 1``, the kernel and shortcut are sliced
                along the channel dimension to match the local channel slice
                held by this rank.  The spatial slice of ``x`` is expected to
                have already been distributed by the caller.  Causal mode is
                not verified to be correct under CP.  Default: ``None``
                (single-device / no CP).
            **mixer_kwargs: Additional keyword arguments forwarded to the
                kernel generator.  The following key is recognised:

                * ``conditioning`` (``torch.Tensor``, shape ``(B, cond_dim)``):
                  conditioning vector for FiLM-enabled kernels such as
                  ``SIRENKernelND`` with a ``film_cfg``.  Ignored (no-op) when
                  the kernel has no FiLM generator.

        Returns:
            Output tensor in the same memory layout as ``x``:
            ``(B, *spatial, hidden_dim)`` when ``is_bhl_input=False``, or
            ``(B, hidden_dim, *spatial)`` when ``is_bhl_input=True``.

        Raises:
            ValueError: If ``cp_group`` is provided together with
                ``is_causal=True``.  This combination is **explicitly rejected**
                because it has not been verified for correctness — the causal
                kernel crop and CP channel slicing interact in ways that may
                silently leak future positions.  Do not rely on the error being
                absent in future versions without re-verification.
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
