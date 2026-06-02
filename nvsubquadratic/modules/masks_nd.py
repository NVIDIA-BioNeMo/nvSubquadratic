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

r"""Learnable spatial modulation masks for N-dimensional convolution kernels.

Background
----------
Long-range convolutional operators such as :class:`~nvsubquadratic.modules.ckconv_nd.CKConvND`
and :class:`~nvsubquadratic.modules.hyena_nd.Hyena` use implicit kernel networks that
produce a dense kernel defined over the full coordinate grid.  Left unconstrained,
these kernels couple every spatial position to every other one — including pairs
that are spatially very distant — which can hurt both optimisation and
generalisation.

The modules in this file implement **soft receptive-field windows**: differentiable
spatial masks :math:`m \in [0, 1]^{*\text{spatial} \times C}` that are multiplied
element-wise into the implicit kernel values *before* the FFT convolution step.
After masking, the effective kernel decays towards zero beyond a characteristic
spatial radius, concentrating each channel's receptive field.

There are two families:

* **Exponential** (:class:`ExponentialModulationND`) — fixed, non-learnable
  decay rates initialised on a log-uniform ramp from slow to fast; used to
  enforce a hard inductive bias without any gradient signal changing the
  bandwidth.

* **Gaussian** (:class:`GaussianModulationND`,
  :class:`BlockAlignedGaussianModulationND`) — learnable standard-deviation
  parameters per spatial axis and channel; the mask is a factorised product of
  Gaussians, one per axis.  During training the bandwidth can grow or shrink,
  subject to ``[min_std, max_std]`` clamp bounds.

How masks interact with the FFT convolution operators
------------------------------------------------------
:class:`~nvsubquadratic.modules.ckconv_nd.CKConvND` evaluates its implicit
kernel network on a coordinate grid to obtain a dense kernel tensor
``k ∈ R^{* spatial × C}``.  The modulation mask ``m`` is applied in coordinate
space *before* the FFT:

.. code-block:: none

    k_masked = modulation(grid, k)   # element-wise, coordinate space
    output   = fftconvNd(x, k_masked)

Because the FFT convolution operates on the windowed kernel ``k_masked``,
the effective receptive field of the operator is bounded by the support of
the mask.  Narrow masks (small std / fast decay) correspond to local
operators; wide masks (large std / slow decay) approach global convolutions.

The convention used throughout this module is:

* The coordinate grid has values in **[−1, 1]** along each axis (center = 0).
* Mask values are in **[0, 1]** where **1 = fully included** (center of the
  kernel, near zero displacement) and **0 = fully excluded** (large
  displacement, far from the center of the convolution kernel).  Masking thus
  *suppresses* the kernel at large displacements, not at zero displacement.

For testing:
    PYTHONPATH=. python nvsubquadratic/modules/masks_nd.py
"""

import math
from collections.abc import Sequence

import torch
from einops import rearrange


def _normalize_init_extent(init_extent: float | Sequence[float] | None, data_dim: int) -> tuple[float, ...]:
    """Broadcast ``init_extent`` to a per-axis tuple of strictly-positive floats.

    Accepts a single float (broadcast to all axes) or a sequence of length
    ``data_dim``.  ``None`` is treated as ``1.0`` on every axis.  Used by
    :class:`GaussianModulationND` so each spatial axis can be initialized
    with its own bandwidth scale.

    Values must be ``> 0``.  Values ``> 1`` are allowed and useful on short
    anisotropic axes: ``init_extent`` multiplicatively scales **both** ends
    of the per-axis logspace ramp, so a large ``init_extent`` on a short
    axis pushes even the narrowest channel up to a usable bandwidth on that
    axis.  The clamp ``[min_std, max_std]`` enforces feasibility, so very
    large values simply saturate the entire ramp at ``max_std`` (= "this
    axis is essentially unmasked at init").

    Args:
        init_extent: A positive float (broadcast to all axes), a sequence of
            ``data_dim`` positive floats (one per axis), or ``None`` (treated
            as ``1.0`` on every axis).
        data_dim: Number of spatial/temporal dimensions.  Determines the
            expected length of a sequence ``init_extent``.

    Returns:
        A tuple of ``data_dim`` positive finite floats.

    Raises:
        TypeError: If ``init_extent`` is a bool, or not a float or sequence.
        ValueError: If ``init_extent`` is a sequence with length != ``data_dim``,
            or if any element is non-positive or non-finite.
    """
    if init_extent is None:
        init_extent = 1.0
    if isinstance(init_extent, bool):
        raise TypeError("init_extent must be a float or a sequence of floats, got bool")
    if isinstance(init_extent, (int, float)):
        extents: tuple[float, ...] = (float(init_extent),) * data_dim
    elif isinstance(init_extent, Sequence) and not isinstance(init_extent, (str, bytes)):
        extents = tuple(float(v) for v in init_extent)
        if len(extents) != data_dim:
            raise ValueError(f"init_extent sequence must have length data_dim={data_dim}, got length {len(extents)}")
    else:
        raise TypeError(f"init_extent must be a float or a sequence of floats, got {type(init_extent).__name__}")
    for ext in extents:
        if not ext > 0.0 or not math.isfinite(ext):
            raise ValueError(f"init_extent values must be finite and > 0, got {ext}")
    return extents


class ExponentialModulationND(torch.nn.Module):
    r"""Fixed exponential-decay spatial window applied to implicit convolutional kernels.

    Geometry
    --------
    Given a coordinate grid normalised to ``[−1, 1]`` (center = 0) and a
    set of per-axis, per-channel decay rates ``w_{d,c} > 0``, the mask at
    spatial position :math:`p = (p_0, \ldots, p_{D-1})` for channel ``c`` is:

    .. math::

        m_c(p) = \prod_{d=0}^{D-1} \exp\!\bigl(-\lvert p_d \rvert \cdot \lvert w_{d,c} \rvert\bigr)

    All values lie in ``(0, 1]``.  The mask equals **1** at the origin
    (displacement 0) and decays towards **0** as any coordinate moves away from
    the center.  Channels with large ``w`` decay quickly (narrow receptive
    field); channels with small ``w`` decay slowly (broad receptive field).

    The ND generalisation follows automatically from the product structure:
    for a 2D image grid the mask is a 2D tent surface; for a 3D volume it
    is a 3D "tent" shaped object.

    Decay rates are **not learnable** (they are registered as a parameter so
    they travel with the module and appear in ``state_dict``, but they are
    marked ``_no_weight_decay = True`` and are not updated by the optimizer).
    The rates are initialised on a linear ramp from ``slow_decay_pct`` to
    ``fast_decay_pct``, divided by ``data_dim`` so that the product across
    axes has a consistent magnitude regardless of the number of dimensions.

    Role in CKConvND
    ----------------
    :class:`~nvsubquadratic.modules.ckconv_nd.CKConvND` optionally passes the
    output of its implicit kernel network through this module before the FFT
    convolution step.  The resulting masked kernel is then convolved with the
    input signal via ``fftconvNd``.

    Args:
        data_dim: Number of spatial/temporal dimensions (1 for sequences, 2 for
            images, 3 for videos).
        num_channels: Number of feature channels ``C``.  Each channel receives
            a distinct decay rate.
        fast_decay_pct: Upper end of the decay-rate ramp (fastest / narrowest
            channel).  Default ``13.81`` (≈ :math:`\ln(10^6)`, so the
            narrowest channel decays to near zero within a small fraction of
            the grid).
        slow_decay_pct: Lower end of the decay-rate ramp (slowest / broadest
            channel).  Default ``2.3`` (≈ :math:`\ln(10)`, so the broadest
            channel retains ≈ 10 % of its value at the grid boundary).
    """

    def __init__(
        self,
        data_dim: int,
        num_channels: int,
        fast_decay_pct: float = 13.81,
        slow_decay_pct: float = 2.3,
    ):
        """Initialise the exponential modulation module.

        Args:
            data_dim: Number of spatial/temporal dimensions.
            num_channels: Number of feature channels to modulate.
            fast_decay_pct: Upper end of the per-channel decay-rate ramp (fastest channel).
            slow_decay_pct: Lower end of the per-channel decay-rate ramp (slowest channel).
        """
        super().__init__()
        self.data_dim = data_dim
        self.num_channels = num_channels
        self.fast_decay_pct = fast_decay_pct
        self.slow_decay_pct = slow_decay_pct

        # Create weight parameter
        _decay_linspace = (1.0 / data_dim) * torch.linspace(
            self.slow_decay_pct, self.fast_decay_pct, self.num_channels
        )
        self.weight = torch.nn.Parameter(torch.stack([_decay_linspace] * data_dim, dim=0))  # [data_dim, num_channels]

        # Add ._no_weight_decay flag to all parameters to avoid weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    def extra_repr(self):
        """Additional printing for the ExponentialModulationND class."""
        return f"data_dim={self.data_dim}, num_channels={self.num_channels}, fast_decay_pct={self.fast_decay_pct}, slow_decay_pct={self.slow_decay_pct}"

    def forward(self, grid: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r"""Apply exponential decay modulation element-wise to kernel features.

        For each spatial position :math:`p` and channel :math:`c` computes:

        .. math::

            \text{out}[\ldots, c] = x[\ldots, c] \cdot
                \prod_{d} \exp\!\bigl(-\lvert p_d \rvert \cdot \lvert w_{d,c} \rvert\bigr)

        The product is over all ``data_dim`` spatial axes.  The mask value is
        **1** at the origin and decreases monotonically towards 0 as the
        displacement from the origin grows.

        Args:
            grid: Coordinate grid of shape ``[1, *spatial_dims, data_dim]`` with
                values in ``[−1, 1]``.  Each entry ``grid[..., d]`` contains the
                normalised coordinate along axis ``d``.  Must be ``torch.float32``
                (lower precision collapses nearby coordinates together).
            x: Kernel feature tensor of shape ``[B, *spatial_dims, num_channels]``
                to be modulated.  ``B`` is the batch size; ``*spatial_dims`` must
                match the spatial shape of ``grid``.

        Returns:
            torch.Tensor: Modulated features with the same shape and dtype as ``x``.

        Raises:
            AssertionError: If ``grid.dtype`` is not ``torch.float32``.
        """
        # Ensure the grid tensor has the correct data type
        assert grid.dtype == torch.float32, (
            f"grid must be float32. At lower precision, indexes will be merged together. Current dtype: {grid.dtype}"
        )

        # Compute the decay factor for each channel based on the learned weights. Compute on float32 and cast to x.dtype.
        _grid = rearrange(grid, "b ... c -> b ... c 1")
        decay = torch.exp(-_grid.abs() * self.weight.float().abs()).prod(dim=-2).to(x.dtype)

        # Apply the decay factor to the input features.
        return x * decay


def _std_from_attenuation(attenuation: float, position: float, data_dim: int) -> float:
    """Compute the per-dimension std that gives a target mask value at a grid position.

    For a ``data_dim``-dimensional Gaussian mask evaluated at the corner
    ``(position, position, …, position)``::

        mask = exp(-0.5 * data_dim * (position / σ)²) = attenuation
        ⟹  σ = position * sqrt( -data_dim / (2 * ln(attenuation)) )

    This helper is used during initialisation of :class:`GaussianModulationND`
    to derive ``min_std`` and ``max_std`` from user-specified attenuation targets.
    All calls from that class pass ``data_dim=1`` because the attenuation targets
    are defined as **single-axis** (1D) measurements; the product structure of
    the full ND mask then gives an ND corner value of ``attenuation ** data_dim``.

    Args:
        attenuation: Desired mask value at ``position`` (must be in ``(0, 1)``).
        position: Absolute normalised grid coordinate at which the attenuation
            target is measured (must be ``> 0``).
        data_dim: Number of spatial dimensions assumed in the formula.  Pass
            ``1`` for single-axis targets (the typical case in this module).

    Returns:
        float: The standard deviation ``σ > 0`` such that the Gaussian mask
        equals ``attenuation`` at the given ``position``.

    Raises:
        AssertionError: If ``attenuation`` is not in ``(0, 1)`` or ``position``
            is not ``> 0``.
    """
    assert 0.0 < attenuation < 1.0, f"attenuation must be in (0, 1), got {attenuation}"
    assert position > 0.0, f"position must be > 0, got {position}"
    return position * math.sqrt(-data_dim / (2.0 * math.log(attenuation)))


class GaussianModulationND(torch.nn.Module):
    r"""Learnable Gaussian-window spatial mask for ND convolutional kernels.

    Geometry
    --------
    For a coordinate grid normalised to ``[−1, 1]`` (center = 0) and
    per-axis, per-channel standard deviations :math:`\sigma_{d,c} > 0`, the
    mask value at position :math:`p = (p_0, \ldots, p_{D-1})` for channel
    ``c`` is the product of per-axis Gaussians:

    .. math::

        m_c(p) = \prod_{d=0}^{D-1}
                 \exp\!\Bigl(-\tfrac{1}{2}\bigl(p_d / \sigma_{d,c}\bigr)^2\Bigr)

    All values lie in ``(0, 1]``.  The mask equals **1** at the origin
    and decays symmetrically in all directions; the level set at value ``v``
    is an axis-aligned ellipsoid with semi-axes :math:`\sigma_{d,c}\sqrt{-2\ln v}`.

    **1 = fully included, 0 = fully excluded.**  A narrow Gaussian (small
    ``σ``) concentrates the effective kernel around the origin, making the
    operator local; a wide Gaussian (large ``σ``) lets the full grid
    contribute, approaching a global convolution.

    ND generalisation
    -----------------
    The mask factorises over axes.  In 2D the mask surface looks like a 2D
    Gaussian bell (not a sphere — each axis has an independent ``σ``).  In
    3D it is a trivariate axis-aligned Gaussian.  Because the mask is a
    *product*, the corner value at position ``(position, position, …,
    position)`` is the product of the individual per-axis Gaussian values,
    which equals ``single_axis_mask_value ** data_dim``.  The attenuation
    parameters (``min_attenuation_at_step``, ``max_attenuation_at_limit``)
    are therefore defined as **single-axis (1D) measurements** — the
    effective ND attenuation at the grid corner is stricter by a factor of
    ``data_dim`` in the exponent.

    Parametrisation and clamping
    ----------------------------
    The learned parameter ``std_param`` of shape ``[data_dim, num_channels]``
    stores raw values that are mapped to strictly-positive std values via
    ``parametrization``:

    * ``'direct'`` — ``std_param`` IS the std; a ``register_forward_pre_hook``
      clamps it into ``[min_std, max_std]`` in-place (``torch.no_grad``), so
      gradients at the boundary are preserved through the activation.
    * ``'log'`` — ``std = exp(std_param)``; hard clamp applied after (breaks
      boundary gradients — see inline warning).
    * ``'softplus'`` — ``std = softplus(std_param)``; hard clamp applied after.

    Initialisation
    --------------
    ``min_attenuation_at_step`` and ``max_attenuation_at_limit`` define the
    **clamp bounds** ``[min_std, max_std]``.  The initial ``std_param`` is a
    logspace ramp from ``min_std`` to ``init_std_high_unit`` on every axis,
    scaled per-axis by ``init_extent``:

    * ``init_std_low[d]  = clamp(min_std            * extent[d], min_std, max_std)``
    * ``init_std_high[d] = clamp(init_std_high_unit * extent[d], min_std, max_std)``

    where ``init_std_high_unit ≈ 0.4724`` is the std at which a 1D Gaussian
    reaches ``0.1`` at position 1.  See ``init_extent`` below.

    All attenuation values are **single-axis (1D)** measurements; see the ND
    generalisation note above.

    Args:
        data_dim: Number of spatial/temporal dimensions (1 for sequences, 2
            for images, 3 for volumes).
        num_channels: Number of feature channels ``C`` to modulate.
        grid_size: Number of grid points per spatial dimension.  Used to
            compute the size of the smallest grid step
            (``min_step = 2 / (grid_size - 1)``), which sets ``min_std``.
            Auto-injected by :class:`~nvsubquadratic.modules.ckconv_nd.CKConvND`.
        min_attenuation_at_step: Target 1D mask value at the first grid step
            from the origin for the **narrowest** channel.  Smaller values
            → narrower minimum std → more local minimum channel.  Default
            ``0.1``.
        max_attenuation_at_limit: Target 1D mask value at the grid boundary
            (``position = 1``) for the **widest** channel.  Larger values
            → wider maximum std → less attenuation at the boundary.  Default
            ``0.95``.
        init_extent: Scalar or per-axis sequence controlling the initial
            bandwidth scale on each axis.  Must be strictly ``> 0``; defaults
            to ``1.0`` on every axis (reference ramp on every axis).

            Examples for an anisotropic ``L_cache = (8, 64, 64)`` cache
            with ``grid_size = 127``:

            * ``init_extent = 1.0`` — all axes use the reference ramp.  On a
              short depth axis the bottom of the ramp can be unusably narrow.
            * ``init_extent = (max_std/min_std, 1.0, 1.0)`` — depth ramp
              saturates at ``max_std`` (axis effectively unmasked at init).
            * ``init_extent = (1.0, 0.25, 0.25)`` — H/W are 4× narrower than
              the reference (extreme localisation on spatial axes).

        parametrization: One of ``'direct'``, ``'log'``, ``'softplus'``.
            Controls the mapping from ``std_param`` to std values.  Default
            ``'direct'``.
    """

    def __init__(
        self,
        data_dim: int,
        num_channels: int,
        grid_size: int,
        min_attenuation_at_step: float = 0.1,
        max_attenuation_at_limit: float = 0.95,
        init_extent: float | Sequence[float] = 1.0,
        parametrization: str = "direct",
    ):
        """Initialise the Gaussian modulation module.

        Args:
            data_dim: Number of spatial/temporal dimensions.
            num_channels: Number of feature channels to modulate.
            grid_size: Number of grid points per spatial dimension.
            min_attenuation_at_step: 1D mask value at the first grid step from
                the origin for the narrowest channel (sets ``min_std``).
            max_attenuation_at_limit: 1D mask value at the grid boundary for the
                widest channel (sets ``max_std``).
            init_extent: Per-axis bandwidth scale for initialisation (> 0).
                Pass a float to broadcast, or a sequence of length ``data_dim``.
            parametrization: Mapping from ``std_param`` to std values.
                One of ``'direct'``, ``'log'``, ``'softplus'``.
        """
        super().__init__()
        assert parametrization in {"log", "softplus", "direct"}, (
            "parametrization must be 'log' or 'softplus' or 'direct'"
        )
        self.data_dim = data_dim
        self.num_channels = num_channels
        self.parametrization = parametrization

        # All attenuation targets are single-axis (1D) measurements.
        # The multi-dim mask is the product of per-dim Gaussians, so the
        # 2D corner value is attenuation^2, etc.
        min_step = 2.0 / (grid_size - 1)
        min_std = _std_from_attenuation(min_attenuation_at_step, min_step, 1)
        max_std = _std_from_attenuation(max_attenuation_at_limit, 1.0, 1)

        # ``init_extent`` is a per-axis multiplicative scale on the entire
        # logspace ramp.  Reference ramp (extent=1) goes from ``min_std`` up
        # to ``init_std_high_unit`` (= the std at which a 1D Gaussian hits
        # 0.1 at position 1).  Scaling both ends lets a short anisotropic
        # axis lift its narrowest channel out of the "mask ≈ 0 everywhere"
        # regime that a shared low end would otherwise impose.
        self.init_extent = _normalize_init_extent(init_extent, data_dim)
        _INIT_EXTENT_ATTENUATION = 0.1
        init_std_high_unit = _std_from_attenuation(_INIT_EXTENT_ATTENUATION, 1.0, 1)
        # Per-axis (low, high) endpoints of the logspace ramp, clamped to
        # the feasible band.  When extent[d] is large, both endpoints
        # saturate at max_std and the ramp collapses to a constant
        # max_std on that axis (axis effectively unmasked at init).
        init_std_low_per_axis = [min(max(min_std * extent, min_std), max_std) for extent in self.init_extent]
        init_std_high_per_axis = [
            min(max(init_std_high_unit * extent, min_std), max_std) for extent in self.init_extent
        ]

        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self._min_step = float(min_step)

        # One logspace ramp per axis with its own (low, high) endpoints.
        # Result shape [data_dim, num_channels].
        init_std = torch.stack(
            [
                torch.logspace(math.log10(low), math.log10(high), num_channels)
                for low, high in zip(init_std_low_per_axis, init_std_high_per_axis)
            ],
            dim=0,
        )
        if parametrization == "log":
            param = init_std.log()
        elif parametrization == "softplus":
            # softplus^{-1}(x) ~ log(exp(x)-1); numerically stable for small x
            param = init_std.expm1().log()
        else:  # direct
            param = init_std
        self.std_param = torch.nn.Parameter(param)  # shape [data_dim, num_channels]

        # Add ._no_weight_decay flag to all parameters to avoid weight decay
        for p in self.parameters():
            p._no_weight_decay = True

        # Use a forward pre-hook to clamp std_param to the limits without breaking the gradient flow.
        if self.parametrization == "direct":
            self._clamp_hook = self.register_forward_pre_hook(self._clamp_direct_std_param_pre_hook)
        else:
            # IMPORTANT! DO NOT FORGET TO MANAGE GRADIENTS ON THE LIMITS FOR OTHER PARAMETRIZATIONS!
            pass

    def _clamp_direct_std_param_pre_hook(self, module, inputs):
        """Clamp std_param into [min_std, max_std] just before forward without tracking grads."""
        with torch.no_grad():
            self.std_param.data.clamp_(min=self.min_std, max=self.max_std)

    def _compute_std(self) -> torch.Tensor:
        """Map the raw ``std_param`` to strictly-positive standard deviations.

        Applies the parametrization-specific mapping and clamps to
        ``[min_std, max_std]``.  For the ``'direct'`` parametrization the
        clamp is applied by a pre-hook, so gradients at the boundary are
        preserved; for ``'log'`` and ``'softplus'`` the clamp is applied
        here (which zeros gradients at the boundary).

        Returns:
            torch.Tensor: Standard deviations of shape ``[data_dim, num_channels]``
            in ``float32``.  Values are guaranteed to lie in
            ``[min_std, max_std]``.
        """
        std = self.std_param.float()  # [data_dim, num_channels]
        if self.parametrization == "direct":
            # Pre-hook clamps parameters; return parameter directly to keep identity gradient
            return std
        elif self.parametrization == "log":
            std = std.exp()
        elif self.parametrization == "softplus":
            std = torch.nn.functional.softplus(std)
        else:
            raise ValueError(f"Invalid parametrization: {self.parametrization}")
        # Clamp the standard deviation to the limits.
        # IMPORTANT! THIS WILL BREAK THE GRADIENT FLOW ON THE LIMITS!
        std = std.clamp(min=self.min_std, max=self.max_std)
        return std

    @staticmethod
    def _mask_value(std: float, position: float) -> float:
        """1D Gaussian mask value: exp(-0.5 * (position / std)^2).

        Args:
            std: Standard deviation of the Gaussian (> 0).
            position: Absolute normalised grid coordinate at which to evaluate
                the mask.

        Returns:
            float: Mask value in ``(0, 1]``.
        """
        return math.exp(-0.5 * (position / std) ** 2)

    def extra_repr(self):
        """Additional printing for the GaussianModulationND class."""
        std = self._compute_std().detach()  # [data_dim, num_channels]
        step = self._min_step
        extent_str = ", ".join(f"{e:.3g}" for e in self.init_extent)
        per_axis_lines = []
        for axis in range(self.data_dim):
            row = std[axis]
            lo = row.min().item()
            hi = row.max().item()
            per_axis_lines.append(
                f"  axis {axis}: extent={self.init_extent[axis]:.3g}, "
                f"std∈[{lo:.4f}, {hi:.4f}], "
                f"narrow ch mask@step={self._mask_value(lo, step):.4f}, "
                f"wide ch mask@boundary={self._mask_value(hi, 1.0):.4f}"
            )
        return (
            f"data_dim={self.data_dim}, num_channels={self.num_channels}, "
            f"parametrization='{self.parametrization}'\n"
            f"  std clamp bounds: [{self.min_std:.4f}, {self.max_std:.4f}]\n"
            f"  init_extent (per axis): ({extent_str})\n" + "\n".join(per_axis_lines)
        )

    def forward(self, grid: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r"""Apply Gaussian spatial modulation element-wise to kernel features.

        For each spatial position :math:`p` and channel :math:`c` computes:

        .. math::

            \text{out}[\ldots, c] = x[\ldots, c] \cdot
                \exp\!\Bigl(-\tfrac{1}{2} \sum_{d} \bigl(p_d / \sigma_{d,c}\bigr)^2\Bigr)

        The exponent is computed as a single einsum (sum over axes) for
        efficiency, avoiding the intermediate ``prod(exp)`` formulation.

        Mask convention: **1 = fully included** (origin), **0 = fully
        excluded** (large displacement).

        Args:
            grid: Coordinate grid of shape ``[1, *spatial_dims, data_dim]`` with
                values in ``[−1, 1]``.  Must be ``torch.float32``.
            x: Kernel feature tensor of shape ``[B, *spatial_dims, num_channels]``.
                ``*spatial_dims`` must match the spatial shape of ``grid``.

        Returns:
            torch.Tensor: Modulated features with the same shape and dtype as
            ``x``.  The internal Gaussian computation is always done in
            ``float32`` and then cast to ``x.dtype``.

        Raises:
            AssertionError: If ``grid.dtype`` is not ``torch.float32``.
        """
        # Ensure the grid tensor has the correct data type
        assert grid.dtype == torch.float32, f"grid must be float32. Current dtype: {grid.dtype}"

        # Compute the standard deviation for each channel based on the learned weights. Compute on float32 and cast to x.dtype.
        std = self._compute_std()  # [data_dim, num_channels]

        # Compute mask
        # Faster: exp(sum) instead of prod(exp); avoid pow/div; one einsum + one exp
        # grid.shape: [b, ..., data_dim], std.shape: [data_dim, num_channels], exponent.shape: [b, ..., num_channels]
        exponent = -0.5 * torch.einsum("b...d,dc->b...c", grid.square(), std.square().reciprocal())
        gauss = exponent.exp_().to(x.dtype)

        # Apply the Gaussian modulation to the input features.
        return x * gauss


class BlockAlignedGaussianModulationND(GaussianModulationND):
    r"""Gaussian modulation with channel-reversed std ordering for block-structured SIRENs.

    Motivation
    ----------
    :class:`GaussianModulationND` initialises ``std_param`` so that channel 0
    has the **narrowest** Gaussian (smallest ``σ``, shortest spatial support,
    broadest spectral support) and the last channel has the **widest** Gaussian.
    This ordering is natural when channel index 0 carries high-frequency content,
    which should be localised in space.

    Block-structured SIRENs such as
    :class:`~nvsubquadratic.modules.kernels_nd.BlockDiagonalMultiOmegaSIRENKernelND`
    with a ``'linear'`` or ``'log'`` frequency schedule assign the **lowest**
    :math:`\omega_0` (low-frequency content) to the **first** block.  Low
    frequencies have long spatial support, so the natural alignment is the
    opposite: **widest** Gaussian on channel 0 (lowest :math:`\omega_0`),
    **narrowest** on the last channel (highest :math:`\omega_0`).

    Implementation
    --------------
    This subclass reverses ``std_param`` along the channel axis (``dim=-1``)
    immediately after the parent's ``__init__``.  All other behaviour —
    forward pass, pre-hook clamping, parametrisation, gradient flow — is
    inherited unchanged from :class:`GaussianModulationND`.

    The channel ordering after reversal:

    * Channel 0: widest Gaussian (largest ``σ``), longest spatial support,
      lowest effective frequency → matched to the lowest-:math:`\omega_0` block.
    * Channel ``C-1``: narrowest Gaussian (smallest ``σ``), shortest spatial
      support, highest effective frequency → matched to the highest-:math:`\omega_0`
      block.

    Args:
        data_dim: Number of spatial/temporal dimensions.
        num_channels: Number of feature channels ``C`` to modulate.
        grid_size: Number of grid points per spatial dimension.
        min_attenuation_at_step: 1D mask value at the first grid step (sets
            ``min_std`` clamp bound).  Default ``0.1``.
        max_attenuation_at_limit: 1D mask value at the grid boundary (sets
            ``max_std`` clamp bound).  Default ``0.95``.
        init_extent: Per-axis bandwidth scale for initialisation (> 0).
            Default ``1.0``.
        parametrization: One of ``'direct'``, ``'log'``, ``'softplus'``.
            Default ``'direct'``.
    """

    def __init__(
        self,
        data_dim: int,
        num_channels: int,
        grid_size: int,
        min_attenuation_at_step: float = 0.1,
        max_attenuation_at_limit: float = 0.95,
        init_extent: float = 1.0,
        parametrization: str = "direct",
    ):
        """Initialise the block-aligned Gaussian mask; see the class docstring for argument semantics."""
        super().__init__(
            data_dim=data_dim,
            num_channels=num_channels,
            grid_size=grid_size,
            min_attenuation_at_step=min_attenuation_at_step,
            max_attenuation_at_limit=max_attenuation_at_limit,
            init_extent=init_extent,
            parametrization=parametrization,
        )
        # Reverse channel axis so the widest Gaussian (last channel in the
        # parent) ends up on channel 0, matching the lowest-ω₀ block of a
        # block-structured SIREN kernel.
        with torch.no_grad():
            self.std_param.data.copy_(self.std_param.data.flip(dims=[-1]))


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Example usage
    data_dim = 2
    num_channels = 7
    grid_size = 63
    # Create grid
    linspace = torch.linspace(-1, 1, grid_size)
    grid = torch.stack(
        torch.meshgrid(*[linspace] * data_dim, indexing="ij"), dim=-1
    )  # [grid_size, grid_size, data_dim]
    grid = grid.unsqueeze(0)  # [1, grid_size, grid_size, data_dim]
    x = torch.ones(1, grid_size, grid_size, num_channels)  # [1, grid_size, grid_size, num_channels]

    modulator = ExponentialModulationND(data_dim, num_channels)
    output = modulator(grid, x)
    exp_masks = output[0].detach().cpu().permute(2, 0, 1)  # [C, H, W]
    print("Exponential masks shape:", exp_masks.shape)

    fig_exp, axes_exp = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4), squeeze=False)
    fig_exp.suptitle("ExponentialModulationND masks")
    for c in range(num_channels):
        ax = axes_exp[0, c]
        im = ax.imshow(exp_masks[c], cmap="viridis", origin="lower")
        ax.set_title(f"channel {c}")
        ax.axis("off")
        fig_exp.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    # Save exponential masks figure next to this script
    script_dir = Path(__file__).parent
    fig_exp.savefig(script_dir / "exponential_masks.png", dpi=150, bbox_inches="tight")

    gaussian_modulator = GaussianModulationND(data_dim, num_channels, grid_size=grid_size, parametrization="direct")
    gaussian_output = gaussian_modulator(grid, x)
    gauss_masks = gaussian_output[0].detach().cpu().permute(2, 0, 1)  # [C, H, W]
    print("Gaussian masks shape:", gauss_masks.shape)

    fig_gauss, axes_gauss = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4), squeeze=False)
    fig_gauss.suptitle("GaussianModulationND masks")
    for c in range(num_channels):
        gauss_masks_c = gauss_masks[c].detach().cpu()
        count_gt_02 = (gauss_masks_c > 0.2).sum().item()
        count_gt_01 = (gauss_masks_c > 0.1).sum().item()
        count_gt_005 = (gauss_masks_c > 0.05).sum().item()
        print(f"channel {c}: count>0.2={count_gt_02}, count>0.1={count_gt_01}, count>0.05={count_gt_005}")
        ax = axes_gauss[0, c]
        im = ax.imshow(gauss_masks_c, cmap="viridis", origin="lower")
        # masks values
        ax.set_title(f"Mask vals: {gaussian_modulator.std_param[0, c].detach().cpu()}")
        ax.axis("off")
        fig_gauss.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    # Save gaussian masks figure next to this script
    fig_gauss.savefig(script_dir / "gaussian_masks.png", dpi=150, bbox_inches="tight")

    # --- BlockAlignedGaussianModulationND sanity check ---
    aligned = BlockAlignedGaussianModulationND(data_dim, num_channels, grid_size=grid_size, parametrization="direct")
    baseline = GaussianModulationND(data_dim, num_channels, grid_size=grid_size, parametrization="direct")
    torch.testing.assert_close(aligned.std_param.data, baseline.std_param.data.flip(dims=[-1]))
    print("BlockAlignedGaussianModulationND std ordering reversed vs baseline: OK")

    plt.show()
