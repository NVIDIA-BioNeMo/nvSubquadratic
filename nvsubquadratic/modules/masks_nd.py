# TODO: Add license header here


"""Modulation masks for N-dimensional data.

These masks are used to modulate the input features of a convolutional kernel.

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
    """Applies exponential decay modulation to input features.

    This module modulates input features by applying an exponential decay function on each dimension of an N-dimensional input.
    The decay rates are parameterized by a set of learned decay rates.

    Args:
        data_dim (int): Dimension of input data (1D for sequences, 2D for images, 3D for videos, etc.).
        num_channels (int): Number of input channels to be modulated.
        fast_decay_pct (float, optional): Percentage for the fastest decay rate. Default is 13.81.
        slow_decay_pct (float, optional): Percentage for the slowest decay rate. Default is 2.3.
    """

    def __init__(
        self,
        data_dim: int,
        num_channels: int,
        fast_decay_pct: float = 13.81,
        slow_decay_pct: float = 2.3,
    ):
        """Initialize the ExponentialModulationND class.

        Args:
            data_dim: Dimension of input data.
            num_channels: Number of input channels to be modulated.
            fast_decay_pct: Percentage for the fastest decay rate.
            slow_decay_pct: Percentage for the slowest decay rate.
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
        """Applies exponential modulation to the input tensor `x` based on the coordinates in `grid`.

        Args:
            grid (torch.Tensor): A tensor representing grid values (shape: [1, * spatial_dims, data_dim]).
                Must have dtype `torch.float32`.
            x (torch.Tensor): Input features to be modulated (shape: [batch_size, * spatial_dims, num_channels]).

        Returns:
            torch.Tensor: The modulated input features (same shape as `x`).

        Raises:
            AssertionError: If `grid` is not of type `torch.float32`.
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

    Args:
        attenuation: Desired mask value (0 < attenuation < 1).
        position: Absolute grid coordinate (> 0).
        data_dim: Number of spatial dimensions.
    """
    assert 0.0 < attenuation < 1.0, f"attenuation must be in (0, 1), got {attenuation}"
    assert position > 0.0, f"position must be > 0, got {position}"
    return position * math.sqrt(-data_dim / (2.0 * math.log(attenuation)))


class GaussianModulationND(torch.nn.Module):
    """Gaussian decay modulation across N spatial/temporal dimensions.

    For each data dimension d and channel c we learn a (positive) standard deviation sigma_{d,c}.
    Given a coordinate grid (centered around 0) we apply:

        mask_{..., c} = Π_d exp( - 0.5 * (grid_d / sigma_{d,c})^2 )

    which is then multiplied elementwise with the input features.

    Mean is fixed (no learnable shift) so modulation remains symmetric around zero.

    **Initialization** — pass ``min_attenuation_at_step`` and
    ``max_attenuation_at_limit`` (plus ``grid_size``, auto-injected by
    CKConvND).  These define the **clamp bounds** — the narrowest any
    channel can get (``min_std``) and the widest (``max_std``).  Optionally
    pass ``init_extent`` to control the initial bandwidth scale **per
    axis**.

    All attenuation values are **single-axis** (1D) measurements.  Since
    the mask is a product of per-dimension Gaussians, the 2D corner value
    is ``attenuation ** 2``, and the 3D corner is ``attenuation ** 3``,
    etc.

    - ``min_attenuation_at_step`` — 1D mask value at the first grid step
      from center for the narrowest *possible* channel.  Sets ``min_std``
      and the **reference** ``init_std_low`` (narrowest channel starts at
      the clamp bound when ``init_extent = 1``).
    - ``max_attenuation_at_limit`` — 1D mask value at the grid boundary
      (position 1) for the widest *possible* channel.  Sets ``max_std``.
    - ``init_extent`` — per-axis bandwidth scale that multiplicatively
      scales **both** ends of the per-axis logspace ramp:
      ``init_std_low[d]  = clamp(min_std            * extent[d], min_std, max_std)``
      ``init_std_high[d] = clamp(init_std_high_unit * extent[d], min_std, max_std)``
      where ``init_std_high_unit ≈ 0.4724`` is the std at which a 1D
      Gaussian reaches ``0.1`` at position 1.  Pass a float (broadcast to
      all axes) or a sequence of length ``data_dim``.  Values must be
      strictly ``> 0``; defaults to ``1.0`` on every axis (recovers the
      reference ramp ``[min_std, init_std_high_unit]``).

      Examples on an anisotropic ``L_cache=(8, 64, 64)`` cube cache
      (mask grid_size = 127):

      * ``init_extent = 1.0`` — all axes use the reference ramp from
        ``min_std ≈ 0.0075`` to ``init_std_high ≈ 0.4724``.  On the
        depth axis the bottom of the ramp is unusably narrow (mask ≈ 0
        across depth for early channels).
      * ``init_extent = (1.0, 0.25, 0.25)`` — depth uses the reference
        ramp; H/W are 4× narrower at both ends (extreme localization).
      * ``init_extent = (max_std/min_std, 1.0, 1.0)`` (≈ ``(416, 1, 1)``
        at defaults) — depth ramp saturates at ``max_std`` end-to-end;
        every depth channel is initialised at the widest possible
        Gaussian, i.e. depth axis is essentially unmasked at init.
        Useful when ``L_cache_d`` is short and depth-axis frequency
        content is naturally bounded by the small kernel grid.

    Only **initialization** is per-axis — ``min_std`` and ``max_std``
    (clamp bounds) remain scalar and shared across axes.

    Args:
        data_dim: Number of spatial/temporal dimensions.
        num_channels: Number of feature channels to modulate.
        min_attenuation_at_step: 1D mask value at first grid step (sets clamp
            lower bound and the reference init lower bound).
        max_attenuation_at_limit: 1D mask value at grid boundary (sets clamp
            upper bound).
        init_extent: Scalar or per-axis sequence controlling the initial
            bandwidth scale on each axis.  Strictly ``> 0``; default
            ``1.0`` (reference ramp on every axis).
        grid_size: Kernel grid points per dimension.  Auto-injected by CKConvND.
        parametrization: ``'log'``, ``'softplus'``, or ``'direct'``.
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
        """Initialize the GaussianModulationND class."""
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
        """Computes the standard deviation for each channel based on the learned weights.

        Returns:
            torch.Tensor: The standard deviation for each channel (shape: [data_dim, num_channels]).
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
        """1D Gaussian mask value: exp(-0.5 * (position / std)^2)."""
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
        """Applies Gaussian modulation to the input tensor `x` based on the coordinates in `grid`.

        Args:
            grid (torch.Tensor): A tensor representing grid values (shape: [1, * spatial_dims, data_dim]).
                Must have dtype `torch.float32`.
            x (torch.Tensor): Input features to be modulated (shape: [batch_size, * spatial_dims, num_channels]).

        Returns:
            torch.Tensor: The modulated input features (same shape as `x`).

        Raises:
            AssertionError: If `grid` is not of type `torch.float32`.
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
    """Gaussian modulation with channel-reversed std_param for block-structured SIRENs.

    The parent :class:`GaussianModulationND` initializes ``std_param`` so that
    channel 0 has the narrowest Gaussian (``min_std``) and the last channel has
    the widest (``init_std_high``).  That ordering assumes the *narrow* mask
    (short spatial support → broad spectral support) should be applied to
    channels that carry high-frequency content.

    Block-structured SIRENs such as
    :class:`~nvsubquadratic.modules.kernels_nd.BlockDiagonalMultiOmegaSIRENKernelND`
    with a ``linear`` or ``log`` schedule put the *lowest* ω₀ (low-frequency
    content) on the first block, so the natural alignment is the opposite:
    widest Gaussian on channel 0, narrowest on the last channel.

    This subclass just reverses ``std_param`` along the channel axis after the
    parent's initialization — no other behaviour changes (forward pass,
    clamping, parametrization, grad flow are all inherited unchanged).

    Args:
        data_dim, num_channels, grid_size, min_attenuation_at_step,
        max_attenuation_at_limit, init_extent, parametrization:
            Passed straight through to :class:`GaussianModulationND`.
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
        """Initialize the block-aligned Gaussian mask; see the class docstring for argument semantics."""
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
