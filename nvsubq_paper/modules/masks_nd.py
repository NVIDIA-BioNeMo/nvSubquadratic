# TODO: Add license header here


"""Modulation masks for N-dimensional data.

These masks are used to modulate the input features of a convolutional kernel.

For testing:
    PYTHONPATH=. python nvsubq_paper/modules/masks_nd.py
"""

import math

import torch
from einops import rearrange


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


class GaussianModulationND(torch.nn.Module):
    """Gaussian decay modulation across N spatial/temporal dimensions.

    For each data dimension d and channel c we learn a (positive) standard deviation sigma_{d,c}.
    Given a coordinate grid (centered around 0) we apply:

        mask_{..., c} = Π_d exp( - 0.5 * (grid_d / sigma_{d,c})^2 )

    which is then multiplied elementwise with the input features.

    Mean is fixed (no learnable shift) so modulation remains symmetric around zero.

    Args:
        data_dim: Number of spatial/temporal dimensions represented in the last axis of `grid`.
        num_channels: Number of feature channels to modulate.
        min_std: Lower bound (after transformation) for stability.
        max_std: Optional upper cap applied during forward (None to disable).
        init_std_low / init_std_high: Range used to initialise stds per channel (linearly spaced) before transforming.
        parametrization: 'log' (exp) or 'softplus' to ensure positivity.
    """

    def __init__(
        self,
        data_dim: int,
        num_channels: int,
        min_std: float = 1e-3,
        max_std: float | None = None,
        init_std_low: float = 0.05,
        init_std_high: float = 1.0,
        parametrization: str = "direct",
    ):
        """Initialize the GaussianModulationND class.

        Args:
            data_dim: Dimension of input data.
            num_channels: Number of input channels to be modulated.
            min_std: Lower bound (after transformation) for stability.
            max_std: Optional upper cap applied during forward (None to disable).
            init_std_low: Range used to initialise stds per channel (linearly spaced) before transforming.
            init_std_high: Range used to initialise stds per channel (linearly spaced) before transforming.
            parametrization: Parametrization to use.
        """
        super().__init__()
        assert parametrization in {"log", "softplus", "direct"}, (
            "parametrization must be 'log' or 'softplus' or 'direct'"
        )
        self.data_dim = data_dim
        self.num_channels = num_channels
        self.min_std = float(min_std)
        self.max_std = float(max_std) if max_std is not None else None
        self.parametrization = parametrization

        # Create weight parameter
        init_std_per_channel = torch.logspace(math.log10(init_std_low), math.log10(init_std_high), num_channels)
        init_std = torch.stack([init_std_per_channel] * data_dim, dim=0)  # [data_dim, num_channels]
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

    @torch.compiler.disable
    def _clamp_direct_std_param_pre_hook(self, module, inputs):
        """Clamp std_param into [min_std, max_std] just before forward without tracking grads."""
        with torch.no_grad():
            self.std_param.clamp_(min=self.min_std)
            if self.max_std is not None:
                self.std_param.clamp_(max=self.max_std)

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
        # Clamp the standard deviation to the limits. IMPORTANT! THIS WILL BREAK THE GRADIENT FLOW ON THE LIMITS!
        std = std.clamp_min(self.min_std)
        if self.max_std is not None:
            std = std.clamp_max(self.max_std)
        return std

    def extra_repr(self):
        """Additional printing for the GaussianModulationND class."""
        return (
            f"data_dim={self.data_dim}, num_channels={self.num_channels}, min_std={self.min_std}, "
            f"max_std={self.max_std}, parametrization='{self.parametrization}'"
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

    gaussian_modulator = GaussianModulationND(
        data_dim, num_channels, init_std_low=0.025, init_std_high=0.35, parametrization="direct"
    )
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

    plt.show()
