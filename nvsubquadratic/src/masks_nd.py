# David W. Romero, 2025-09-09

"""
Modulation masks for N-dimensional data.

These masks are used to modulate the input features of a convolutional kernel.

For testing:
    PYTHONPATH=. python nvsubquadratic/src/masks_nd.py
"""

import torch
from einops import rearrange


class ExponentialModulationND(torch.nn.Module):
    """
    Applies exponential decay modulation to input features.

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
        super().__init__()
        self.data_dim = data_dim
        self.num_channels = num_channels
        self.fast_decay_pct = fast_decay_pct
        self.slow_decay_pct = slow_decay_pct

        # Create weight parameter
        _decay_linspace = (1.0 / data_dim) * torch.linspace(self.slow_decay_pct, self.fast_decay_pct, self.num_channels)
        self.weight = torch.nn.Parameter(torch.stack([_decay_linspace] * data_dim, dim=0))  # [data_dim, num_channels]

        # Add ._no_wd flag to all parameters to avoid weight decay
        for param in self.parameters():
            param._no_wd = True

    def extra_repr(self):
        return f"data_dim={self.data_dim}, num_channels={self.num_channels}, fast_decay_pct={self.fast_decay_pct}, slow_decay_pct={self.slow_decay_pct}"

    def forward(self, grid: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Applies exponential modulation to the input tensor `x` based on the coordinates in `grid`.

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
