# TODO: Add license header here


"""Modulation masks for N-dimensional data.

These masks are used to modulate the input features of a convolutional kernel.

For testing:
    PYTHONPATH=. python nvsubquadratic/modules/masks_nd.py
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

    Kernel Size (Fraction of Grid):
        The kernel size is specified as a **fraction of the grid** (0.0 to 1.0), making it
        resolution-independent. This is more intuitive: kernel_size=0.25 covers 25% of the grid.

        The relationship to pixels is:
            pixel_diameter = kernel_size_fraction * (grid_size - 1)

        For example, with init_kernel_size=0.25 (covers 25% of the grid):
            - On 65x65 grid: 0.25 * 64 = 16 pixels diameter
            - On 129x129 grid: 0.25 * 128 = 32 pixels diameter

        Use get_kernel_size() for the fraction, get_kernel_size_pixels(grid_size) for pixel size.

    Args:
        data_dim: Number of spatial/temporal dimensions represented in the last axis of `grid`.
        num_channels: Number of feature channels to modulate.
        init_kernel_size_low: Initial kernel size (fraction of grid) for the smallest channel.
        init_kernel_size_high: Initial kernel size (fraction of grid) for the largest channel.
            Channels are initialized with kernel sizes log-spaced between low and high.
        clip_value: Threshold used to define effective kernel size (required).
            The relationship is: kernel_size_fraction = σ * √(-2 * ln(clip_value)).
        min_kernel_size: Minimum kernel size as a fraction of the grid (0.0 to 1.0).
            This sets a lower bound on the kernel size, ensuring the Gaussian doesn't become too narrow.
        max_kernel_size: Maximum kernel size as a fraction of the grid (0.0 to 1.0). Optional.
        parametrization: 'log', 'softplus', or 'direct' to ensure positivity.
    """

    def __init__(
        self,
        data_dim: int,
        num_channels: int,
        init_kernel_size_low: float,
        init_kernel_size_high: float,
        clip_value: float,
        min_kernel_size: float = 0.01,
        max_kernel_size: float | None = None,
        parametrization: str = "direct",
    ):
        """Initialize the GaussianModulationND class.

        Args:
            data_dim: Dimension of input data.
            num_channels: Number of input channels to be modulated.
            init_kernel_size_low: Initial kernel size (fraction of grid) for smallest channel.
            init_kernel_size_high: Initial kernel size (fraction of grid) for largest channel.
            clip_value: Threshold for defining effective kernel size (required).
            min_kernel_size: Minimum kernel size as fraction of grid. Default 0.01.
            max_kernel_size: Maximum kernel size as fraction of grid (optional).
            parametrization: Parametrization to use ('log', 'softplus', or 'direct').
        """
        super().__init__()
        assert parametrization in {"log", "softplus", "direct"}, (
            "parametrization must be 'log' or 'softplus' or 'direct'"
        )
        if not (0 < clip_value < 1):
            raise ValueError("clip_value must be in (0, 1)")

        self.data_dim = data_dim
        self.num_channels = num_channels
        self.parametrization = parametrization
        self.clip_value = float(clip_value)

        # Precompute the factor √(-2 * ln(clip_value)) used in kernel_size <-> std conversion
        self._cutoff_factor = math.sqrt(-2.0 * math.log(self.clip_value))

        # Store kernel size bounds and convert to std internally
        self.min_kernel_size = float(min_kernel_size)
        self.max_kernel_size = float(max_kernel_size) if max_kernel_size is not None else None

        # Internal std bounds (derived from kernel_size bounds)
        self._min_std = self.min_kernel_size / self._cutoff_factor
        self._max_std = self.max_kernel_size / self._cutoff_factor if self.max_kernel_size is not None else None

        # Convert init kernel sizes to std
        # kernel_size_fraction = σ * cutoff_factor => σ = kernel_size_fraction / cutoff_factor
        init_std_low = init_kernel_size_low / self._cutoff_factor
        init_std_high = init_kernel_size_high / self._cutoff_factor

        init_std_per_channel = torch.logspace(math.log10(init_std_low), math.log10(init_std_high), num_channels)

        # Create weight parameter
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
            self.std_param.clamp_(min=self._min_std)
            if self._max_std is not None:
                self.std_param.clamp_(max=self._max_std)

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

        # Check if the standard deviation is out of bounds.
        if self._max_std is not None and (std > self._max_std).any():
            raise ValueError(
                f"std is out of bounds: expected < {self._max_std} (max_kernel_size={self.max_kernel_size})"
            )
        if (std < self._min_std).any():
            raise ValueError(
                f"std is out of bounds: expected > {self._min_std} (min_kernel_size={self.min_kernel_size})"
            )
        return std

    def get_kernel_size(self) -> torch.Tensor:
        """Get the current effective kernel size as a fraction of the grid.

        The kernel size is defined as the fraction of the grid (0.0 to 1.0) where
        the Gaussian is above clip_value.

        The relationship is: kernel_size_fraction = σ * √(-2 * ln(clip_value))

        Returns:
            torch.Tensor: Kernel size fraction per dimension and channel, shape [data_dim, num_channels].
        """
        std = self._compute_std()  # [data_dim, num_channels]
        # kernel_size_fraction = cutoff = σ * cutoff_factor
        # where cutoff is the radius in normalized coords, and fraction = diameter / 2.0 = cutoff
        return std * self._cutoff_factor

    def get_kernel_size_pixels(self, grid_size: int | tuple[int, ...]) -> torch.Tensor:
        """Get the current effective kernel size in pixels (diameter) for a given grid size.

        Converts the kernel size fraction to pixel units based on the grid resolution.
        The formula is: pixel_diameter = kernel_size_fraction * (grid_size - 1)

        Args:
            grid_size: Size of the grid. Can be a single int (same for all dims) or a tuple
                of ints (one per dimension).

        Returns:
            torch.Tensor: Kernel diameter in pixels per dimension and channel,
                shape [data_dim, num_channels].
        """
        kernel_size_fraction = self.get_kernel_size()

        if isinstance(grid_size, int):
            grid_size = (grid_size,) * self.data_dim
        assert len(grid_size) == self.data_dim, (
            f"grid_size length ({len(grid_size)}) must match data_dim ({self.data_dim})"
        )

        # Convert to tensor for broadcasting: [data_dim, 1]
        grid_size_tensor = torch.tensor(grid_size, dtype=torch.float32, device=kernel_size_fraction.device)
        grid_size_tensor = grid_size_tensor.unsqueeze(-1)  # [data_dim, 1]

        # pixel_diameter = kernel_size_fraction * (grid_size - 1)
        return kernel_size_fraction * (grid_size_tensor - 1)

    def extra_repr(self):
        """Additional printing for the GaussianModulationND class."""
        max_info = f", max_kernel_size={self.max_kernel_size}" if self.max_kernel_size is not None else ""
        return (
            f"data_dim={self.data_dim}, num_channels={self.num_channels}, "
            f"min_kernel_size={self.min_kernel_size}{max_info}, "
            f"clip_value={self.clip_value}, parametrization='{self.parametrization}'"
        )

    def forward(self, grid: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Applies Gaussian modulation to the input tensor `x` based on the coordinates in `grid`.

        Args:
            grid (torch.Tensor): A tensor representing grid values (shape: [1, * spatial_dims, data_dim]).
                Must have dtype `torch.float32`.
            x (torch.Tensor | None): Input features to be modulated (shape: [batch_size, * spatial_dims, num_channels]). If None, only the mask is returned.

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
        gauss = exponent.exp_()

        if x is not None:
            return x * gauss.to(x.dtype)
        else:
            return gauss


class TrapezoidModulationND(torch.nn.Module):
    """Trapezoidal (linear) decay modulation across N spatial/temporal dimensions.

    For each data dimension d and channel c we learn a kernel size (as fraction of grid).
    The mask has a flat center region where mask=1, then linearly decays to 0.

    The mask function per dimension is:
        mask(x) = clamp((kernel_size - |x|) / transition_width, 0, 1)

    where:
        kernel_size = boundary radius in normalized coords [-1, 1]
        inner_edge = kernel_size - transition_width  (mask = 1 here)
        outer_edge = kernel_size                      (mask = 0 here)

    The transition_width parameter controls the sharpness of the transition:
    - Small transition_width → sharp transition (approaching hard cutoff)
    - Large transition_width → gradual transition (more smoothing)

    Kernel Size (Fraction of Grid):
        The kernel size is specified as a **fraction of the half-grid** (0.0 to 1.0).
        For a grid spanning [-1, 1], kernel_size=0.25 means the boundary is at radius 0.25.

    Args:
        data_dim: Number of spatial/temporal dimensions.
        num_channels: Number of feature channels to modulate.
        init_kernel_size_low: Initial kernel size (fraction of grid) for smallest channel.
        init_kernel_size_high: Initial kernel size (fraction of grid) for largest channel.
        transition_fraction: Transition width as fraction of kernel_radius (0 to 1).
            E.g., 0.5 means transition spans half the kernel radius. Default: 0.5.
        min_kernel_size: Minimum kernel size as a fraction of the grid. Default: 0.01.
        max_kernel_size: Maximum kernel size as a fraction of the grid. Optional.
        parametrization: 'log', 'softplus', or 'direct' to ensure positivity.
    """

    def __init__(
        self,
        data_dim: int,
        num_channels: int,
        init_kernel_size_low: float,
        init_kernel_size_high: float,
        transition_fraction: float = 0.5,
        min_kernel_size: float = 0.01,
        max_kernel_size: float | None = None,
        parametrization: str = "direct",
    ):
        """Initialize the TrapezoidModulationND class."""
        super().__init__()
        assert parametrization in {"log", "softplus", "direct"}, (
            "parametrization must be 'log' or 'softplus' or 'direct'"
        )
        if not (0 < transition_fraction <= 1):
            raise ValueError("transition_fraction must be in (0, 1]")

        self.data_dim = data_dim
        self.num_channels = num_channels
        self.parametrization = parametrization
        self.transition_fraction = float(transition_fraction)

        # Store kernel size bounds
        self.min_kernel_size = float(min_kernel_size)
        self.max_kernel_size = float(max_kernel_size) if max_kernel_size is not None else None

        # Initialize kernel sizes (log-spaced between low and high)
        init_kernel_sizes = torch.logspace(
            math.log10(init_kernel_size_low), math.log10(init_kernel_size_high), num_channels
        )

        # Create weight parameter for kernel_size directly
        # kernel_size is the learnable parameter (fraction of grid)
        init_ks = torch.stack([init_kernel_sizes] * data_dim, dim=0)  # [data_dim, num_channels]
        if parametrization == "log":
            param = init_ks.log()
        elif parametrization == "softplus":
            param = init_ks.expm1().log()
        else:  # direct
            param = init_ks
        self.kernel_size_param = torch.nn.Parameter(param)  # shape [data_dim, num_channels]

        # Add ._no_weight_decay flag to all parameters
        for p in self.parameters():
            p._no_weight_decay = True

        # Use a forward pre-hook to clamp parameters
        if self.parametrization == "direct":
            self._clamp_hook = self.register_forward_pre_hook(self._clamp_direct_param_pre_hook)

    @torch.compiler.disable
    def _clamp_direct_param_pre_hook(self, module, inputs):
        """Clamp kernel_size_param into bounds just before forward."""
        with torch.no_grad():
            self.kernel_size_param.clamp_(min=self.min_kernel_size)
            if self.max_kernel_size is not None:
                self.kernel_size_param.clamp_(max=self.max_kernel_size)

    def _compute_kernel_size(self) -> torch.Tensor:
        """Compute the kernel size for each channel.

        Returns:
            torch.Tensor: Kernel size per dimension and channel, shape [data_dim, num_channels].
        """
        ks = self.kernel_size_param.float()
        if self.parametrization == "direct":
            return ks
        elif self.parametrization == "log":
            ks = ks.exp()
        elif self.parametrization == "softplus":
            ks = torch.nn.functional.softplus(ks)
        else:
            raise ValueError(f"Invalid parametrization: {self.parametrization}")

        # Clamp to bounds
        ks = ks.clamp(min=self.min_kernel_size)
        if self.max_kernel_size is not None:
            ks = ks.clamp(max=self.max_kernel_size)
        return ks

    def get_kernel_size(self) -> torch.Tensor:
        """Get the current kernel size as a fraction of the grid.

        Returns:
            torch.Tensor: Kernel size per dimension and channel, shape [data_dim, num_channels].
        """
        return self._compute_kernel_size()

    def get_kernel_size_pixels(self, grid_size: int | tuple[int, ...]) -> torch.Tensor:
        """Get the current kernel size in pixels for a given grid size.

        Args:
            grid_size: Size of the grid. Can be int or tuple per dimension.

        Returns:
            torch.Tensor: Kernel size in pixels, shape [data_dim, num_channels].
        """
        kernel_size_fraction = self.get_kernel_size()

        if isinstance(grid_size, int):
            grid_size = (grid_size,) * self.data_dim
        assert len(grid_size) == self.data_dim

        grid_size_tensor = torch.tensor(grid_size, dtype=torch.float32, device=kernel_size_fraction.device)
        grid_size_tensor = grid_size_tensor.unsqueeze(-1)  # [data_dim, 1]

        return kernel_size_fraction * (grid_size_tensor - 1)

    def extra_repr(self):
        """Additional printing for the TrapezoidModulationND class."""
        max_info = f", max_kernel_size={self.max_kernel_size}" if self.max_kernel_size is not None else ""
        return (
            f"data_dim={self.data_dim}, num_channels={self.num_channels}, "
            f"min_kernel_size={self.min_kernel_size}{max_info}, "
            f"transition_fraction={self.transition_fraction}, parametrization='{self.parametrization}'"
        )

    def forward(self, grid: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Applies trapezoidal modulation to the input tensor.

        Args:
            grid: Grid coordinates, shape [1, *spatial_dims, data_dim]. Values in [-1, 1].
            x: Input features, shape [batch, *spatial_dims, num_channels]. If None, returns mask only.

        Returns:
            torch.Tensor: Modulated features (same shape as x) or mask if x is None.
        """
        assert grid.dtype == torch.float32, f"grid must be float32. Current dtype: {grid.dtype}"

        kernel_size = self._compute_kernel_size()  # [data_dim, num_channels]
        # kernel_size is the boundary radius (where mask = 0)
        transition_width = kernel_size * self.transition_fraction  # [data_dim, num_channels]

        # Compute trapezoidal decay: clamp((kernel_size - |x|) / transition_width, 0, 1)
        # grid: [1, *spatial_dims, data_dim] -> need to compute per-channel
        grid_abs = grid.abs()  # [1, *spatial_dims, data_dim]

        # Efficient vectorized computation:
        # Expand grid: [1, *spatial_dims, data_dim, 1], kernel_size: [data_dim, num_channels]
        grid_expanded = grid_abs.unsqueeze(-1)  # [1, *spatial_dims, data_dim, 1]
        linear_decay = ((kernel_size - grid_expanded) / transition_width).clamp(min=0.0, max=1.0)
        # linear_decay: [1, *spatial_dims, data_dim, num_channels]

        # Product over dimensions
        mask = linear_decay.prod(dim=-2)  # [1, *spatial_dims, num_channels]

        if x is not None:
            return x * mask.to(x.dtype)
        else:
            return mask


class SigmoidModulationND(torch.nn.Module):
    """Sigmoid decay modulation across N spatial/temporal dimensions.

    For each data dimension d and channel c we learn a kernel size (as fraction of grid).
    The mask uses an inverted sigmoid function centered at the kernel boundary.

    The mask function per dimension is:
        mask(x) = σ(-(|x| - kernel_size) * temperature)

    where σ is the sigmoid function and kernel_size is the boundary radius. This gives:
        - mask ≈ 1 when |x| << kernel_size (center)
        - mask = 0.5 when |x| = kernel_size (at the boundary)
        - mask ≈ 0 when |x| >> kernel_size (outside)

    Higher temperature makes the transition sharper (approaching hard cutoff).
    Lower temperature makes the transition smoother (more gradual rolloff).

    Kernel Size (Fraction of Grid):
        The kernel size is specified as a **fraction of the half-grid** (0.0 to 1.0).
        For a grid spanning [-1, 1], kernel_size=0.25 means the boundary is at radius 0.25.

    Args:
        data_dim: Number of spatial/temporal dimensions.
        num_channels: Number of feature channels to modulate.
        init_kernel_size_low: Initial kernel size (fraction of grid) for smallest channel.
        init_kernel_size_high: Initial kernel size (fraction of grid) for largest channel.
        temperature: Controls transition sharpness. Higher = sharper. Default: 10.0.
        min_kernel_size: Minimum kernel size as a fraction of the grid. Default: 0.01.
        max_kernel_size: Maximum kernel size as a fraction of the grid. Optional.
        parametrization: 'log', 'softplus', or 'direct' to ensure positivity.
    """

    def __init__(
        self,
        data_dim: int,
        num_channels: int,
        init_kernel_size_low: float,
        init_kernel_size_high: float,
        temperature: float = 10.0,
        min_kernel_size: float = 0.01,
        max_kernel_size: float | None = None,
        parametrization: str = "direct",
    ):
        """Initialize the SigmoidModulationND class."""
        super().__init__()
        assert parametrization in {"log", "softplus", "direct"}, (
            "parametrization must be 'log' or 'softplus' or 'direct'"
        )

        self.data_dim = data_dim
        self.num_channels = num_channels
        self.parametrization = parametrization
        self.temperature = float(temperature)

        # Store kernel size bounds
        self.min_kernel_size = float(min_kernel_size)
        self.max_kernel_size = float(max_kernel_size) if max_kernel_size is not None else None

        # Initialize kernel sizes (log-spaced between low and high)
        init_kernel_sizes = torch.logspace(
            math.log10(init_kernel_size_low), math.log10(init_kernel_size_high), num_channels
        )

        # Create weight parameter for kernel_size directly
        init_ks = torch.stack([init_kernel_sizes] * data_dim, dim=0)  # [data_dim, num_channels]
        if parametrization == "log":
            param = init_ks.log()
        elif parametrization == "softplus":
            param = init_ks.expm1().log()
        else:  # direct
            param = init_ks
        self.kernel_size_param = torch.nn.Parameter(param)  # shape [data_dim, num_channels]

        # Add ._no_weight_decay flag to all parameters
        for p in self.parameters():
            p._no_weight_decay = True

        # Use a forward pre-hook to clamp parameters
        if self.parametrization == "direct":
            self._clamp_hook = self.register_forward_pre_hook(self._clamp_direct_param_pre_hook)

    @torch.compiler.disable
    def _clamp_direct_param_pre_hook(self, module, inputs):
        """Clamp kernel_size_param into bounds just before forward."""
        with torch.no_grad():
            self.kernel_size_param.clamp_(min=self.min_kernel_size)
            if self.max_kernel_size is not None:
                self.kernel_size_param.clamp_(max=self.max_kernel_size)

    def _compute_kernel_size(self) -> torch.Tensor:
        """Compute the kernel size for each channel.

        Returns:
            torch.Tensor: Kernel size per dimension and channel, shape [data_dim, num_channels].
        """
        ks = self.kernel_size_param.float()
        if self.parametrization == "direct":
            return ks
        elif self.parametrization == "log":
            ks = ks.exp()
        elif self.parametrization == "softplus":
            ks = torch.nn.functional.softplus(ks)
        else:
            raise ValueError(f"Invalid parametrization: {self.parametrization}")

        # Clamp to bounds
        ks = ks.clamp(min=self.min_kernel_size)
        if self.max_kernel_size is not None:
            ks = ks.clamp(max=self.max_kernel_size)
        return ks

    def get_kernel_size(self) -> torch.Tensor:
        """Get the current kernel size as a fraction of the grid.

        Returns:
            torch.Tensor: Kernel size per dimension and channel, shape [data_dim, num_channels].
        """
        return self._compute_kernel_size()

    def get_kernel_size_pixels(self, grid_size: int | tuple[int, ...]) -> torch.Tensor:
        """Get the current kernel size in pixels for a given grid size.

        Args:
            grid_size: Size of the grid. Can be int or tuple per dimension.

        Returns:
            torch.Tensor: Kernel size in pixels, shape [data_dim, num_channels].
        """
        kernel_size_fraction = self.get_kernel_size()

        if isinstance(grid_size, int):
            grid_size = (grid_size,) * self.data_dim
        assert len(grid_size) == self.data_dim

        grid_size_tensor = torch.tensor(grid_size, dtype=torch.float32, device=kernel_size_fraction.device)
        grid_size_tensor = grid_size_tensor.unsqueeze(-1)  # [data_dim, 1]

        return kernel_size_fraction * (grid_size_tensor - 1)

    def extra_repr(self):
        """Additional printing for the SigmoidModulationND class."""
        max_info = f", max_kernel_size={self.max_kernel_size}" if self.max_kernel_size is not None else ""
        return (
            f"data_dim={self.data_dim}, num_channels={self.num_channels}, "
            f"min_kernel_size={self.min_kernel_size}{max_info}, "
            f"temperature={self.temperature}, parametrization='{self.parametrization}'"
        )

    def forward(self, grid: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Applies sigmoid modulation to the input tensor.

        Args:
            grid: Grid coordinates, shape [1, *spatial_dims, data_dim]. Values in [-1, 1].
            x: Input features, shape [batch, *spatial_dims, num_channels]. If None, returns mask only.

        Returns:
            torch.Tensor: Modulated features (same shape as x) or mask if x is None.
        """
        assert grid.dtype == torch.float32, f"grid must be float32. Current dtype: {grid.dtype}"

        kernel_size = self._compute_kernel_size()  # [data_dim, num_channels]
        # kernel_size is the boundary radius (where mask = 0.5)

        # Compute sigmoid decay: σ(-(|x| - kernel_size) * temperature)
        grid_abs = grid.abs()  # [1, *spatial_dims, data_dim]
        grid_expanded = grid_abs.unsqueeze(-1)  # [1, *spatial_dims, data_dim, 1]

        # Distance from boundary (negative inside, positive outside)
        dist_from_boundary = grid_expanded - kernel_size  # [1, *spatial_dims, data_dim, num_channels]
        sigmoid_decay = torch.sigmoid(-dist_from_boundary * self.temperature)

        # Product over dimensions
        mask = sigmoid_decay.prod(dim=-2)  # [1, *spatial_dims, num_channels]

        if x is not None:
            return x * mask.to(x.dtype)
        else:
            return mask


class SpectralMaskNDBase(torch.nn.Module):
    """Base class for spectral masks used in learnable downsampling (DiffStride-style).

    This class provides common functionality for spectral masks:
    - Stride-based parameterization with cutoff computation
    - Grid caching for efficiency
    - Crop bounds computation

    Subclasses implement the actual mask shape (Gaussian, Linear, Sigmoid, etc.).

    Args:
        data_dim: Number of spatial dimensions (1, 2, or 3).
        init_stride_value: Initial stride value(s).
            Can be a float (same for all dims) or a tuple of floats per dimension.
        min_stride_value: Minimum allowed stride (>=1.0). Limits maximum cutoff.
        max_stride_value: Maximum allowed stride. Limits minimum cutoff. None to disable.
        parametrization: How to parameterize cutoff ('direct', 'log', 'softplus').
    """

    def __init__(
        self,
        data_dim: int,
        init_stride_value: float | tuple[float, ...] = 1.0,
        min_stride_value: float = 1.0,
        max_stride_value: float | None = None,
        parametrization: str = "direct",
    ):
        """Initialize the SpectralMaskNDBase class."""
        super().__init__()
        assert parametrization in {"log", "softplus", "direct"}, (
            "parametrization must be 'log' or 'softplus' or 'direct'"
        )
        assert min_stride_value > 0.0, "min_stride_value must be > 0.0"
        assert max_stride_value is None or max_stride_value > min_stride_value, (
            "max_stride_value must be greater than min_stride_value if provided"
        )

        self.data_dim = data_dim
        self.min_stride_value = float(min_stride_value)
        self.max_stride_value = float(max_stride_value) if max_stride_value is not None else None
        self.parametrization = parametrization

        # Convert stride_init to cutoff_init: cutoff = 1 / stride (for grid in [-1, 1])
        if isinstance(init_stride_value, (int, float)):
            init_stride_value = [float(init_stride_value)] * data_dim
        else:
            init_stride_value = list(init_stride_value)
            assert len(init_stride_value) == data_dim, (
                f"init_stride_value length ({len(init_stride_value)}) must match data_dim ({data_dim})"
            )

        # Convert stride bounds to cutoff bounds (note: inverse relationship)
        # min_stride → max_cutoff (less downsampling)
        # max_stride → min_cutoff (more downsampling)
        self.max_cutoff = 1.0 / self.min_stride_value
        self.min_cutoff = 1.0 / self.max_stride_value if self.max_stride_value is not None else None

        init_cutoff = torch.tensor([1.0 / s for s in init_stride_value], dtype=torch.float32)

        # Apply parametrization transform
        if parametrization == "log":
            param = init_cutoff.log()
        elif parametrization == "softplus":
            param = init_cutoff.expm1().log()
        else:  # direct
            param = init_cutoff

        self.cutoff_param = torch.nn.Parameter(param)  # shape: [data_dim]

        # Disable weight decay for this parameter
        for p in self.parameters():
            p._no_weight_decay = True

        # Use a forward pre-hook to clamp cutoff_param for direct parametrization
        if self.parametrization == "direct":
            self._clamp_hook = self.register_forward_pre_hook(self._clamp_direct_cutoff_param_pre_hook)

        # Grid cache to avoid regenerating when spatial_dims and bounds don't change
        self._cached_spatial_dims: tuple[int, ...] | None = None
        self._cached_bounds: list[tuple[int, int]] | None = None
        self._cached_grid: torch.Tensor | None = None

    @torch.compiler.disable
    def _clamp_direct_cutoff_param_pre_hook(self, module, inputs):
        """Clamp cutoff_param into [min_cutoff, max_cutoff] just before forward."""
        with torch.no_grad():
            if self.min_cutoff is not None:
                self.cutoff_param.clamp_(min=self.min_cutoff)
            if self.max_cutoff is not None:
                self.cutoff_param.clamp_(max=self.max_cutoff)

    def _compute_cutoff(self) -> torch.Tensor:
        """Compute the cutoff from the parameterized value.

        Returns:
            torch.Tensor: The cutoff for each dimension, shape [data_dim].
        """
        cutoff = self.cutoff_param.float()  # [data_dim]
        if self.parametrization == "direct":
            return cutoff
        elif self.parametrization == "log":
            cutoff = cutoff.exp()
        elif self.parametrization == "softplus":
            cutoff = torch.nn.functional.softplus(cutoff)
        else:
            raise ValueError(f"Invalid parametrization: {self.parametrization}")

        # Check bounds for non-direct parametrizations
        if self.min_cutoff is not None and (cutoff < self.min_cutoff).any():
            raise ValueError(f"cutoff below min_cutoff: {cutoff.tolist()} < {self.min_cutoff}")
        if self.max_cutoff is not None and (cutoff > self.max_cutoff).any():
            raise ValueError(f"cutoff above max_cutoff: {cutoff.tolist()} > {self.max_cutoff}")
        return cutoff

    def get_stride(self) -> torch.Tensor:
        """Get the current effective stride based on learned cutoff.

        The stride is computed as: stride = 1 / cutoff.

        Returns:
            torch.Tensor: Stride per dimension, shape [data_dim].
        """
        cutoff = self._compute_cutoff()
        return 1.0 / cutoff

    def extra_repr(self) -> str:
        """Additional printing for the SpectralMaskNDBase class."""
        return (
            f"data_dim={self.data_dim}, "
            f"min_stride={self.min_stride_value}, max_stride={self.max_stride_value}, "
            f"parametrization='{self.parametrization}'"
        )

    def _compute_crop_bounds(self, cutoff: torch.Tensor, spatial_dims: tuple[int, ...]) -> list[tuple[int, int]]:
        """Compute crop start/end indices for each dimension using analytical bounds.

        For a grid spanning [-1, 1] with N points (linspace), the coordinates are:
            coords[i] = -1 + i * 2/(N-1)  for i = 0, ..., N-1

        For symmetric cropping (|coord| < cutoff):
            start_idx = ceil((1 - cutoff) * (N-1) / 2)
            end_idx = floor((1 + cutoff) * (N-1) / 2) + 1

        For Hermitian cropping (0 <= coord < cutoff, last dim only):
            start_idx = floor((N-1) / 2)  (first non-negative index)
            end_idx = floor((1 + cutoff) * (N-1) / 2) + 1

        Args:
            cutoff: Cutoff values per dimension, shape [data_dim].
            spatial_dims: Tuple of spatial dimension sizes.

        Returns:
            List of (start_idx, end_idx) tuples for each dimension.
        """
        bounds = []
        cutoff_clamped = cutoff.clamp(max=1.0)  # Clamp cutoff to valid range

        for d in range(self.data_dim):
            N = spatial_dims[d]
            c = cutoff_clamped[d].item()
            is_last_dim = d == self.data_dim - 1

            if is_last_dim:
                # Hermitian: keep coords in [0, cutoff)
                start_idx = (N - 1) // 2
                end_idx = min(N, math.floor((1.0 + c) * (N - 1) / 2) + 1)
            else:
                # Symmetric: keep coords in (-cutoff, cutoff)
                start_idx = max(0, math.ceil((1.0 - c) * (N - 1) / 2))
                end_idx = min(N, math.floor((1.0 + c) * (N - 1) / 2) + 1)

            # Ensure at least one element (DC component)
            if end_idx <= start_idx:
                center = (N - 1) // 2
                start_idx, end_idx = center, center + 1

            bounds.append((start_idx, end_idx))

        return bounds

    def _get_cached_grid(
        self,
        spatial_dims: tuple[int, ...],
        bounds: list[tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Get the cropped grid, using cache if available."""
        cache_valid = (
            self._cached_grid is not None
            and self._cached_spatial_dims == spatial_dims
            and self._cached_bounds == bounds
            and self._cached_grid.device == device
        )

        if cache_valid:
            return self._cached_grid

        grid = self._generate_cropped_grid(spatial_dims, bounds, device)

        self._cached_spatial_dims = spatial_dims
        self._cached_bounds = bounds
        self._cached_grid = grid

        return grid

    def _generate_cropped_grid(
        self,
        spatial_dims: tuple[int, ...],
        bounds: list[tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate a coordinate grid containing only the cropped region."""
        coords_1d = []
        for d, (start_idx, end_idx) in enumerate(bounds):
            N = spatial_dims[d]
            indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.float32)
            coords_d = -1.0 + indices * (2.0 / (N - 1)) if N > 1 else torch.zeros_like(indices)
            coords_1d.append(coords_d)

        if self.data_dim == 1:
            grid = coords_1d[0].unsqueeze(-1)
        else:
            mesh = torch.meshgrid(*coords_1d, indexing="ij")
            grid = torch.stack(mesh, dim=-1)

        return grid.unsqueeze(0)

    def clear_cache(self) -> None:
        """Clear the cached grid."""
        self._cached_spatial_dims = None
        self._cached_bounds = None
        self._cached_grid = None

    def _compute_mask(self, grid: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Compute the mask values on the given grid.

        This method should be overridden by subclasses to implement specific mask shapes.

        Args:
            grid: Coordinate grid, shape [1, *cropped_dims, data_dim].
            cutoff: Cutoff values per dimension, shape [data_dim].

        Returns:
            torch.Tensor: Mask values, shape [1, *cropped_dims].
        """
        raise NotImplementedError("Subclasses must implement _compute_mask")

    def forward(self, spatial_dims: tuple[int, ...], device: torch.device | None = None) -> torch.Tensor:
        """Compute the spectral mask for given spatial dimensions.

        Args:
            spatial_dims: Tuple of spatial dimension sizes of the input tensor.
            device: Device to create the mask on. If None, uses the parameter's device.

        Returns:
            torch.Tensor: Cropped mask, shape [1, *cropped_spatial_dims].
        """
        assert len(spatial_dims) == self.data_dim, (
            f"spatial_dims length ({len(spatial_dims)}) must match data_dim ({self.data_dim})"
        )

        if device is None:
            device = self.cutoff_param.device

        cutoff = self._compute_cutoff()
        bounds = self._compute_crop_bounds(cutoff, spatial_dims)
        cropped_grid = self._get_cached_grid(spatial_dims, bounds, device)

        return self._compute_mask(cropped_grid, cutoff)


class SpectralGaussianMaskND(SpectralMaskNDBase):
    """Gaussian spectral mask for learnable downsampling (DiffStride-style).

    This mask operates in the frequency domain to perform learnable downsampling.
    A single Gaussian std is learned per spatial dimension (shared across all channels).

    The relationship between std (σ) and effective stride (S) is derived from the clip value:
        cutoff = σ * √(-2 * ln(clip_value))
        stride = 1 / cutoff  (for grid normalized to [-1, 1])

    Therefore:
        σ = 1 / (stride * √(-2 * ln(clip_value)))

    The mask is cropped based on the clip value threshold, keeping only the region
    where the Gaussian is above the threshold. For the last dimension, only positive
    frequencies are kept (Hermitian symmetry for rfft).

    Args:
        data_dim: Number of spatial dimensions (1, 2, or 3).
        clip_value: Threshold below which the mask is considered zero for cropping.
            Determines the relationship between std and stride.
        init_stride_value: Initial stride value(s) used to compute initial std.
            Can be a float (same for all dims) or a tuple of floats per dimension.
        min_stride_value: Minimum allowed stride (>1.0). Limits maximum std (least downsampling).
        max_stride_value: Maximum allowed stride. Limits minimum std (most downsampling). None to disable.
        parametrization: How to parameterize std ('direct', 'log', 'softplus').
    """

    def __init__(
        self,
        data_dim: int,
        clip_value: float = 0.1,  # 10% of the maximum value of the Gaussian mask
        init_stride_value: float | tuple[float, ...] = 1.0,
        min_stride_value: float = 1.0,
        max_stride_value: float | None = None,
        parametrization: str = "direct",
    ):
        """Initialize the SpectralGaussianMaskND class."""
        assert 0 < clip_value < 1, "clip_value must be in (0, 1)"
        self.clip_value = float(clip_value)
        # Precompute the factor √(-2 * ln(clip_value)) used in std <-> cutoff conversion
        self._gaussian_cutoff_factor = math.sqrt(-2.0 * math.log(self.clip_value))

        super().__init__(
            data_dim=data_dim,
            init_stride_value=init_stride_value,
            min_stride_value=min_stride_value,
            max_stride_value=max_stride_value,
            parametrization=parametrization,
        )

    def _compute_std(self) -> torch.Tensor:
        """Compute the Gaussian std from the cutoff.

        The relationship is: cutoff = σ * √(-2 * ln(clip_value))
        Therefore: σ = cutoff / √(-2 * ln(clip_value))

        Returns:
            torch.Tensor: The std for each dimension, shape [data_dim].
        """
        cutoff = self._compute_cutoff()
        return cutoff / self._gaussian_cutoff_factor

    @property
    def std_param(self) -> torch.Tensor:
        """Backwards-compatible property to get the computed std values.

        Note: This is computed from cutoff_param, not a learnable parameter itself.
        For the actual learnable parameter, use cutoff_param.
        """
        return self._compute_std()

    def extra_repr(self) -> str:
        """Additional printing for the SpectralGaussianMaskND class."""
        return (
            f"data_dim={self.data_dim}, clip_value={self.clip_value}, "
            f"min_stride={self.min_stride_value}, max_stride={self.max_stride_value}, "
            f"parametrization='{self.parametrization}'"
        )

    def _compute_mask(self, grid: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Compute the Gaussian mask values on the given grid.

        Args:
            grid: Coordinate grid, shape [1, *cropped_dims, data_dim].
            cutoff: Cutoff values per dimension, shape [data_dim].

        Returns:
            torch.Tensor: Mask values, shape [1, *cropped_dims].
        """
        # Compute std from cutoff: σ = cutoff / √(-2 * ln(clip_value))
        std = cutoff / self._gaussian_cutoff_factor

        # Compute the Gaussian mask on the cropped grid (vectorized)
        # mask = Π_d exp(-0.5 * (grid_d / σ_d)²) = exp(-0.5 * Σ_d (grid_d / σ_d)²)
        exponent = -0.5 * (grid.square() * std.square().reciprocal()).sum(dim=-1)
        mask = exponent.exp()  # [1, *cropped_spatial_dims]
        return mask


class SpectralLinearMaskND(SpectralMaskNDBase):
    """Linear (trapezoidal) spectral mask for learnable downsampling (DiffStride-style).

    This mask implements the DiffStride approach with a trapezoidal shape:
    - Flat passband (mask = 1) for |x| < inner_edge
    - Linear transition from 1 to 0 between inner_edge and outer_edge (= cutoff)
    - Stopband (mask = 0) for |x| > cutoff

    The mask function per dimension is:
        mask(x) = clamp((cutoff - |x|) / transition_width, 0, 1)

    where:
        inner_edge = cutoff - transition_width  (mask = 1 here)
        outer_edge = cutoff                      (mask = 0 here)

    The `transition_width` parameter controls the sharpness of the transition:
    - Small transition_width → sharp transition (approaching hard cutoff)
    - Large transition_width → gradual transition (more smoothing)

    If transition_width >= cutoff, the mask degrades to a triangular shape
    (linear from 1 at center to 0 at cutoff, like the original implementation).

    The relationship between cutoff and stride is:
        stride = 1 / cutoff  (for grid normalized to [-1, 1])

    Args:
        data_dim: Number of spatial dimensions (1, 2, or 3).
        transition_width: Width of the linear transition region (in normalized coords).
            Controls sharpness: smaller = sharper. Default: 0.1.
            Can also be specified as a fraction of cutoff using `transition_fraction`.
        transition_fraction: Alternative to transition_width. Specifies transition width
            as a fraction of the cutoff (0 to 1). E.g., 0.5 means transition spans half
            the cutoff region. Mutually exclusive with transition_width.
        init_stride_value: Initial stride value(s).
            Can be a float (same for all dims) or a tuple of floats per dimension.
        min_stride_value: Minimum allowed stride (>=1.0). Limits maximum cutoff.
        max_stride_value: Maximum allowed stride. Limits minimum cutoff. None to disable.
        parametrization: How to parameterize cutoff ('direct', 'log', 'softplus').

    Reference:
        "Learning strides in convolutional neural networks" (DiffStride)
        https://arxiv.org/abs/2202.01653
    """

    def __init__(
        self,
        data_dim: int,
        transition_width: float | None = None,
        transition_fraction: float | None = None,
        init_stride_value: float | tuple[float, ...] = 1.0,
        min_stride_value: float = 1.0,
        max_stride_value: float | None = None,
        parametrization: str = "direct",
    ):
        """Initialize the SpectralLinearMaskND class."""
        super().__init__(
            data_dim=data_dim,
            init_stride_value=init_stride_value,
            min_stride_value=min_stride_value,
            max_stride_value=max_stride_value,
            parametrization=parametrization,
        )

        # Handle transition_width vs transition_fraction
        if transition_width is not None and transition_fraction is not None:
            raise ValueError("Cannot specify both transition_width and transition_fraction")

        if transition_fraction is not None:
            if not (0 < transition_fraction <= 1):
                raise ValueError("transition_fraction must be in (0, 1]")
            self.transition_fraction = float(transition_fraction)
            self.transition_width = None
        elif transition_width is not None:
            if transition_width <= 0:
                raise ValueError("transition_width must be positive")
            self.transition_width = float(transition_width)
            self.transition_fraction = None
        else:
            # Default: use fraction-based (50% of cutoff)
            self.transition_fraction = 0.5
            self.transition_width = None

    def _get_transition_width(self, cutoff: torch.Tensor) -> torch.Tensor:
        """Get the transition width, either fixed or as fraction of cutoff."""
        if self.transition_width is not None:
            return torch.tensor(self.transition_width, device=cutoff.device, dtype=cutoff.dtype)
        else:
            return cutoff * self.transition_fraction

    def extra_repr(self) -> str:
        """Additional printing for the SpectralLinearMaskND class."""
        tw_str = (
            f"transition_width={self.transition_width}"
            if self.transition_width
            else f"transition_fraction={self.transition_fraction}"
        )
        return (
            f"data_dim={self.data_dim}, {tw_str}, "
            f"min_stride={self.min_stride_value}, max_stride={self.max_stride_value}, "
            f"parametrization='{self.parametrization}'"
        )

    def _compute_mask(self, grid: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Compute the linear (trapezoidal) mask values on the given grid.

        mask(x) = clamp((cutoff - |x|) / transition_width, 0, 1)

        Args:
            grid: Coordinate grid, shape [1, *cropped_dims, data_dim].
            cutoff: Cutoff values per dimension, shape [data_dim].

        Returns:
            torch.Tensor: Mask values, shape [1, *cropped_dims].
        """
        transition_width = self._get_transition_width(cutoff)

        # Compute trapezoidal decay per dimension: clamp((cutoff - |x|) / transition_width, 0, 1)
        # grid: [1, *cropped_spatial_dims, data_dim], cutoff: [data_dim]
        linear_decay = ((cutoff - grid.abs()) / transition_width).clamp(min=0.0, max=1.0)

        # Product over dimensions
        mask = linear_decay.prod(dim=-1)  # [1, *cropped_dims]
        return mask


class SpectralSigmoidMaskND(SpectralMaskNDBase):
    """Sigmoid spectral mask for learnable downsampling with smooth transitions.

    This mask uses an inverted sigmoid function centered at the cutoff boundary
    to provide smooth transitions with controllable sharpness via temperature.

    The mask function is:
        mask(x) = Π_d σ(-(|x_d| - cutoff_d) * temperature)

    where σ is the sigmoid function. This gives:
        - mask ≈ 1 when |x_d| << cutoff_d (center/low frequencies)
        - mask = 0.5 when |x_d| = cutoff_d (at the cutoff boundary)
        - mask ≈ 0 when |x_d| >> cutoff_d (high frequencies, outside crop region)

    Higher temperature makes the transition sharper (approaching hard cutoff).
    Lower temperature makes the transition smoother (more gradual rolloff).

    The relationship between cutoff and stride is:
        stride = 1 / cutoff  (for grid normalized to [-1, 1])

    Args:
        data_dim: Number of spatial dimensions (1, 2, or 3).
        temperature: Controls transition sharpness. Higher = sharper. Default: 10.0.
        init_stride_value: Initial stride value(s).
            Can be a float (same for all dims) or a tuple of floats per dimension.
        min_stride_value: Minimum allowed stride (>=1.0). Limits maximum cutoff.
        max_stride_value: Maximum allowed stride. Limits minimum cutoff. None to disable.
        parametrization: How to parameterize cutoff ('direct', 'log', 'softplus').
    """

    def __init__(
        self,
        data_dim: int,
        temperature: float = 10.0,
        init_stride_value: float | tuple[float, ...] = 1.0,
        min_stride_value: float = 1.0,
        max_stride_value: float | None = None,
        parametrization: str = "direct",
    ):
        """Initialize the SpectralSigmoidMaskND class."""
        super().__init__(
            data_dim=data_dim,
            init_stride_value=init_stride_value,
            min_stride_value=min_stride_value,
            max_stride_value=max_stride_value,
            parametrization=parametrization,
        )
        self.temperature = float(temperature)

    def extra_repr(self) -> str:
        """Additional printing for the SpectralSigmoidMaskND class."""
        return (
            f"data_dim={self.data_dim}, temperature={self.temperature}, "
            f"min_stride={self.min_stride_value}, max_stride={self.max_stride_value}, "
            f"parametrization='{self.parametrization}'"
        )

    def _compute_mask(self, grid: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Compute the sigmoid mask values on the given grid.

        mask(x) = Π_d σ(-(|x_d| - cutoff_d) * temperature)

        Args:
            grid: Coordinate grid, shape [1, *cropped_dims, data_dim].
            cutoff: Cutoff values per dimension, shape [data_dim].

        Returns:
            torch.Tensor: Mask values, shape [1, *cropped_dims].
        """
        # Compute sigmoid decay per dimension: σ(-(|x_d| - cutoff_d) * temperature)
        # grid: [1, *cropped_spatial_dims, data_dim], cutoff: [data_dim]
        dist_from_cutoff = grid.abs() - cutoff  # distance from cutoff (negative inside, positive outside)
        sigmoid_decay = torch.sigmoid(-dist_from_cutoff * self.temperature)  # [1, *cropped_dims, data_dim]

        # Product over dimensions
        mask = sigmoid_decay.prod(dim=-1)  # [1, *cropped_dims]
        return mask


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

    # Test SpectralGaussianMaskND
    print("\n" + "=" * 60)
    print("Testing SpectralGaussianMaskND")
    print("=" * 60)

    # Test with different stride initializations
    x_shape = (64, 64)
    for init_stride in [1.5, 2.0, 4.0]:
        spectral_mask = SpectralGaussianMaskND(
            data_dim=2,
            clip_value=0.1,
            init_stride_value=init_stride,
            min_stride_value=1.0,
            max_stride_value=None,
            parametrization="direct",
        )
        print(f"\ninit_stride_value={init_stride}")
        print(f"  cutoff_param: {spectral_mask.cutoff_param.detach().cpu().tolist()}")
        print(f"  get_stride(): {spectral_mask.get_stride().detach().cpu().tolist()}")

        # Get the mask using spatial_dims (no external grid needed)
        mask = spectral_mask(x_shape)
        print(f"  Input x_shape: {x_shape}")
        print(f"  Output mask shape: {mask.shape}")
        print(f"  Mask min/max: {mask.min().item():.4f} / {mask.max().item():.4f}")

    # Visualize all spectral mask types
    def visualize_spectral_mask(mask_module, x_shape, ax, title):
        """Helper to visualize a spectral mask."""
        mask = mask_module(x_shape)
        # Pad mask to full grid size for visualization
        full_mask = torch.zeros(x_shape[0], x_shape[1])
        mask_h, mask_w = mask.shape[1], mask.shape[2]
        # Center the mask (for height) and align to left (for width, Hermitian)
        h_start = (x_shape[0] - mask_h) // 2
        full_mask[h_start : h_start + mask_h, :mask_w] = mask[0]

        im = ax.imshow(full_mask.detach().cpu(), cmap="viridis", origin="lower", extent=[-1, 1, -1, 1])
        ax.set_title(title)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        return im

    # Compare all mask types at same stride
    fig_compare, axes_compare = plt.subplots(3, 3, figsize=(12, 12))
    fig_compare.suptitle("Spectral Masks Comparison (Gaussian, Linear, Sigmoid)")

    strides = [1.5, 2.0, 4.0]
    mask_types = [
        ("Gaussian", lambda s: SpectralGaussianMaskND(data_dim=2, clip_value=0.1, init_stride_value=s)),
        (
            "Linear (frac=0.5)",
            lambda s: SpectralLinearMaskND(data_dim=2, transition_fraction=0.5, init_stride_value=s),
        ),
        ("Sigmoid (T=10)", lambda s: SpectralSigmoidMaskND(data_dim=2, temperature=10.0, init_stride_value=s)),
    ]

    for row, (mask_name, mask_factory) in enumerate(mask_types):
        for col, stride in enumerate(strides):
            ax = axes_compare[row, col]
            mask_module = mask_factory(stride)
            im = visualize_spectral_mask(mask_module, x_shape, ax, f"{mask_name}, stride={stride}")
            fig_compare.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_compare.savefig(script_dir / "spectral_masks_comparison.png", dpi=150, bbox_inches="tight")

    # Test SpectralLinearMaskND
    print("\n" + "=" * 60)
    print("Testing SpectralLinearMaskND")
    print("=" * 60)

    for init_stride in [1.5, 2.0, 4.0]:
        spectral_mask = SpectralLinearMaskND(
            data_dim=2,
            transition_fraction=0.5,
            init_stride_value=init_stride,
            min_stride_value=1.0,
            parametrization="direct",
        )
        mask = spectral_mask(x_shape)
        print(f"\ninit_stride_value={init_stride}, transition_fraction=0.5")
        print(f"  cutoff_param: {spectral_mask.cutoff_param.detach().cpu().tolist()}")
        print(f"  get_stride(): {spectral_mask.get_stride().detach().cpu().tolist()}")
        print(f"  Output mask shape: {mask.shape}")
        print(f"  Mask min/max: {mask.min().item():.4f} / {mask.max().item():.4f}")

    # Test different transition fractions
    print("\n  Testing different transition fractions (stride=2.0):")
    for frac in [0.2, 0.5, 0.8, 1.0]:
        spectral_mask = SpectralLinearMaskND(
            data_dim=2, transition_fraction=frac, init_stride_value=2.0, parametrization="direct"
        )
        mask = spectral_mask(x_shape)
        print(
            f"    frac={frac}: mask min/max/mean = {mask.min().item():.4f} / {mask.max().item():.4f} / {mask.mean().item():.4f}"
        )

    # Test SpectralSigmoidMaskND with different temperatures
    print("\n" + "=" * 60)
    print("Testing SpectralSigmoidMaskND")
    print("=" * 60)

    for temperature in [5.0, 10.0, 20.0, 50.0]:
        spectral_mask = SpectralSigmoidMaskND(
            data_dim=2, temperature=temperature, init_stride_value=2.0, min_stride_value=1.0, parametrization="direct"
        )
        mask = spectral_mask(x_shape)
        print(f"\ntemperature={temperature}, stride=2.0")
        print(f"  cutoff_param: {spectral_mask.cutoff_param.detach().cpu().tolist()}")
        print(f"  Output mask shape: {mask.shape}")
        print(f"  Mask min/max: {mask.min().item():.4f} / {mask.max().item():.4f}")

    # Compare Sigmoid at different temperatures
    fig_temps, axes_temps = plt.subplots(1, 4, figsize=(16, 4))
    fig_temps.suptitle("SpectralSigmoidMaskND at Different Temperatures (stride=2.0)")

    for i, temperature in enumerate([5.0, 10.0, 20.0, 50.0]):
        ax = axes_temps[i]
        mask_module = SpectralSigmoidMaskND(data_dim=2, temperature=temperature, init_stride_value=2.0)
        im = visualize_spectral_mask(mask_module, x_shape, ax, f"T={temperature}")
        fig_temps.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_temps.savefig(script_dir / "spectral_sigmoid_temperatures.png", dpi=150, bbox_inches="tight")

    # Compare Linear at different transition fractions
    fig_fracs, axes_fracs = plt.subplots(1, 4, figsize=(16, 4))
    fig_fracs.suptitle("SpectralLinearMaskND at Different Transition Fractions (stride=2.0)")

    for i, frac in enumerate([0.2, 0.5, 0.8, 1.0]):
        ax = axes_fracs[i]
        mask_module = SpectralLinearMaskND(data_dim=2, transition_fraction=frac, init_stride_value=2.0)
        im = visualize_spectral_mask(mask_module, x_shape, ax, f"frac={frac}")
        fig_fracs.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_fracs.savefig(script_dir / "spectral_linear_transition_fractions.png", dpi=150, bbox_inches="tight")

    # 1D cross-section comparison
    fig_1d, ax_1d = plt.subplots(1, 1, figsize=(12, 7))
    fig_1d.suptitle("1D Cross-Section of Spectral Masks (stride=2.0)")

    stride = 2.0
    x_1d = torch.linspace(-1, 1, 200)

    # Gaussian
    gaussian_mask = SpectralGaussianMaskND(data_dim=1, clip_value=0.1, init_stride_value=stride)
    cutoff_gauss = gaussian_mask._compute_cutoff().detach()
    std = cutoff_gauss / gaussian_mask._gaussian_cutoff_factor
    y_gauss = torch.exp(-0.5 * (x_1d / std).pow(2))
    ax_1d.plot(x_1d.numpy(), y_gauss.detach().numpy(), label="Gaussian (clip=0.1)", linewidth=2)

    # Linear (trapezoidal) at different transition fractions
    for frac in [0.2, 0.5, 1.0]:
        linear_mask = SpectralLinearMaskND(data_dim=1, transition_fraction=frac, init_stride_value=stride)
        cutoff_linear = linear_mask._compute_cutoff().detach()
        transition_width = cutoff_linear * frac
        y_linear = ((cutoff_linear - x_1d.abs()) / transition_width).clamp(min=0.0, max=1.0)
        label = f"Linear (frac={frac})" if frac < 1.0 else "Linear (frac=1.0, triangular)"
        ax_1d.plot(x_1d.numpy(), y_linear.detach().numpy(), label=label, linewidth=2, linestyle="-.")

    # Sigmoid at different temperatures
    for temp in [5.0, 10.0, 20.0]:
        sigmoid_mask = SpectralSigmoidMaskND(data_dim=1, temperature=temp, init_stride_value=stride)
        cutoff_sig = sigmoid_mask._compute_cutoff().detach()
        y_sig = torch.sigmoid(-(x_1d.abs() - cutoff_sig) * temp)
        ax_1d.plot(x_1d.numpy(), y_sig.detach().numpy(), label=f"Sigmoid (T={temp})", linewidth=2, linestyle="--")

    ax_1d.axvline(-1 / stride, color="gray", linestyle=":", alpha=0.7, label=f"cutoff (1/stride={1 / stride:.2f})")
    ax_1d.axvline(1 / stride, color="gray", linestyle=":", alpha=0.7)
    ax_1d.set_xlabel("Normalized Frequency")
    ax_1d.set_ylabel("Mask Value")
    ax_1d.legend()
    ax_1d.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_1d.savefig(script_dir / "spectral_masks_1d_comparison.png", dpi=150, bbox_inches="tight")

    plt.show()
