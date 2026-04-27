"""KAN-based implicit kernel for ND convolutions (no positional embedding).

Drop-in replacement for SIRENKernelND: same interface
``forward(seq_lens, conditioning=None) -> (kernel, grid)``.

Raw grid coordinates in [-1, 1] are fed directly into ``num_layers``
KANLinear (B-spline) layers from warpKAN.  No positional embedding or
linear output projection is used.  All computation is in float32 as
required by KANLinear.
"""

import torch
from einops import rearrange

from warpkan.torch_ext.kanlinear import KANLinear


class KANKernelND(torch.nn.Module):
    """Kernel parameterized entirely by KAN (B-spline) layers.

    Raw coordinates in [-1, 1] are passed through ``num_layers`` KANLinear
    layers with no positional embedding and no separate linear output
    projection.  The final KANLinear maps directly to ``out_dim``.

    All forward computation stays in float32 (KANLinear requirement).

    B-spline grid spacing is fixed at dx = 1 / L_cache so the knot density
    matches the spatial resolution of the kernel grid.

    Args:
        out_dim: Number of output channels for the generated kernel.
        data_dim: Number of spatial input dimensions (e.g. 2 for images).
            Also the ``in_dim`` of the first KANLinear layer.
        mlp_hidden_dim: Hidden width of all KAN layers except the last.
        num_layers: Total number of KANLinear layers.  Must be >= 2.
            Layer dims: data_dim → mlp_hidden_dim (×num_layers-1) → out_dim.
        L_cache: Grid cache size (maximum supported spatial extent before
            cache growth). Also sets the B-spline knot spacing: dx = 1/L_cache.
        order: B-spline order (default 3 = cubic).
        grid_range: B-spline knot domain for all layers.  Defaults to
            [-5.0, 5.0]; grid_num is computed as
            ``int((grid_range[1] - grid_range[0]) * L_cache)`` to achieve
            dx = 1 / L_cache.
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        L_cache: int,
        order: int = 3,
        grid_range: list[float] | None = None,
        base_activation: type[torch.nn.Module] = torch.nn.SiLU,
    ):
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        if grid_range is None:
            grid_range = [-5.0, 5.0]

        # dx = 1 / L_cache  →  grid_num = (high - low) * L_cache
        grid_num = int((grid_range[1] - grid_range[0]) * L_cache)

        super().__init__()
        self.out_dim = out_dim
        self.data_dim = data_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_layers = num_layers
        self.L_cache = L_cache

        # Grid cache — same construction as SIRENPositionalEmbeddingND.
        with torch.inference_mode(False):
            with torch.no_grad():
                t = torch.linspace(-1, 1, 2 * L_cache - 1, dtype=torch.float32)
                grid_cache = rearrange(
                    torch.stack(torch.meshgrid(*[t] * data_dim, indexing="ij"), dim=-1),
                    "... -> 1 ...",
                )
        self.register_buffer("grid_cache", grid_cache, persistent=False)
        self.step_size = 1.0 / (L_cache - 1)

        # KAN layers: data_dim → mlp_hidden_dim (×num_layers-1) → out_dim.
        self.kan_layers = torch.nn.ModuleList()
        in_d = data_dim
        for i in range(num_layers):
            out_d = out_dim if i == num_layers - 1 else mlp_hidden_dim
            self.kan_layers.append(
                KANLinear(
                    in_dim=in_d,
                    out_dim=out_d,
                    order=order,
                    grid_num=grid_num,
                    grid_range=grid_range,
                    has_mlp=True,
                    enable_standalone_scale_spline=True,
                    base_activation=base_activation,
                )
            )
            in_d = mlp_hidden_dim

        for param in self.parameters():
            param._no_weight_decay = True

    def _get_grid(self, seq_lens: tuple[int, ...]) -> torch.Tensor:
        assert len(seq_lens) == self.data_dim
        seq_len = max(seq_lens)

        if self.L_cache < seq_len:
            with torch.inference_mode(False):
                with torch.no_grad():
                    max_limit = 1.0 + self.step_size * (seq_len - self.L_cache)
                    t = torch.linspace(
                        -max_limit,
                        max_limit,
                        2 * seq_len - 1,
                        device=self.grid_cache.device,
                        dtype=torch.float32,
                    )
                    self.grid_cache = rearrange(
                        torch.stack(
                            torch.meshgrid(*[t] * self.data_dim, indexing="ij"), dim=-1
                        ),
                        "... -> 1 ...",
                    )
                    self.L_cache = seq_len

        assert self.grid_cache.dtype == torch.float32
        offsets = [self.L_cache - s for s in seq_lens]
        slices = [slice(off, off + (s * 2) - 1) for off, s in zip(offsets, seq_lens)]
        return self.grid_cache[:, *slices]  # [1, *spatial, data_dim]

    def forward(
        self, seq_lens: tuple[int, ...], conditioning: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the KAN kernel for the given spatial grid.

        Args:
            seq_lens: Spatial extents per dimension.
            conditioning: Unused; accepted for API compatibility with SIRENKernelND.

        Returns:
            (kernel, grid): kernel shape [1, *spatial, out_dim] in float32,
                grid shape [1, *spatial, data_dim] in float32.
        """
        grid = self._get_grid(seq_lens)  # [1, *spatial, data_dim], float32
        h = grid
        for kan in self.kan_layers:
            h = kan(h)
        return h, grid
