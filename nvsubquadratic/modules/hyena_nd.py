# David W. Romero, 2025-09-09

"""Hyena-style global convolutional mixer implementation for ND signals."""

import math

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.utils import qk_norm, rope


class Hyena(torch.nn.Module):
    """Hyena-style global convolutional mixer."""

    def __init__(
        self,
        global_conv_cfg: LazyConfig,
        short_conv_cfg: LazyConfig,
        gate_nonlinear_cfg: LazyConfig,
        pixelhyena_norm_cfg: LazyConfig,
        apply_qk_norm: bool,
        use_rope: bool,
        rope_base: float = 10000.0,
    ):
        """Initialize the Hyena-style global convolutional mixer.

        Args:
            global_conv_cfg: LazyConfig - LazyConfig for the global convolutional layer.
            short_conv_cfg: LazyConfig - LazyConfig for the short convolutional layer.
            gate_nonlinear_cfg: LazyConfig - LazyConfig for the gate nonlinear layer.
            pixelhyena_norm_cfg: LazyConfig - LazyConfig for the pixelhyena normalization layer. Use torch.nn.Identity for no normalization.
            apply_qk_norm: bool - Whether to apply normalization to the query and key.
            use_rope: bool - Whether to use RoPE.
            rope_base: float - The base of the RoPE (default: 10000.0).

        Raises:
            AssertionError: If the short conv is not an instance of torch.nn.ConvNd (1d, 2d, or 3d).
        """
        super().__init__()

        # Core global convs: feature and gate branches
        self.global_conv = instantiate(global_conv_cfg)
        self.short_conv = instantiate(short_conv_cfg)
        assert isinstance(self.short_conv, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)), (
            f"Short conv must be an instance of torch.nn.ConvNd (1d, 2d, or 3d). Current type: {type(self.short_conv)}"
        )

        # Initialize the short conv
        bound = math.sqrt(1.0 / math.prod(self.short_conv.kernel_size))
        torch.nn.init.uniform_(self.short_conv.weight, -bound, bound)
        if self.short_conv.bias is not None:
            torch.nn.init.zeros_(self.short_conv.bias)

        # Nonlinear gate
        self.gate_nonlinear = instantiate(gate_nonlinear_cfg)

        # Pixelhyena normalization (use torch.nn.Identity for no normalization)
        self.pixelhyena_norm = instantiate(pixelhyena_norm_cfg)
        # Exclude self.pixelhyena_norm from the parameter group with weight decay
        for param in self.pixelhyena_norm.parameters():
            param._no_wd = True

        # QK Normalization
        self.apply_qk_norm = apply_qk_norm

        # RoPE
        self.use_rope = use_rope
        self.rope_base = rope_base
        # RoPE caches (keyed by shape, dtype, device)
        if self.use_rope:
            self._rope1d_cache = {}
            self._rope2d_cache = {}
            self._rope3d_cache = {}

    def _rope_cache_2d(
        self, height: int, width: int, dim_half: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute and cache 2D RoPE tables for Y and X axes.

        Args:
            height: Number of rows H.
            width: Number of columns W.
            dim_half: Per-axis channel dimension. Must be even because rotations operate on pairs.
            device: Target device for the cached tensors.
            dtype: Target dtype for the cached tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ``(cos_y, sin_y, cos_x, sin_x)``
            where shapes are:
            - cos_y/sin_y: [dim_half, H]
            - cos_x/sin_x: [dim_half, W]

        Notes:
            The cache key is ``(H, W, D_axis, device, dtype)``. Tables are built under ``torch.no_grad()`` and reused across calls.
        """
        key = (int(height), int(width), int(dim_half), str(device), str(dtype))
        if key in self._rope2d_cache:
            return self._rope2d_cache[key]
        # If not in cache, compute and cache
        with torch.no_grad():
            cos_y, sin_y, cos_x, sin_x = rope.construct_rope_2d_cache_bhl(
                height, width, dim_half, device, dtype, self.rope_base
            )
            self._rope2d_cache[key] = (cos_y, sin_y, cos_x, sin_x)
        return cos_y, sin_y, cos_x, sin_x

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Hyena-style global convolutional mixer.

        Args:
            query: torch.Tensor - The query tensor of shape (batch_size, * spatial_dims, hidden_dim).
            key: torch.Tensor - The key tensor of shape (batch_size, * spatial_dims, hidden_dim).
            value: torch.Tensor - The value tensor of shape (batch_size, * spatial_dims, hidden_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, * spatial_dims, hidden_dim).
        """
        # Reshape query, key, and value to [B, C, * spatial_dims] (Required for short convolutional projections).
        query = rearrange(query, "b ... c -> b c ...")
        key = rearrange(key, "b ... c -> b c ...")
        value = rearrange(value, "b ... c -> b c ...")

        # Apply short convolutional projection
        if not isinstance(self.short_conv, torch.nn.Identity):
            # Concatenate query, key, and value, apply the short conv projection and split again
            x = torch.cat([query, key, value], dim=1)
            x = self.short_conv(x)
            # Split query, key, and value
            query, key, value = x.split(query.shape[1], dim=1)
            # Avoid in-place ops on views returned by split
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # Apply QK normalization
        if self.apply_qk_norm:
            query, key = qk_norm.apply_qk_norm(query, key, dim=1)

        # Apply RoPE
        if self.use_rope:
            dimensionality_input = query.ndim - 2
            if dimensionality_input == 1:
                assert query.shape[1] % 2 == 0, (
                    f"With 1D RoPE, the number of channels must be divisible by 2. Got {query.shape[1]}."
                )
                assert key.shape[1] % 2 == 0, (
                    f"With 1D RoPE, the number of channels must be divisible by 2. Got {key.shape[1]}."
                )
                raise NotImplementedError("1D RoPE is not implemented yet.")
            elif dimensionality_input == 2:
                assert query.shape[1] % 4 == 0, (
                    f"With 2D RoPE, the number of channels must be divisible by 4. Got {query.shape[1]}."
                )
                assert key.shape[1] % 4 == 0, (
                    f"With 2D RoPE, the number of channels must be divisible by 4. Got {key.shape[1]}."
                )

                # Gather or contsruct the RoPE 2D cache
                _, hidden_dim, height, width = query.shape
                rope_2d_cache = self._rope_cache_2d(height, width, hidden_dim // 2, query.device, query.dtype)
                # Apply RoPE to query and key
                query = rope.apply_rope_2d_bhl(query, rope_2d_cache)
                key = rope.apply_rope_2d_bhl(key, rope_2d_cache)

            elif dimensionality_input == 3:
                assert query.shape[1] % 8 == 0, (
                    f"With 3D RoPE, the number of channels must be divisible by 8. Got {query.shape[1]}."
                )
                assert key.shape[1] % 8 == 0, (
                    f"With 3D RoPE, the number of channels must be divisible by 8. Got {key.shape[1]}."
                )
                raise NotImplementedError("3D RoPE is not implemented yet.")
            else:
                raise NotImplementedError(f"RoPE is not implemented for {dimensionality_input}D inputs.")

        # First gate
        # z = query * self.gate_nonlinear(key)
        query.mul_(self.gate_nonlinear(key))

        # Apply global convolution
        y = self.global_conv(query, is_bhl_input=True)

        # Second gate
        # y = self.gate_nonlinear(y) * value in-place
        value.mul_(self.gate_nonlinear(y))

        # Reshape back to [B, * spatial_dims, C]
        return rearrange(value, "b c ... -> b ... c")
