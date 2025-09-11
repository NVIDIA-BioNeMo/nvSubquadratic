# David W. Romero, 2025-09-09

"""RoPE utilities."""

import torch
from einops import rearrange


############################################################
# BHL Functions
############################################################


def rotate_half_bhl(x: torch.Tensor) -> torch.Tensor:
    """Rotate the inpul along the hidden channel dimension as: [x1, x2] -> [-x2, x1].

    Args:
        x: torch.Tensor - The input tensor of shape (batch_size, hidden_dim, * spatial_dims).

    Returns:
        torch.Tensor - The output tensor of shape (batch_size, hidden_dim, * spatial_dims).
    """
    d = x.shape[1]
    x1 = x[:, : d // 2]
    x2 = x[:, d // 2 :]
    return torch.cat([-x2, x1], dim=1)


def construct_rope_2d_cache_bhl(
    height: int, width: int, dim_half: int, device: torch.device, dtype: torch.dtype, rope_base: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the 2D RoPE cache for a given height, width, and hidden dimension.

    Args:
        height: int - The height of the input tensor.
        width: int - The width of the input tensor.
        dim_half: int - The per-axis channel dimension.
        device: torch.device - The device to store the cache on.
        dtype: torch.dtype - The dtype of the cache.
        rope_base: float - The base of the RoPE.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The 2D RoPE cache for a given height, width, and hidden dimension.
            Organized as (cos_y, sin_y, cos_x, sin_x), with shapes: [dim_half, H], [dim_half, H], [dim_half, W], [dim_half, W].
    """
    theta = 1.0 / (rope_base ** (torch.arange(0, dim_half, 2, device=device, dtype=dtype) / dim_half))
    pos_y = torch.arange(height, device=device, dtype=dtype)
    pos_x = torch.arange(width, device=device, dtype=dtype)
    angles_y = pos_y[None, :] * theta[:, None]
    angles_x = pos_x[None, :] * theta[:, None]
    cos_y = torch.cos(angles_y).repeat_interleave(2, dim=0)  # [dim_half, H]
    sin_y = torch.sin(angles_y).repeat_interleave(2, dim=0)
    cos_x = torch.cos(angles_x).repeat_interleave(2, dim=0)  # [dim_half, W]
    sin_x = torch.sin(angles_x).repeat_interleave(2, dim=0)
    return cos_y, sin_y, cos_x, sin_x


def apply_rope_2d_bhl(
    x: torch.Tensor, rope_2d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Apply 2D RoPE to a tensor laid out as [B, C, H, W].

    The channel dimension C is split into two equal parts: ``C_y`` and ``C_x``.
    RoPE is applied independently along Y (to ``C_y``) and X (to ``C_x``). For
    pairwise rotations, ``C`` must be divisible by 4 so that each half is even.

    Args:
        x: Input tensor of shape ``[B, C, H, W]`` with ``C % 4 == 0``.
        rope_2d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] - The cache of 2D RoPE cos/sin for y and x axes, organized as (cos_y, sin_y, cos_x, sin_x).

    Returns:
        Tensor with the same shape as ``x``. Rotations are written back
        in-place via views to reduce allocations.

    Broadcasting:
        - cos_y/sin_y are reshaped to ``[1, C/2, H, 1]`` for the first half.
        - cos_x/sin_x are reshaped to ``[1, C/2, 1, W]`` for the second half.
    """
    _, hidden_dim, _, _ = x.shape
    hidden_dim_half = hidden_dim // 2

    # Split hidden dim: first half encodes Y, second half encodes X.
    cos_y, sin_y, cos_x, sin_x = rope_2d_cache
    cos_y = rearrange(cos_y, "d h -> 1 d h 1")
    sin_y = rearrange(sin_y, "d h -> 1 d h 1")
    cos_x = rearrange(cos_x, "d w -> 1 d 1 w")
    sin_x = rearrange(sin_x, "d w -> 1 d 1 w")

    x_y = x[:, :hidden_dim_half]
    x_x = x[:, hidden_dim_half:]

    # Apply RoPE to each axis with in-place fused ops to reduce allocations
    # x_y = x_y * cos_y + self._rotate_half(x_y) * sin_y
    rot_y = rotate_half_bhl(x_y)
    x_y.mul_(cos_y)
    x_y.addcmul_(rot_y, sin_y, value=1.0)

    # x_x = x_x * cos_x + self._rotate_half(x_x) * sin_x
    rot_x = rotate_half_bhl(x_x)
    x_x.mul_(cos_x)
    x_x.addcmul_(rot_x, sin_x, value=1.0)

    # Results are written back into x via views; no concat needed
    return x


def rotate_half_blh(x: torch.Tensor) -> torch.Tensor:
    """Rotate the input along the hidden channel dimension as: [x1, x2] -> [-x2, x1].

    Args:
        x: torch.Tensor - The input tensor of shape (batch_size, * spatial_dims, hidden_dim).

    Returns:
        torch.Tensor - The output tensor of shape (batch_size, * spatial_dims, hidden_dim).
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)
