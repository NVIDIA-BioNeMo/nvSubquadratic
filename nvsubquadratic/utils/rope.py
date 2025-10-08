# TODO: Add license header here


"""Rotary Position Embeddings (RoPE) utilities.

This module provides fast, allocation-friendly primitives to construct and apply
Rotary Position Embeddings for 1D, 2D, and 3D inputs in two common memory
layouts:

- BHL: ``[batch, hidden, * spatial_dims]``
- BLH: ``[batch, * spatial_dims, hidden]``

Both families expose:

- rotate_half_XXX (BHL or BLH): Pairwise channel rotation ``[x1, x2] -> [-x2, x1]`` along the
  hidden/channel dimension.
- construct_rope_{1d,2d,3d}_cache_XXX: Build per-axis cosine/sine caches.
- apply_rope_{1d,2d,3d}_XXX: Apply RoPE in-place via fused ops using the caches.

Channel partition and constraints
---------------------------------
- 1D: Channels are used as one block. Per-axis dim must be even.
  Overall hidden dim must be divisible by 2.
- 2D: Channels are split in two equal parts: first half for Y, second half for X.
  Each half must be even. Overall hidden dim must be divisible by 4.
- 3D: Channels are split in three equal parts: first third for Z, second third for Y,
  final third for X. Each third must be even. Overall hidden dim must be divisible by 6.

Cache organization and shapes
-----------------------------
- BHL caches (channel-first per-axis):
  - 1D: ``(cos, sin)`` with shapes ``[dim, T]``
  - 2D: ``(cos_y, sin_y, cos_x, sin_x)`` with shapes ``[dim_half, H]``, ``[dim_half, W]``
  - 3D: ``(cos_z, sin_z, cos_y, sin_y, cos_x, sin_x)`` with shapes
    ``[dim_third, D]``, ``[dim_third, H]``, ``[dim_third, W]``
- BLH caches (time/space-first per-axis):
  - 1D: ``(cos, sin)`` with shapes ``[T, dim]``
  - 2D: ``(cos_y, sin_y, cos_x, sin_x)`` with shapes ``[H, dim_half]``, ``[W, dim_half]``
  - 3D: ``(cos_z, sin_z, cos_y, sin_y, cos_x, sin_x)`` with shapes
    ``[D, dim_third]``, ``[H, dim_third]``, ``[W, dim_third]``

Broadcasting (apply functions)
------------------------------
Each ``apply_rope_*`` reshapes caches for broadcast along the appropriate spatial axis
and channel slice. See the function docstrings for the exact broadcast shapes used.

Performance and semantics
-------------------------
- ``apply_rope_*`` perform in-place fused operations (``mul_`` and ``addcmul_``) on
  views of the input to minimize allocations. A temporary rotated view is created via
  ``rotate_half_*``.
- Caches are constructed with explicit ``device`` and ``dtype``; build them once and
  reuse across forward calls when shape-compatible.
- The angular frequency schedule uses ``theta = 1 / (rope_base ** (arange(0, d/2) / (d/2)))``
  expanded by interleaving to match the full per-axis channel size.

Layouts summary
---------------
- BHL inputs:
  - 1D: ``x.shape == [B, C, T]`` -> use ``construct_rope_1d_cache_bhl`` and ``apply_rope_1d_bhl``
  - 2D: ``x.shape == [B, C, H, W]`` -> use ``construct_rope_2d_cache_bhl`` and ``apply_rope_2d_bhl``
  - 3D: ``x.shape == [B, C, D, H, W]`` -> use ``construct_rope_3d_cache_bhl`` and ``apply_rope_3d_bhl``
- BLH inputs:
  - 1D: ``x.shape == [B, T, C]`` -> use ``construct_rope_1d_cache_blh`` and ``apply_rope_1d_blh``
  - 2D: ``x.shape == [B, H, W, C]`` -> use ``construct_rope_2d_cache_blh`` and ``apply_rope_2d_blh``
  - 3D: ``x.shape == [B, D, H, W, C]`` -> use ``construct_rope_3d_cache_blh`` and ``apply_rope_3d_blh``

Notes:
-----
- "per-axis dim" refers to the channel slice assigned to each spatial axis before
  pairwise rotation; it must be even for the rotation to be well-defined.
- Ensure hidden dimension divisibility as stated above before calling ``apply_rope_*``.
"""

__all__ = [
    "apply_rope_1d_bhl",
    "apply_rope_1d_blh",
    "apply_rope_2d_bhl",
    "apply_rope_2d_blh",
    "apply_rope_3d_bhl",
    "apply_rope_3d_blh",
    "construct_rope_1d_cache_bhl",
    "construct_rope_1d_cache_blh",
    "construct_rope_2d_cache_bhl",
    "construct_rope_2d_cache_blh",
    "construct_rope_3d_cache_bhl",
    "construct_rope_3d_cache_blh",
    "rotate_half_bhl",
    "rotate_half_blh",
]

import torch
from einops import rearrange


############################################################
# BHL Functions
############################################################
def rotate_half_bhl(x: torch.Tensor) -> torch.Tensor:
    """Rotate the input along the hidden channel dimension as: [x1, x2] -> [-x2, x1].

    Args:
        x: torch.Tensor - The input tensor of shape (batch_size, hidden_dim, * spatial_dims).

    Returns:
        torch.Tensor - The output tensor of shape (batch_size, hidden_dim, * spatial_dims).
    """
    d = x.shape[1]
    x1 = x[:, : d // 2]
    x2 = x[:, d // 2 :]
    return torch.cat([-x2, x1], dim=1)


def construct_rope_1d_cache_bhl(
    seq_len: int, dim: int, device: torch.device, dtype: torch.dtype, rope_base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the 1D RoPE cache for a given sequence length and hidden dimension.

    Args:
        seq_len: int - The length of the input sequence.
        dim: int - The hidden dimension.
        device: torch.device - The device to store the cache on.
        dtype: torch.dtype - The dtype of the cache.
        rope_base: float - The base of the RoPE.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            The 1D RoPE cache organized as:
            (cos, sin)
            with shapes:
            - cos: [dim, seq_len]
            - sin: [dim, seq_len]

    Notes:
        - For pairwise rotations, each per-axis channel size (dim) must be even.
        - Overall per-head dim must be divisible by 2 (since D = 2 * dim, and dim must be even).
    """
    # Frequencies for each axis share the same theta definition per-axis
    theta = 1.0 / (rope_base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    # Position
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    # Angles
    angles = pos[None, :] * theta[:, None]
    cos = torch.cos(angles).repeat_interleave(2, dim=0)  # [dim, seq_len]
    sin = torch.sin(angles).repeat_interleave(2, dim=0)  # [dim, seq_len]
    return cos, sin


def construct_rope_2d_cache_bhl(
    height: int, width: int, dim_half: int, device: torch.device, dtype: torch.dtype, rope_base: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the 2D RoPE cache for a given (height, width) and per-axis dimension.

    Args:
        height: int - The height of the input tensor.
        width: int - The width of the input tensor.
        dim_half: int - The per-axis channel dimension.
        device: torch.device - The device to store the cache on.
        dtype: torch.dtype - The dtype of the cache.
        rope_base: float - The base of the RoPE.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The 2D RoPE cache organized as:
            (cos_y, sin_y, cos_x, sin_x)
            with shapes:
            - cos_y, sin_y: [dim_half, height]
            - cos_x, sin_x: [dim_half, width]

    Notes:
        - For pairwise rotations, each per-axis channel size (dim_half) must be even.
        - Overall per-head dim must be divisible by 4 (since D = 2 * dim_half, and dim_half must be even).
    """
    # Frequencies for each axis share the same theta definition per-axis
    theta = 1.0 / (rope_base ** (torch.arange(0, dim_half, 2, device=device, dtype=dtype) / dim_half))
    # Y (height)
    pos_y = torch.arange(height, device=device, dtype=dtype)
    angles_y = pos_y[None, :] * theta[:, None]
    cos_y = torch.cos(angles_y).repeat_interleave(2, dim=0)  # [dim_half, H]
    sin_y = torch.sin(angles_y).repeat_interleave(2, dim=0)  # [dim_half, H]
    # X (width)
    pos_x = torch.arange(width, device=device, dtype=dtype)
    angles_x = pos_x[None, :] * theta[:, None]
    cos_x = torch.cos(angles_x).repeat_interleave(2, dim=0)  # [dim_half, W]
    sin_x = torch.sin(angles_x).repeat_interleave(2, dim=0)  # [dim_half, W]
    return cos_y, sin_y, cos_x, sin_x


def construct_rope_3d_cache_bhl(
    depth: int,
    height: int,
    width: int,
    dim_third: int,
    device: torch.device,
    dtype: torch.dtype,
    rope_base: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the 3D RoPE cache for given (depth, height, width) and per-axis dimension.

    Args:
        depth: int - The depth of the input tensor (Z axis).
        height: int - The height of the input tensor (Y axis).
        width: int - The width of the input tensor (X axis).
        dim_third: int - The per-axis channel dimension (one third of per-head dim).
        device: torch.device - The device to store the cache on.
        dtype: torch.dtype - The dtype of the cache.
        rope_base: float - The base of the RoPE.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The 3D RoPE cache organized as:
            (cos_z, sin_z, cos_y, sin_y, cos_x, sin_x)
            with shapes:
            - cos_z, sin_z: [dim_third, depth]
            - cos_y, sin_y: [dim_third, height]
            - cos_x, sin_x: [dim_third, width]

    Notes:
        - For pairwise rotations, each per-axis channel size (dim_third) must be even.
        - Overall per-head dim must be divisible by 6 (since D = 3 * dim_third, and dim_third must be even).
    """
    # Frequencies for each axis share the same theta definition per-axis
    theta = 1.0 / (rope_base ** (torch.arange(0, dim_third, 2, device=device, dtype=dtype) / dim_third))
    # Z (depth)
    pos_z = torch.arange(depth, device=device, dtype=dtype)
    angles_z = pos_z[None, :] * theta[:, None]
    cos_z = torch.cos(angles_z).repeat_interleave(2, dim=0)  # [dim_third, depth]
    sin_z = torch.sin(angles_z).repeat_interleave(2, dim=0)  # [dim_third, depth]
    # Y (height)
    pos_y = torch.arange(height, device=device, dtype=dtype)
    angles_y = pos_y[None, :] * theta[:, None]
    cos_y = torch.cos(angles_y).repeat_interleave(2, dim=0)  # [dim_third, height]
    sin_y = torch.sin(angles_y).repeat_interleave(2, dim=0)  # [dim_third, height]
    # X (width)
    pos_x = torch.arange(width, device=device, dtype=dtype)
    angles_x = pos_x[None, :] * theta[:, None]
    cos_x = torch.cos(angles_x).repeat_interleave(2, dim=0)  # [dim_third, width]
    sin_x = torch.sin(angles_x).repeat_interleave(2, dim=0)  # [dim_third, width]
    return cos_z, sin_z, cos_y, sin_y, cos_x, sin_x


def apply_rope_1d_bhl(x: torch.Tensor, rope_1d_cache: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply 1D RoPE to a tensor laid out as [batch_size, hidden_dim, seq_len].

    Args:
        x: Input tensor of shape ``[batch_size, hidden_dim, seq_len]``.
        rope_1d_cache: tuple[torch.Tensor, torch.Tensor] - The cache of 1D RoPE cos/sin for the input sequence.

    Returns:
        Tensor with the same shape as ``x``.

    Broadcasting:
        - cos/sin are reshaped to ``[1, hidden_dim, seq_len]``.
    """
    cos, sin = rope_1d_cache
    cos = rearrange(cos, "d t -> 1 d t")
    sin = rearrange(sin, "d t -> 1 d t")

    out_x = x

    # Apply RoPE in-place
    # x = x * cos + rotate_half(x) * sin
    rot_x = rotate_half_bhl(x)
    out_x.mul_(cos)
    out_x.addcmul_(rot_x, sin, value=1.0)
    return out_x


def apply_rope_2d_bhl(
    x: torch.Tensor, rope_2d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Apply 2D RoPE to a tensor laid out as [batch_size, hidden_dim, H, W].

    The channel dimension C is split into two equal parts: ``C_y`` and ``C_x``.
    RoPE is applied independently along Y (to ``C_y``) and X (to ``C_x``). For
    pairwise rotations, ``C`` must be divisible by 4 so that each half is even.

    Args:
        x: Input tensor of shape ``[batch_size, hidden_dim, H, W]`` with ``hidden_dim % 4 == 0``.
        rope_2d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] - The cache of 2D RoPE cos/sin for y and x axes, organized as (cos_y, sin_y, cos_x, sin_x).

    Returns:
        Tensor with the same shape as ``x``. Rotations are written back
        in-place via views to reduce allocations.

    Broadcasting:
        - cos_y/sin_y are reshaped to ``[1, hidden_dim/2, H, 1]`` for the first half.
        - cos_x/sin_x are reshaped to ``[1, hidden_dim/2, 1, W]`` for the second half.
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

    # Results are written back into x
    return torch.cat([x_y, x_x], dim=1)


def apply_rope_3d_bhl(
    x: torch.Tensor,
    rope_3d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply 3D RoPE to a tensor laid out as [batch_size, hidden_dim, D, H, W].

    The channel dimension C is split into three equal parts: ``C_z``, ``C_y``, and ``C_x``.
    RoPE is applied independently along Z (to ``C_z``), Y (to ``C_y``), and X (to ``C_x``).
    For pairwise rotations, ``C`` must be divisible by 6 so that each third is even.

    Args:
        x: Input tensor of shape ``[batch_size, hidden_dim, D, H, W]`` with ``hidden_dim % 6 == 0``.
        rope_3d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The cache of 3D RoPE cos/sin for z, y, and x axes, organized as
            (cos_z, sin_z, cos_y, sin_y, cos_x, sin_x).

    Returns:
        Tensor with the same shape as ``x``. Rotations are written back
        in-place via views to reduce allocations.

    Broadcasting:
        - cos_z/sin_z are reshaped to ``[1, hidden_dim/3, D, 1, 1]`` for the first third.
        - cos_y/sin_y are reshaped to ``[1, hidden_dim/3, 1, H, 1]`` for the second third.
        - cos_x/sin_x are reshaped to ``[1, hidden_dim/3, 1, 1, W]`` for the final third.
    """
    _, hidden_dim, _, _, _ = x.shape
    hidden_dim_third = hidden_dim // 3

    # Split hidden dim: first third encodes Z, second Y, third X.
    cos_z, sin_z, cos_y, sin_y, cos_x, sin_x = rope_3d_cache
    cos_z = rearrange(cos_z, "d z -> 1 d z 1 1")
    sin_z = rearrange(sin_z, "d z -> 1 d z 1 1")
    cos_y = rearrange(cos_y, "d h -> 1 d 1 h 1")
    sin_y = rearrange(sin_y, "d h -> 1 d 1 h 1")
    cos_x = rearrange(cos_x, "d w -> 1 d 1 1 w")
    sin_x = rearrange(sin_x, "d w -> 1 d 1 1 w")

    x_z = x[:, :hidden_dim_third]
    x_y = x[:, hidden_dim_third : 2 * hidden_dim_third]
    x_x = x[:, 2 * hidden_dim_third :]

    # Apply RoPE per-axis with in-place fused ops
    # x_z = x_z * cos_z + rotate_half(x_z) * sin_z
    rot_z = rotate_half_bhl(x_z)
    x_z.mul_(cos_z)
    x_z.addcmul_(rot_z, sin_z, value=1.0)

    # x_y = x_y * cos_y + rotate_half(x_y) * sin_y
    rot_y = rotate_half_bhl(x_y)
    x_y.mul_(cos_y)
    x_y.addcmul_(rot_y, sin_y, value=1.0)

    # x_x = x_x * cos_x + rotate_half(x_x) * sin_x
    rot_x = rotate_half_bhl(x_x)
    x_x.mul_(cos_x)
    x_x.addcmul_(rot_x, sin_x, value=1.0)

    # Results are written back into x
    return torch.cat([x_z, x_y, x_x], dim=1)


############################################################
# BLH Functions
############################################################
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


def construct_rope_1d_cache_blh(
    seq_len: int, dim: int, device: torch.device, dtype: torch.dtype, rope_base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the 1D RoPE cache for a given sequence length and hidden dimension.

    Args:
        seq_len: int - The length of the input sequence.
        dim: int - The hidden dimension.
        device: torch.device - The device to store the cache on.
        dtype: torch.dtype - The dtype of the cache.
        rope_base: float - The base of the RoPE.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            The 1D RoPE cache organized as:
            (cos, sin)
            with shapes:
            - cos: [seq_len, dim]
            - sin: [seq_len, dim]

    Notes:
        - For pairwise rotations, each per-axis channel size (dim) must be even.
        - Overall per-head dim must be divisible by 2 (since D = 2 * dim, and dim must be even).
    """
    # Frequencies for each axis share the same theta definition per-axis
    theta = 1.0 / (rope_base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    # Position
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    # Angles
    angles = pos[:, None] * theta[None, :]
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [seq_len, dim]
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # [seq_len, dim]
    return cos, sin


def construct_rope_2d_cache_blh(
    height: int, width: int, dim_half: int, device: torch.device, dtype: torch.dtype, rope_base: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the 2D RoPE cache for a given (height, width) and per-axis dimension.

    Args:
        height: int - The height of the input tensor.
        width: int - The width of the input tensor.
        dim_half: int - The per-axis channel dimension.
        device: torch.device - The device to store the cache on.
        dtype: torch.dtype - The dtype of the cache.
        rope_base: float - The base of the RoPE.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The 2D RoPE cache organized as:
            (cos_y, sin_y, cos_x, sin_x)
            with shapes:
            - cos_y, sin_y: [height, dim_half]
            - cos_x, sin_x: [width, dim_half]

    Notes:
        - For pairwise rotations, each per-axis channel size (dim_half) must be even.
        - Overall per-head dim must be divisible by 4 (since D = 2 * dim_half, and dim_half must be even).
    """
    # Frequencies for each axis share the same theta definition per-axis
    theta = 1.0 / (rope_base ** (torch.arange(0, dim_half, 2, device=device, dtype=dtype) / dim_half))
    # Y (height)
    pos_y = torch.arange(height, device=device, dtype=dtype)
    angles_y = pos_y[:, None] * theta[None, :]
    cos_y = torch.cos(angles_y).repeat_interleave(2, dim=-1)  # [H, dim_half]
    sin_y = torch.sin(angles_y).repeat_interleave(2, dim=-1)  # [H, dim_half]
    # X (width)
    pos_x = torch.arange(width, device=device, dtype=dtype)
    angles_x = pos_x[:, None] * theta[None, :]
    cos_x = torch.cos(angles_x).repeat_interleave(2, dim=-1)  # [W, dim_half]
    sin_x = torch.sin(angles_x).repeat_interleave(2, dim=-1)  # [W, dim_half]
    return cos_y, sin_y, cos_x, sin_x


def construct_rope_3d_cache_blh(
    depth: int,
    height: int,
    width: int,
    dim_third: int,
    device: torch.device,
    dtype: torch.dtype,
    rope_base: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the 3D RoPE cache for given (depth, height, width) and per-axis dimension.

    Args:
        depth: int - The depth of the input tensor (Z axis).
        height: int - The height of the input tensor (Y axis).
        width: int - The width of the input tensor (X axis).
        dim_third: int - The per-axis channel dimension (one third of per-head dim).
        device: torch.device - The device to store the cache on.
        dtype: torch.dtype - The dtype of the cache.
        rope_base: float - The base of the RoPE.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The 3D RoPE cache organized as:
            (cos_z, sin_z, cos_y, sin_y, cos_x, sin_x)
            with shapes:
            - cos_z, sin_z: [depth, dim_third]
            - cos_y, sin_y: [height, dim_third]
            - cos_x, sin_x: [width, dim_third]

    Notes:
        - For pairwise rotations, each per-axis channel size (dim_third) must be even.
        - Overall per-head dim must be divisible by 6 (since D = 3 * dim_third, and dim_third must be even).
    """
    # Frequencies for each axis share the same theta definition per-axis
    theta = 1.0 / (rope_base ** (torch.arange(0, dim_third, 2, device=device, dtype=dtype) / dim_third))
    # Z (depth)
    pos_z = torch.arange(depth, device=device, dtype=dtype)
    angles_z = pos_z[:, None] * theta[None, :]
    cos_z = torch.cos(angles_z).repeat_interleave(2, dim=-1)  # [depth, dim_third]
    sin_z = torch.sin(angles_z).repeat_interleave(2, dim=-1)  # [depth, dim_third]
    # Y (height)
    pos_y = torch.arange(height, device=device, dtype=dtype)
    angles_y = pos_y[:, None] * theta[None, :]
    cos_y = torch.cos(angles_y).repeat_interleave(2, dim=-1)  # [height, dim_third]
    sin_y = torch.sin(angles_y).repeat_interleave(2, dim=-1)  # [height, dim_third]
    # X (width)
    pos_x = torch.arange(width, device=device, dtype=dtype)
    angles_x = pos_x[:, None] * theta[None, :]
    cos_x = torch.cos(angles_x).repeat_interleave(2, dim=-1)  # [width, dim_third]
    sin_x = torch.sin(angles_x).repeat_interleave(2, dim=-1)  # [width, dim_third]
    return cos_z, sin_z, cos_y, sin_y, cos_x, sin_x


def apply_rope_1d_blh(x: torch.Tensor, rope_1d_cache: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply 1D RoPE to a tensor laid out as [batch_size, seq_len, hidden_dim].

    Args:
        x: Input tensor of shape ``[batch_size, seq_len, hidden_dim]``.
        rope_1d_cache: tuple[torch.Tensor, torch.Tensor] - The cache of 1D RoPE cos/sin for the input sequence.

    Returns:
        Tensor with the same shape as ``x``.

    Broadcasting:
        - cos/sin are reshaped to ``[1, seq_len, hidden_dim]``.
    """
    cos, sin = rope_1d_cache
    cos = rearrange(cos, "t d -> 1 t d")
    sin = rearrange(sin, "t d -> 1 t d")

    out_x = x

    # Apply RoPE in-place
    # x = x * cos + rotate_half(x) * sin
    rot_x = rotate_half_blh(x)
    out_x.mul_(cos)
    out_x.addcmul_(rot_x, sin, value=1.0)
    return out_x


def apply_rope_2d_blh(
    x: torch.Tensor, rope_2d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Apply 2D RoPE to a tensor laid out as [batch_size, H, W, hidden_dim].

    The channel dimension C is split into two equal parts: ``C_y`` and ``C_x``.
    RoPE is applied independently along Y (to ``C_y``) and X (to ``C_x``). For
    pairwise rotations, ``C`` must be divisible by 4 so that each half is even.

    Args:
        x: Input tensor of shape ``[batch_size, H, W, hidden_dim]`` with ``hidden_dim % 4 == 0``.
        rope_2d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] - The cache of 2D RoPE cos/sin for y and x axes, organized as (cos_y, sin_y, cos_x, sin_x).

    Returns:
        Tensor with the same shape as ``x``.

    Broadcasting:
        - cos_y/sin_y are reshaped to ``[1, hidden_dim/2, H, 1]`` for the first half.
        - cos_x/sin_x are reshaped to ``[1, hidden_dim/2, 1, W]`` for the second half.
    """
    _, _, _, hidden_dim = x.shape
    hidden_dim_half = hidden_dim // 2

    # Split hidden dim: first half encodes Y, second half encodes X.
    cos_y, sin_y, cos_x, sin_x = rope_2d_cache
    cos_y = rearrange(cos_y, "h d -> 1 h 1 d")
    sin_y = rearrange(sin_y, "h d -> 1 h 1 d")
    cos_x = rearrange(cos_x, "w d -> 1 1 w d")
    sin_x = rearrange(sin_x, "w d -> 1 1 w d")

    x_y = x[..., :hidden_dim_half]
    x_x = x[..., hidden_dim_half:]

    # Apply RoPE to each axis with in-place fused ops to reduce allocations
    # x_y = x_y * cos_y + self._rotate_half(x_y) * sin_y
    rot_y = rotate_half_blh(x_y)
    x_y.mul_(cos_y)
    x_y.addcmul_(rot_y, sin_y, value=1.0)

    # x_x = x_x * cos_x + self._rotate_half(x_x) * sin_x
    rot_x = rotate_half_blh(x_x)
    x_x.mul_(cos_x)
    x_x.addcmul_(rot_x, sin_x, value=1.0)

    # Results are written back into x
    return torch.cat([x_y, x_x], dim=-1)


def apply_rope_3d_blh(
    x: torch.Tensor,
    rope_3d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply 3D RoPE to a tensor laid out as [batch_size, D, H, W, hidden_dim].

    The channel dimension C is split into three equal parts: ``C_z``, ``C_y``, and ``C_x``.
    RoPE is applied independently along Z (to ``C_z``), Y (to ``C_y``), and X (to ``C_x``).
    For pairwise rotations, ``C`` must be divisible by 6 so that each third is even.

    Args:
        x: Input tensor of shape ``[batch_size, D, H, W, hidden_dim]`` with ``hidden_dim % 6 == 0``.
        rope_3d_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The cache of 3D RoPE cos/sin for z, y, and x axes, organized as
            (cos_z, sin_z, cos_y, sin_y, cos_x, sin_x).

    Returns:
        Tensor with the same shape as ``x``. Rotations are written back
        in-place via views to reduce allocations.

    Broadcasting:
        - cos_z/sin_z are reshaped to ``[1, D, 1, 1, hidden_dim/3]`` for the first third.
        - cos_y/sin_y are reshaped to ``[1, 1, H, 1, hidden_dim/3]`` for the second third.
        - cos_x/sin_x are reshaped to ``[1, 1, 1, W, hidden_dim/3]`` for the final third.
    """
    _, _, _, _, hidden_dim = x.shape
    hidden_dim_third = hidden_dim // 3

    # Split hidden dim: first third encodes Z, second Y, third X.
    cos_z, sin_z, cos_y, sin_y, cos_x, sin_x = rope_3d_cache
    cos_z = rearrange(cos_z, "z d -> 1 z 1 1 d")
    sin_z = rearrange(sin_z, "z d -> 1 z 1 1 d")
    cos_y = rearrange(cos_y, "h d -> 1 1 h 1 d")
    sin_y = rearrange(sin_y, "h d -> 1 1 h 1 d")
    cos_x = rearrange(cos_x, "w d -> 1 1 1 w d")
    sin_x = rearrange(sin_x, "w d -> 1 1 1 w d")

    x_z = x[..., :hidden_dim_third]
    x_y = x[..., hidden_dim_third : 2 * hidden_dim_third]
    x_x = x[..., 2 * hidden_dim_third :]

    # Apply RoPE per-axis with in-place fused ops
    # x_z = x_z * cos_z + rotate_half(x_z) * sin_z
    rot_z = rotate_half_blh(x_z)
    x_z.mul_(cos_z)
    x_z.addcmul_(rot_z, sin_z, value=1.0)

    # x_y = x_y * cos_y + rotate_half(x_y) * sin_y
    rot_y = rotate_half_blh(x_y)
    x_y.mul_(cos_y)
    x_y.addcmul_(rot_y, sin_y, value=1.0)

    # x_x = x_x * cos_x + rotate_half(x_x) * sin_x
    rot_x = rotate_half_blh(x_x)
    x_x.mul_(cos_x)
    x_x.addcmul_(rot_x, sin_x, value=1.0)

    # Results are written back into x
    return torch.cat([x_z, x_y, x_x], dim=-1)
