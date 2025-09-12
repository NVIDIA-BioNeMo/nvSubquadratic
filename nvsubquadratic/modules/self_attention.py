import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel

from nvsubquadratic.utils import qk_norm, rope


class SelfAttention(torch.nn.Module):
    """Multi-head self-attention with optional QK normalization and Rotary Positional Embeddings (RoPE).

    Input / Output
    - Accepts sequences ``[B, T, C]`` or images ``[B, H, W, C]``.
    - Returns the same leading shape with channel dimension ``C``.

    Dimensions
    - ``C`` must be divisible by ``num_heads``; per-head dim is ``D = C / num_heads``.

    RoPE support
    - 1D (sequences): applied over length ``T``; requires ``D`` even.
    - 2D (images): applied over ``(H, W)`` by splitting per-head dim into two halves
      (Y-axis, X-axis); requires ``D`` divisible by 4.
    - 3D (videos or other 3D data): applied over ``(H, W, D)`` by splitting per-head dim into three halves
      (Z-axis, X-axis, Y-axis); requires ``D`` divisible by 8.

    QK normalization
    - When enabled, queries and keys are L2-normalized per head along the last dimension.

    Implementation notes
    - Uses PyTorch scaled_dot_product_attention and prefers FlashAttention kernels when available.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        apply_qk_norm: bool,
        use_rope: bool,
        attn_dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        """Initialize the SelfAttention module.

        Args:
            hidden_dim: The dimension of the hidden states.
            num_heads: The number of attention heads.
            apply_qk_norm: Whether to apply QK normalization.
            use_rope: Whether to apply RoPE.
            attn_dropout: The dropout rate for the attention weights.
            rope_base: The base of the RoPE.

        Raises:
            AssertionError: If hidden_dim is not divisible by num_heads.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.apply_qk_norm = apply_qk_norm
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.attn_dropout = attn_dropout

        # RoPE caches (keyed by shape, dtype, device)
        if self.use_rope:
            self._rope1d_cache = {}
            self._rope2d_cache = {}
            self._rope3d_cache = {}

    def _flatten_spatial(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flattens the spatial dimensions of the input and returns the original spatial shape.

        Args:
            x: torch.Tensor - The input tensor of shape [batch_size, *spatial_dims, hidden_dim].

        Returns:
            tuple[torch.Tensor, tuple[int, ...]]: The flattened tensor and the original spatial shape.
        """
        assert x.ndim in [3, 4, 5], (
            "Input must be tensors with shape (batch_size, *spatial_dims, hidden_dim), where len(spatial_dims) can be 1, 2, or 3."
        )
        spatial_shape = x.shape[1:-1]
        x = rearrange(x, "b ... c -> b (...) c")
        return x, spatial_shape

    def _unflatten_spatial(self, x: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        """Invert _flatten_spatial, based on the provided spatial shape.

        Args:
            x: torch.Tensor - The flattened tensor of shape [batch_size, *spatial_dims, hidden_dim].
            spatial_shape: tuple[int, ...] - The original spatial shape.

        Returns:
            torch.Tensor - The unflattened tensor of shape [batch_size, *spatial_dims, hidden_dim].
        """
        assert len(spatial_shape) in [1, 2, 3], "Spatial shape must be a tuple of length 1, 2, or 3."
        if len(spatial_shape) == 1:
            return x
        elif len(spatial_shape) == 2:
            return rearrange(x, "b (h w) c -> b h w c", h=spatial_shape[0], w=spatial_shape[1])
        else:
            return rearrange(x, "b (h w d) c -> b h w d c", h=spatial_shape[0], w=spatial_shape[1], d=spatial_shape[2])

    def _rope_cache_1d(self, seq_len: int, dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache 1D RoPE cos/sin for input of length seq_len.

        Returns:
            cos: [seq_len, dim]
            sin: [seq_len, dim]
        """
        key = (int(seq_len), int(dim), str(device), str(dtype))
        if key in self._rope1d_cache:
            return self._rope1d_cache[key]
        # If not in cache, compute and cache
        with torch.no_grad():
            cos, sin = rope.construct_rope_1d_cache_blh(seq_len, dim, device, dtype, self.rope_base)
        self._rope1d_cache[key] = (cos, sin)
        return cos, sin

    def _rope_cache_2d(
        self, height: int, width: int, dim_half: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache 2D RoPE cos/sin for y and x axes.

        dim_half is the per-axis sub-dimension (D/2 per head). For 2D RoPE we
        split the head dim D into [D/2 (y-axis), D/2 (x-axis)], and each axis
        internally uses rotate_half, which itself requires that the per-axis
        dim is even. Therefore, overall D must be divisible by 4.

        Returns:
        - cos_y, sin_y: [H, dim_half]
        - cos_x, sin_x: [W, dim_half]
        """
        key = (int(height), int(width), int(dim_half), str(device), str(dtype))
        if key in self._rope2d_cache:
            return self._rope2d_cache[key]
        # If not in cache, compute and cache
        with torch.no_grad():
            cos_y, sin_y, cos_x, sin_x = rope.construct_rope_2d_cache_blh(
                height, width, dim_half, device, dtype, self.rope_base
            )
            self._rope2d_cache[key] = (cos_y, sin_y, cos_x, sin_x)
        return cos_y, sin_y, cos_x, sin_x

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention with optional QK-norm and RoPE.

        Args:
            query: [B, *spatial_dims, hidden_dim]
            key: [B, *spatial_dims, hidden_dim]
            value: [B, *spatial_dims, hidden_dim]

        Returns:
            out: [B, *spatial_dims, hidden_dim] (same as input)
        """
        # [B, * spatial_dims, hidden_dim] -> [B * num_heads, * spatial_dims, head_dim]
        query = rearrange(query, "b ... (h d) -> (b h) ... d", h=self.num_heads)
        key = rearrange(key, "b ... (h d) -> (b h) ... d", h=self.num_heads)
        value = rearrange(value, "b ... (h d) -> (b h) ... d", h=self.num_heads)

        # Optional QK normalization
        if self.apply_qk_norm:
            query, key = qk_norm.apply_qk_norm(query, key, dim=-1)

        # Optional RoPE positional encoding (before normalization)
        if self.use_rope:
            dimensionality_input = query.ndim - 2
            if dimensionality_input == 1:
                assert query.shape[1] % 2 == 0, (
                    f"With 1D RoPE, the number of channels must be divisible by 2. Got {query.shape[1]}."
                )
                assert key.shape[1] % 2 == 0, (
                    f"With 1D RoPE, the number of channels must be divisible by 2. Got {key.shape[1]}."
                )
                # Gather or contsruct the RoPE 1D cache
                _, seq_len, hidden_dim = query.shape
                rope_1d_cache = self._rope_cache_1d(seq_len, hidden_dim, query.device, query.dtype)
                # Apply RoPE to query and key
                query = rope.apply_rope_1d_blh(query, rope_1d_cache)
                key = rope.apply_rope_1d_blh(key, rope_1d_cache)

            elif dimensionality_input == 2:
                assert query.shape[1] % 4 == 0, (
                    f"With 2D RoPE, the number of channels must be divisible by 4. Got {query.shape[1]}."
                )
                assert key.shape[1] % 4 == 0, (
                    f"With 2D RoPE, the number of channels must be divisible by 4. Got {key.shape[1]}."
                )
                # Gather or contsruct the RoPE 2D cache
                _, height, width, hidden_dim = query.shape
                rope_2d_cache = self._rope_cache_2d(height, width, hidden_dim // 2, query.device, query.dtype)
                # Apply RoPE to query and key
                query = rope.apply_rope_2d_blh(query, rope_2d_cache)
                key = rope.apply_rope_2d_blh(key, rope_2d_cache)

            elif dimensionality_input == 3:
                assert query.shape[1] % 6 == 0, (
                    f"With 3D RoPE, the number of channels must be divisible by 6. Got {query.shape[1]}."
                )
                assert key.shape[1] % 6 == 0, (
                    f"With 3D RoPE, the number of channels must be divisible by 6. Got {key.shape[1]}."
                )
                raise NotImplementedError("3D RoPE is not implemented yet.")
            else:
                raise NotImplementedError(f"RoPE is not implemented for {dimensionality_input}D inputs.")

        # Flatten spatial dims if present
        query, spatial_shape = self._flatten_spatial(query)
        key, _ = self._flatten_spatial(key)
        value, _ = self._flatten_spatial(value)

        # Scaled dot-product attention (uses FlashAttention kernels when available)
        # SDPA applies 1/sqrt(d) scaling internally.
        in_dtype = query.dtype
        # Use bfloat16 to align with high-performance SDPA/FlashAttention kernels
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                # Requires unpacking the number of heads from the batch dimension
                rearrange(query.to(torch.bfloat16), "(b h) t d -> b h t d", h=self.num_heads),
                rearrange(key.to(torch.bfloat16), "(b h) t d -> b h t d", h=self.num_heads),
                rearrange(value.to(torch.bfloat16), "(b h) t d -> b h t d", h=self.num_heads),
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        # Convert back to original dtype [batch_size, num_heads, flatten(* spatial_dims), head_dim]
        out = out.to(in_dtype)

        # Merge heads: [batch_size, num_heads, flatten(* spatial_dims), head_dim] -> [batch_size, flatten(* spatial_dims), (num_heads * head_dim)]
        out = rearrange(out, "b h t d -> b t (h d)", h=self.num_heads)

        # Unflatten back to spatial dims
        out = self._unflatten_spatial(out, spatial_shape)
        return out
