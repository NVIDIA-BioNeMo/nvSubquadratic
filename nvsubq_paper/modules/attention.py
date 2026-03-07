# TODO: Add license header here


"""Multi-head self-attention with optional QK normalization and Rotary Positional Embeddings (RoPE)."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel

from nvsubq_paper.parallel.utils import zigzag_gather_from_group_ranks, zigzag_split_across_group_ranks
from nvsubq_paper.utils import qk_norm, rope


class Attention(torch.nn.Module):
    """Multi-head attention (support for self and cross-attention) with optional QK normalization and Rotary Positional Embeddings (RoPE).

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

    Context Parallelism Limitations
    - **WARNING**: This implementation is for **illustration and compatibility only**.
    - It uses zigzag all-gather/split for CP, which causes significant memory issues at long context
      lengths because the attention layer does not implement internal ring attention.
    - **For production use with long contexts**, use PyTorch's standard context-parallel attention
      blocks (e.g., as in torchtitan: https://docs.pytorch.org/tutorials/unstable/context_parallel.html).
    - Future work: Migrate to PyTorch's standard CP attention API, which may also eliminate
      the requirement for zigzag-style communication patterns.

    Args:
        hidden_dim (int): The dimension of the hidden states.
        num_heads (int): The number of attention heads.
        apply_qk_norm (bool): Whether to apply QK normalization.
        use_rope (bool): Whether to apply RoPE.
        is_causal (bool): Whether the attention is causal. Defaults to False.
        attn_dropout (float): The dropout rate for the attention weights.
        rope_base (float): The base of the RoPE.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        apply_qk_norm: bool,
        use_rope: bool,
        is_causal: bool = False,
        attn_dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        """Initialize the SelfAttention module."""
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.apply_qk_norm = apply_qk_norm
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.is_causal = is_causal
        self.attn_dropout = attn_dropout

        # RoPE caches (keyed by shape, dtype, device)
        if self.use_rope:
            self._rope1d_cache = {}
            self._rope2d_cache = {}
            self._rope3d_cache = {}

    def extra_repr(self) -> str:
        """Extra repr for the Attention module."""
        return f"num_heads={self.num_heads}, apply_qk_norm={self.apply_qk_norm}, is_causal={self.is_causal}, attn_dropout={self.attn_dropout}, use_rope={self.use_rope}, rope_base={self.rope_base}"

    def _rope_cache_1d(self, seq_len: int, dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute and cache 1D RoPE cos/sin for input of length seq_len.

        Args:
            seq_len: Number of positions T.
            dim: Per-axis channel dimension. Must be even because rotations operate on pairs.
            device: Target device for the cached tensors.
            dtype: Target dtype for the cached tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)``
            where shapes are:
            - cos: [seq_len, dim]
            - sin: [seq_len, dim]

        Notes:
            The cache key is ``(T, D_axis, device, dtype)``. Tables are built under ``torch.no_grad()`` and reused across calls.
        """
        key = (int(seq_len), int(dim), str(device), str(dtype))
        if key in self._rope1d_cache:
            return self._rope1d_cache[key]
        # If not in cache, compute and cache
        with torch.inference_mode(False):
            with torch.no_grad():
                cos, sin = rope.construct_rope_1d_cache_blh(seq_len, dim, device, dtype, self.rope_base)
                self._rope1d_cache[key] = (cos, sin)
        return cos, sin

    def _rope_cache_2d(
        self, height: int, width: int, dim_half: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute and cache 2D RoPE cos/sin for Y and X axes.

        Args:
            height: Number of rows H.
            width: Number of columns W.
            dim_half: Per-axis channel dimension. Must be even because rotations operate on pairs.
            device: Target device for the cached tensors.
            dtype: Target dtype for the cached tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ``(cos_y, sin_y, cos_x, sin_x)``
            where shapes are:
            - cos_y, sin_y: [height, dim_half]
            - cos_x, sin_x: [width, dim_half]

        Notes:
            The cache key is ``(H, W, D_axis, device, dtype)``. Tables are built under ``torch.no_grad()`` and reused across calls.
        """
        key = (int(height), int(width), int(dim_half), str(device), str(dtype))
        if key in self._rope2d_cache:
            return self._rope2d_cache[key]
        # If not in cache, compute and cache
        with torch.inference_mode(False):
            with torch.no_grad():
                cos_y, sin_y, cos_x, sin_x = rope.construct_rope_2d_cache_blh(
                    height, width, dim_half, device, dtype, self.rope_base
                )
                self._rope2d_cache[key] = (cos_y, sin_y, cos_x, sin_x)
        return cos_y, sin_y, cos_x, sin_x

    def _rope_cache_3d(
        self, depth: int, height: int, width: int, dim_third: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute and cache 3D RoPE cos/sin for Z, Y, and X axes.

        Args:
            depth: Number of depth D.
            height: Number of rows H.
            width: Number of columns W.
            dim_third: Per-axis channel dimension. Must be even because rotations operate on pairs.
            device: Target device for the cached tensors.
            dtype: Target dtype for the cached tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ``(cos_z, sin_z, cos_y, sin_y, cos_x, sin_x)``
            where shapes are:
            - cos_z, sin_z: [depth, dim_third]
            - cos_y, sin_y: [height, dim_third]
            - cos_x, sin_x: [width, dim_third]

        Notes:
            The cache key is ``(D, H, W, D_axis, device, dtype)``. Tables are built under ``torch.no_grad()`` and reused across calls.
        """
        key = (int(depth), int(height), int(width), int(dim_third), str(device), str(dtype))
        if key in self._rope3d_cache:
            return self._rope3d_cache[key]
        # If not in cache, compute and cache
        with torch.inference_mode(False):
            with torch.no_grad():
                cos_z, sin_z, cos_y, sin_y, cos_x, sin_x = rope.construct_rope_3d_cache_blh(
                    depth, height, width, dim_third, device, dtype, self.rope_base
                )
                self._rope3d_cache[key] = (cos_z, sin_z, cos_y, sin_y, cos_x, sin_x)
        return cos_z, sin_z, cos_y, sin_y, cos_x, sin_x

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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cp_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """Apply multi-head self-attention with optional QK-norm and RoPE.

        Args:
            query: [B, *spatial_dims, hidden_dim]
            key: [B, *spatial_dims, hidden_dim]
            value: [B, *spatial_dims, hidden_dim]
            cp_group: torch.distributed.ProcessGroup - Context parallel process group.

        Returns:
            out: [B, *spatial_dims, hidden_dim] (same as input)
        """
        # CP communication for self-attention (Megatron-style):
        # CP only splits sequence/spatial dimension, not hidden_dim or heads
        # Pattern: All-gather K/V along sequence, keep Q local (or gather Q too for non-causal)

        if cp_group is not None and cp_group.size() > 1:
            raise ValueError("Context parallelism must be revisited.")
            # For non-causal attention, gather full sequence for Q, K, V
            # Input: [B, *spatial_partial, hidden_dim] where spatial_partial = spatial/cp_size
            # Output: [B, *spatial_full, hidden_dim]
            query = zigzag_gather_from_group_ranks(query, group=cp_group, seq_dim=1)
            key = zigzag_gather_from_group_ranks(key, group=cp_group, seq_dim=1)
            value = zigzag_gather_from_group_ranks(value, group=cp_group, seq_dim=1)

        # [B, * spatial_dims, hidden_dim] -> [B * num_heads, * spatial_dims, head_dim]
        # All heads are computed locally (no head splitting with CP)
        query = rearrange(query, "b ... (h d) -> (b h) ... d", h=self.num_heads)
        key = rearrange(key, "b ... (h d) -> (b h) ... d", h=self.num_heads)
        value = rearrange(value, "b ... (h d) -> (b h) ... d", h=self.num_heads)
        local_num_heads = self.num_heads  # TODO(@farhad): This looks to me like an error.

        # Optional RoPE positional encoding (before normalization)
        if self.use_rope:
            # Get the dimensionality of the input and apply RoPE based on the dimensionality
            dimensionality_input = query.ndim - 2
            if dimensionality_input == 1:
                assert query.shape[-1] % 2 == 0, (
                    f"With 1D RoPE, the number of channels must be divisible by 2. Got {query.shape[-1]}."
                )
                assert key.shape[-1] % 2 == 0, (
                    f"With 1D RoPE, the number of channels must be divisible by 2. Got {key.shape[-1]}."
                )
                # Gather or contsruct the RoPE 1D cache
                _, seq_len, hidden_dim = query.shape
                rope_1d_cache = self._rope_cache_1d(seq_len, hidden_dim, query.device, query.dtype)
                # Apply RoPE to query and key
                query = rope.apply_rope_1d_blh(query, rope_1d_cache)
                key = rope.apply_rope_1d_blh(key, rope_1d_cache)

            elif dimensionality_input == 2:
                assert query.shape[-1] % 4 == 0, (
                    f"With 2D RoPE, the number of channels must be divisible by 4. Got {query.shape[-1]}."
                )
                assert key.shape[-1] % 4 == 0, (
                    f"With 2D RoPE, the number of channels must be divisible by 4. Got {key.shape[-1]}."
                )
                # Gather or contsruct the RoPE 2D cache
                _, height, width, hidden_dim = query.shape
                rope_2d_cache = self._rope_cache_2d(height, width, hidden_dim // 2, query.device, query.dtype)
                # Apply RoPE to query and key
                query = rope.apply_rope_2d_blh(query, rope_2d_cache)
                key = rope.apply_rope_2d_blh(key, rope_2d_cache)

            elif dimensionality_input == 3:
                assert query.shape[-1] % 6 == 0, (
                    f"With 3D RoPE, the number of channels must be divisible by 6. Got {query.shape[-1]}."
                )
                assert key.shape[-1] % 6 == 0, (
                    f"With 3D RoPE, the number of channels must be divisible by 6. Got {key.shape[-1]}."
                )
                # Gather or contsruct the RoPE 3D cache
                _, depth, height, width, hidden_dim = query.shape
                rope_3d_cache = self._rope_cache_3d(depth, height, width, hidden_dim // 3, query.device, query.dtype)
                # Apply RoPE to query and key
                query = rope.apply_rope_3d_blh(query, rope_3d_cache)
                key = rope.apply_rope_3d_blh(key, rope_3d_cache)

            else:
                raise NotImplementedError(f"RoPE is not implemented for {dimensionality_input}D inputs.")

        # Optional QK normalization (after RoPE)
        if self.apply_qk_norm:
            query, key = qk_norm.apply_qk_norm(query, key, dim=-1)

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
                rearrange(query.to(torch.bfloat16), "(b h) t d -> b h t d", h=local_num_heads),
                rearrange(key.to(torch.bfloat16), "(b h) t d -> b h t d", h=local_num_heads),
                rearrange(value.to(torch.bfloat16), "(b h) t d -> b h t d", h=local_num_heads),
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=self.is_causal,
                # Scale is 1.0 if QK normalization is applied, otherwise self.scale
                # When you L2-normalize Q and K, you are effectively doing cosine attention.
                # PyTorch SDPA still applies the 1/sqrt(d) scaling to the logits. That makes the
                # logits too small by a factor sqrt(d), flattening attention and harming learning.
                scale=self.scale if not self.apply_qk_norm else 1.0,
            )
        # Convert back to original dtype [batch_size, local_num_heads, flatten(* spatial_dims), head_dim]
        out = out.to(in_dtype)

        # Merge local heads: [batch_size, local_num_heads, flatten(* spatial_dims), head_dim] -> [batch_size, flatten(* spatial_dims), (local_num_heads * head_dim)]
        out = rearrange(out, "b h t d -> b t (h d)", h=local_num_heads)

        # Unflatten back to spatial dims
        out = self._unflatten_spatial(out, spatial_shape)

        # CP communication: split sequence back to original distribution
        if cp_group is not None and cp_group.size() > 1:
            # Each rank has: [B, *spatial_full, hidden_dim] (full sequence, full hidden_dim)
            # Split sequence back: [B, *spatial_full, hidden_dim] -> [B, *spatial/cp, hidden_dim]
            out = zigzag_split_across_group_ranks(out, group=cp_group, seq_dim=1)

        return out
