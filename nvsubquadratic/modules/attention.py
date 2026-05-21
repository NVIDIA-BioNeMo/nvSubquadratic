# TODO: Add license header here


"""Multi-head self-attention with optional QK normalization and Rotary Positional Embeddings (RoPE)."""

import torch
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.parallel.utils import zigzag_gather_from_group_ranks, zigzag_split_across_group_ranks
from nvsubquadratic.utils import qk_norm, rope


class Attention(torch.nn.Module):
    """Multi-head attention (support for self and cross-attention) with optional QK normalization and Rotary Positional Embeddings (RoPE).

    **Input / Output:**

    - Accepts sequences ``[B, T, C]`` or images ``[B, H, W, C]``.
    - Returns the same leading shape with channel dimension ``C``.

    **Dimensions:**

    - ``C`` must be divisible by ``num_heads``; per-head dim is ``D = C / num_heads``.

    **RoPE support:**

    - 1D (sequences): applied over length ``T``; requires ``D`` even.
    - 2D (images): applied over ``(H, W)`` by splitting per-head dim into two halves
      (Y-axis, X-axis); requires ``D`` divisible by 4.
    - 3D (videos or other 3D data): applied over ``(H, W, D)`` by splitting per-head dim into three halves
      (Z-axis, X-axis, Y-axis); requires ``D`` divisible by 8.

    **QK normalization:**

    - When enabled, queries and keys are L2-normalized per head along the last dimension.

    **Implementation notes:**

    - Uses PyTorch ``scaled_dot_product_attention`` and prefers FlashAttention kernels when available.

    **Context-parallel limitations:**

    - This implementation is for **illustration and compatibility only**.
    - It uses zigzag all-gather/split for CP, which causes significant memory issues at long context
      lengths because the attention layer does not implement internal ring attention.
    - For production use with long contexts, use PyTorch's standard context-parallel attention
      blocks (e.g., as in torchtitan: https://docs.pytorch.org/tutorials/unstable/context_parallel.html).
    - Future work: migrate to PyTorch's standard CP attention API, which may also eliminate
      the requirement for zigzag-style communication patterns.

    Args:
        hidden_dim (int): The dimension of the hidden states.
        num_heads (int): The number of attention heads.
        apply_qk_norm (bool): Whether to apply QK normalization.
        use_rope (bool): Whether to apply RoPE.
        is_causal (bool): Whether the attention is causal. Defaults to False.
        attn_dropout (float): The dropout rate for the attention weights.
        rope_base (float): The base of the RoPE.
        rope_spatial_dims (tuple[int, ...]): Required when ``use_rope=True``.
            Spatial dimensions used to precompute the RoPE cos/sin tables.
            Examples: ``(4096,)`` for 1D, ``(64, 64)`` for 2D, ``(8, 64, 64)`` for 3D.
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
        rope_spatial_dims: tuple[int, ...] | None = None,
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

        # ── Precomputed RoPE cos/sin buffers ──────────────────────────────
        #
        # Following the ViT5Attention pattern: cos/sin tables are built once
        # at init and stored as non-persistent registered buffers so they:
        #   - survive .to(device) / .half() without manual bookkeeping,
        #   - are visible to CUDA-graph capture (no dynamic allocation in forward),
        #   - cause no graph breaks with torch.compile (no dict lookups,
        #     no torch.no_grad/inference_mode context managers in forward),
        #   - are NOT serialised into checkpoints (persistent=False), since
        #     they are deterministically reconstructed from __init__ args.
        if self.use_rope:
            assert rope_spatial_dims is not None, (
                "rope_spatial_dims is required when use_rope=True. "
                "Pass a tuple of spatial dimensions, e.g. (4096,) for 1D, "
                "(64, 64) for 2D, (8, 64, 64) for 3D."
            )
            self._rope_ndim = len(rope_spatial_dims)

            if self._rope_ndim == 1:
                (seq_len,) = rope_spatial_dims
                assert self.head_dim % 2 == 0, f"1D RoPE requires head_dim divisible by 2, got {self.head_dim}"
                cos, sin = rope.construct_rope_1d_cache_blh(
                    seq_len, self.head_dim, torch.device("cpu"), torch.float32, self.rope_base
                )
                self.register_buffer("rope_cos", cos, persistent=False)
                self.register_buffer("rope_sin", sin, persistent=False)

            elif self._rope_ndim == 2:
                height, width = rope_spatial_dims
                assert self.head_dim % 4 == 0, f"2D RoPE requires head_dim divisible by 4, got {self.head_dim}"
                cos_y, sin_y, cos_x, sin_x = rope.construct_rope_2d_cache_blh(
                    height, width, self.head_dim // 2, torch.device("cpu"), torch.float32, self.rope_base
                )
                self.register_buffer("rope_cos_y", cos_y, persistent=False)
                self.register_buffer("rope_sin_y", sin_y, persistent=False)
                self.register_buffer("rope_cos_x", cos_x, persistent=False)
                self.register_buffer("rope_sin_x", sin_x, persistent=False)

            elif self._rope_ndim == 3:
                depth, height, width = rope_spatial_dims
                assert self.head_dim % 6 == 0, f"3D RoPE requires head_dim divisible by 6, got {self.head_dim}"
                cos_z, sin_z, cos_y, sin_y, cos_x, sin_x = rope.construct_rope_3d_cache_blh(
                    depth, height, width, self.head_dim // 3, torch.device("cpu"), torch.float32, self.rope_base
                )
                self.register_buffer("rope_cos_z", cos_z, persistent=False)
                self.register_buffer("rope_sin_z", sin_z, persistent=False)
                self.register_buffer("rope_cos_y", cos_y, persistent=False)
                self.register_buffer("rope_sin_y", sin_y, persistent=False)
                self.register_buffer("rope_cos_x", cos_x, persistent=False)
                self.register_buffer("rope_sin_x", sin_x, persistent=False)

            else:
                raise ValueError(f"rope_spatial_dims must be 1D, 2D, or 3D, got {self._rope_ndim}D")

    def extra_repr(self) -> str:
        """Extra repr for the Attention module."""
        return f"num_heads={self.num_heads}, apply_qk_norm={self.apply_qk_norm}, is_causal={self.is_causal}, attn_dropout={self.attn_dropout}, use_rope={self.use_rope}, rope_base={self.rope_base}"

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
            query: ``[B, *spatial_dims, hidden_dim]``
            key: ``[B, *spatial_dims, hidden_dim]``
            value: ``[B, *spatial_dims, hidden_dim]``
            cp_group: ``torch.distributed.ProcessGroup`` — context-parallel process group.

        Returns:
            out: ``[B, *spatial_dims, hidden_dim]`` (same as input)
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
        # Cos/sin buffers were precomputed at init — no dynamic allocation,
        # no dict lookups, no context managers; safe for torch.compile.
        if self.use_rope:
            if self._rope_ndim == 1:
                cache = (self.rope_cos, self.rope_sin)
                query = rope.apply_rope_1d_blh(query, cache)
                key = rope.apply_rope_1d_blh(key, cache)
            elif self._rope_ndim == 2:
                cache = (self.rope_cos_y, self.rope_sin_y, self.rope_cos_x, self.rope_sin_x)
                query = rope.apply_rope_2d_blh(query, cache)
                key = rope.apply_rope_2d_blh(key, cache)
            elif self._rope_ndim == 3:
                cache = (
                    self.rope_cos_z,
                    self.rope_sin_z,
                    self.rope_cos_y,
                    self.rope_sin_y,
                    self.rope_cos_x,
                    self.rope_sin_x,
                )
                query = rope.apply_rope_3d_blh(query, cache)
                key = rope.apply_rope_3d_blh(key, cache)

        # Optional QK normalization (after RoPE)
        if self.apply_qk_norm:
            query, key = qk_norm.apply_qk_norm(query, key, dim=-1)

        # Flatten spatial dims if present
        query, spatial_shape = self._flatten_spatial(query)
        key, _ = self._flatten_spatial(key)
        value, _ = self._flatten_spatial(value)

        # Reshape from [B*H, T, D] to [B, H, T, D] for SDPA
        query = rearrange(query, "(b h) t d -> b h t d", h=local_num_heads)
        key = rearrange(key, "(b h) t d -> b h t d", h=local_num_heads)
        value = rearrange(value, "(b h) t d -> b h t d", h=local_num_heads)

        # Scaled dot-product attention — let PyTorch auto-select the best
        # backend (CuDNN on H100, FlashAttention on A100, etc.).
        # No manual dtype cast: autocast handles precision.
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.is_causal,
            # When QK-norm is applied (cosine attention), disable the default
            # 1/sqrt(d) scaling — it would flatten the normalised logits.
            scale=self.scale if not self.apply_qk_norm else 1.0,
        )

        # Merge heads: [B, H, T, D] -> [B, T, H*D]
        out = rearrange(out, "b h t d -> b t (h d)", h=local_num_heads)

        # Unflatten back to spatial dims
        out = self._unflatten_spatial(out, spatial_shape)

        # CP communication: split sequence back to original distribution
        if cp_group is not None and cp_group.size() > 1:
            # Each rank has: [B, *spatial_full, hidden_dim] (full sequence, full hidden_dim)
            # Split sequence back: [B, *spatial_full, hidden_dim] -> [B, *spatial/cp, hidden_dim]
            out = zigzag_split_across_group_ranks(out, group=cp_group, seq_dim=1)

        return out
