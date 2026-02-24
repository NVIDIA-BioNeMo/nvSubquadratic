# TODO: Add license header here


"""Hyena-style global convolutional mixer implementation for ND signals."""

from typing import Optional

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.distributed_depthwise_conv_nd import (
    DistributedDepthwiseConv1d,
    DistributedDepthwiseConv2d,
    DistributedDepthwiseConv3d,
)
from nvsubquadratic.parallel.a2a_comms import AllToAllSingleFunction
from nvsubquadratic.utils import rope


class Hyena(torch.nn.Module):
    """Hyena-style global convolutional mixer."""

    def __init__(
        self,
        global_conv_cfg: LazyConfig,
        short_conv_cfg: LazyConfig,
        gate_nonlinear_cfg: LazyConfig,
        pixelhyena_norm_cfg: LazyConfig,
        use_rope: bool,
        qk_norm_cfg: Optional[LazyConfig] | None,
        rope_base: float = 10000.0,
        output_norm_cfg: LazyConfig = LazyConfig(torch.nn.Identity)(),
    ):
        """Initialize the Hyena-style global convolutional mixer.

        Args:
            global_conv_cfg: LazyConfig - LazyConfig for the global convolutional layer.
            short_conv_cfg: LazyConfig - LazyConfig for the short convolutional layer.
            gate_nonlinear_cfg: LazyConfig - LazyConfig for the gate nonlinear layer.
            pixelhyena_norm_cfg: LazyConfig - LazyConfig for the pixelhyena normalization layer. Use torch.nn.Identity for no normalization.
            use_rope: bool - Whether to use RoPE.
            qk_norm_cfg: Optional[LazyConfig] - LazyConfig for Q/K normalization (e.g., L2Norm, RMSNorm). None to disable.
            rope_base: float - The base of the RoPE (default: 10000.0).
            output_norm_cfg: LazyConfig - LazyConfig for the output normalization layer. Defaults to torch.nn.Identity.

        Raises:
            AssertionError: If the short conv is not an instance of torch.nn.ConvNd (1d, 2d, or 3d) or torch.nn.Identity.
        """
        super().__init__()

        # Core global convs: feature and gate branches
        self.global_conv = instantiate(global_conv_cfg)
        self.short_conv = instantiate(short_conv_cfg)
        assert isinstance(
            self.short_conv,
            (
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Conv3d,
                torch.nn.Identity,
                DistributedDepthwiseConv1d,
                DistributedDepthwiseConv2d,
                DistributedDepthwiseConv3d,
            ),
        ), (
            f"Short conv must be an instance of torch.nn.ConvNd (1d, 2d, or 3d) or torch.nn.Identity. Current type: {type(self.short_conv)}"
        )

        # Nonlinear gate
        self.gate_nonlinear = instantiate(gate_nonlinear_cfg)

        # Pixelhyena normalization (use torch.nn.Identity for no normalization)
        self.pixelhyena_norm = instantiate(pixelhyena_norm_cfg)
        # Exclude self.pixelhyena_norm from the parameter group with weight decay
        for param in self.pixelhyena_norm.parameters():
            param._no_weight_decay = True

        # Optional value normalization (use torch.nn.Identity for no normalization)
        self.output_norm = instantiate(output_norm_cfg)
        for param in self.output_norm.parameters():
            param._no_weight_decay = True

        # QK Normalization (separate instances for Q and K to support stateful norms like RMSNorm)
        if qk_norm_cfg is not None:
            self.q_norm = instantiate(qk_norm_cfg)
            self.k_norm = instantiate(qk_norm_cfg)
        else:
            self.q_norm = None
            self.k_norm = None

        # RoPE
        self.use_rope = use_rope
        self.rope_base = rope_base
        # RoPE caches (keyed by shape, dtype, device)
        if self.use_rope:
            self._rope1d_cache = {}
            self._rope2d_cache = {}
            self._rope3d_cache = {}

    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        # Get is_causal from global_conv if it has that attribute
        is_causal = getattr(self.global_conv, "is_causal", None)
        qk_norm_str = self.q_norm.__class__.__name__ if self.q_norm is not None else "None"
        parts = [f"qk_norm={qk_norm_str}", f"use_rope={self.use_rope}"]
        if is_causal is not None:
            parts.append(f"is_causal={is_causal}")
        return ", ".join(parts)

    def _rope_cache_1d(self, seq_len: int, dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute and cache 1D RoPE tables for input of length seq_len.

        Args:
            seq_len: Number of positions T.
            dim: Per-axis channel dimension. Must be even because rotations operate on pairs.
            device: Target device for the cached tensors.
            dtype: Target dtype for the cached tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)``
            where shapes are:
            - cos: [dim, seq_len]
            - sin: [dim, seq_len]

        Notes:
            The cache key is ``(T, D_axis, device, dtype)``. Tables are built under ``torch.no_grad()`` and reused across calls.
        """
        key = (int(seq_len), int(dim), str(device), str(dtype))
        if key in self._rope1d_cache:
            return self._rope1d_cache[key]
        # If not in cache, compute and cache
        with torch.inference_mode(False):
            with torch.no_grad():
                cos, sin = rope.construct_rope_1d_cache_bhl(seq_len, dim, device, dtype, self.rope_base)
                self._rope1d_cache[key] = (cos, sin)
        return cos, sin

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
            - cos_y/sin_y: [dim_half, height]
            - cos_x/sin_x: [dim_half, width]

        Notes:
            The cache key is ``(H, W, D_axis, device, dtype)``. Tables are built under ``torch.no_grad()`` and reused across calls.
        """
        key = (int(height), int(width), int(dim_half), str(device), str(dtype))
        if key in self._rope2d_cache:
            return self._rope2d_cache[key]
        # If not in cache, compute and cache
        with torch.inference_mode(False):
            with torch.no_grad():
                cos_y, sin_y, cos_x, sin_x = rope.construct_rope_2d_cache_bhl(
                    height, width, dim_half, device, dtype, self.rope_base
                )
                self._rope2d_cache[key] = (cos_y, sin_y, cos_x, sin_x)
        return cos_y, sin_y, cos_x, sin_x

    def _rope_cache_3d(
        self, depth: int, height: int, width: int, dim_third: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute and cache 3D RoPE tables for Z, Y, and X axes.

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
            - cos_z, sin_z: [dim_third, depth]
            - cos_y, sin_y: [dim_third, height]
            - cos_x, sin_x: [dim_third, width]

        Notes:
            The cache key is ``(D, H, W, D_axis, device, dtype)``. Tables are built under ``torch.no_grad()`` and reused across calls.
        """
        key = (int(depth), int(height), int(width), int(dim_third), str(device), str(dtype))
        if key in self._rope3d_cache:
            return self._rope3d_cache[key]
        # If not in cache, compute and cache
        with torch.inference_mode(False):
            with torch.no_grad():
                cos_z, sin_z, cos_y, sin_y, cos_x, sin_x = rope.construct_rope_3d_cache_bhl(
                    depth, height, width, dim_third, device, dtype, self.rope_base
                )
                self._rope3d_cache[key] = (cos_z, sin_z, cos_y, sin_y, cos_x, sin_x)
        return cos_z, sin_z, cos_y, sin_y, cos_x, sin_x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cp_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """Forward pass of the Hyena-style global convolutional mixer.

        Args:
            query: torch.Tensor - The query tensor of shape (batch_size, * spatial_dims, hidden_dim).
            key: torch.Tensor - The key tensor of shape (batch_size, * spatial_dims, hidden_dim).
            value: torch.Tensor - The value tensor of shape (batch_size, * spatial_dims, hidden_dim).
            cp_group: torch.distributed.ProcessGroup - Context parallel process group.

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
            x = torch.cat([query, key, value], dim=1)  # [B, 3 * hidden_dim, *spatial_dims]

            # CP communication - gather along first spatial dimension while splitting across channels/hidden dimension
            if cp_group is not None and cp_group.size() > 1:
                x = AllToAllSingleFunction.apply(x, cp_group, "split_to_full", True)

            # Always pass cp_group to distributed convolutions
            if hasattr(self.short_conv, "__class__") and "Distributed" in self.short_conv.__class__.__name__:
                x = self.short_conv(x, cp_group)
            else:
                # Standard PyTorch convolution doesn't support cp_group
                x = self.short_conv(x)

            # CP communication - scatter along first spatial dimension while gathering across channels/hidden dimension
            if cp_group is not None and cp_group.size() > 1:
                x = AllToAllSingleFunction.apply(x, cp_group, "full_to_split", True)

            # Split query, key, and value along channels/hidden dimension
            query, key, value = x.split(query.shape[1], dim=1)
            # Avoid in-place ops on views returned by split
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # Optional RoPE positional encoding (before normalization)
        if self.use_rope:
            # Get the dimensionality of the input and apply RoPE based on the dimensionality
            dimensionality_input = query.ndim - 2
            if dimensionality_input == 1:
                assert query.shape[1] % 2 == 0, (
                    f"With 1D RoPE, the number of channels must be divisible by 2. Got {query.shape[1]}."
                )
                assert key.shape[1] % 2 == 0, (
                    f"With 1D RoPE, the number of channels must be divisible by 2. Got {key.shape[1]}."
                )
                # Gather or contsruct the RoPE 1D cache
                _, hidden_dim, seq_len = query.shape
                rope_1d_cache = self._rope_cache_1d(seq_len, hidden_dim, query.device, query.dtype)
                # Apply RoPE to query and key
                query = rope.apply_rope_1d_bhl(query, rope_1d_cache)
                key = rope.apply_rope_1d_bhl(key, rope_1d_cache)

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
                assert query.shape[1] % 6 == 0, (
                    f"With 3D RoPE, the number of channels must be divisible by 6. Got {query.shape[1]}."
                )
                assert key.shape[1] % 6 == 0, (
                    f"With 3D RoPE, the number of channels must be divisible by 6. Got {key.shape[1]}."
                )
                # Gather or contsruct the RoPE 3D cache
                _, hidden_dim, depth, height, width = query.shape
                rope_3d_cache = self._rope_cache_3d(depth, height, width, hidden_dim // 3, query.device, query.dtype)
                # Apply RoPE to query and key
                query = rope.apply_rope_3d_bhl(query, rope_3d_cache)
                key = rope.apply_rope_3d_bhl(key, rope_3d_cache)

            else:
                raise NotImplementedError(f"RoPE is not implemented for {dimensionality_input}D inputs.")

        # Apply QK normalization (after RoPE).
        # Tensors are BHL: [B, C, *spatial]. Move C to last dim for norms that
        # expect channel-last (RMSNorm, LayerNorm, L2Norm with dim=-1).
        if self.q_norm is not None:
            query = self.q_norm(query.movedim(1, -1)).movedim(-1, 1)
            if isinstance(self.gate_nonlinear, torch.nn.Identity):
                key = self.k_norm(key.movedim(1, -1)).movedim(-1, 1)
            # key = self.k_norm(key.movedim(1, -1)).movedim(-1, 1)

        # First gate
        # z = query * key. We remove the nonlinearity here to align more with the Mamba defition.
        query = query * self.gate_nonlinear(key)

        # Apply PixelHyena normalization (use torch.nn.Identity for no normalization)
        if not isinstance(self.pixelhyena_norm, torch.nn.Identity):
            if isinstance(self.pixelhyena_norm, torch.nn.GroupNorm):
                query = self.pixelhyena_norm(query)
            else:
                shape = query.shape  # [B, C, *spatial]
                query = query.movedim(1, -1).reshape(-1, shape[1])
                query = self.pixelhyena_norm(query)
                query = query.view(shape[0], *shape[2:], shape[1]).movedim(-1, 1)

        # CP communication - gather along first spatial dimension while splitting across channels/hidden dimension
        if cp_group is not None and cp_group.size() > 1:
            query = AllToAllSingleFunction.apply(query, cp_group, "split_to_full", True)

        # Apply global convolution
        y = self.global_conv(query, is_bhl_input=True, cp_group=cp_group)

        # CP communication - scatter along first spatial dimension while gathering across channels/hidden dimension
        if cp_group is not None and cp_group.size() > 1:
            y = AllToAllSingleFunction.apply(y, cp_group, "full_to_split", True)

        # Second gate
        y = y * self.gate_nonlinear(value)

        # Optional value normalization before applying the second gate.
        # We add a normalization layer at the end of the second gate to align more with the Mamba defition.
        # In particular, this normalization in combination with the previous nonlinearity could help us get
        # rid of the problems around using circular convolutions.
        if not isinstance(self.output_norm, torch.nn.Identity):
            if isinstance(self.output_norm, torch.nn.GroupNorm):
                y = self.output_norm(y)
            else:
                shape = y.shape  # [B, C, *spatial]
                y = y.movedim(1, -1).reshape(-1, shape[1])
                y = self.output_norm(y)
                y = y.view(shape[0], *shape[2:], shape[1]).movedim(-1, 1)

        # Reshape back to [B, * spatial_dims, C]
        return rearrange(y, "b c ... -> b ... c")
