# TODO: Add license header here


"""Hyena-ND: gated global convolutional mixer for 1D/2D/3D signals.

Computation graph (per-block):

    Q, K, V ← linear projections of input  (done outside this module)
         │
    short_conv([Q; K; V])                   depthwise short conv on concatenated QKV
         │
    RoPE(Q, K)                              optional rotary positional encoding
         │
    QK-Norm(Q [, K])                        optional per-channel normalization
         │                                  (K is only normalized when gate_nonlinear is Identity)
    z = Q ⊙ σ(K)                            first multiplicative gate
         │
    PixelHyena-Norm(z)                      optional normalization (GroupNorm / RMSNorm / ...)
         │
    h = GlobalConv(z)                       long-range convolution (FFTConv, etc.)
         │
    y = h ⊙ σ₂(V)                           second multiplicative gate
         │
    Output-Norm(y)                          optional normalization before projection
         │
    return y                                [B, *spatial, C]

σ denotes `gate_nonlinear` (first gate) and σ₂ denotes `gate_nonlinear_2`
(second gate).  By default σ₂ = σ.  When both are Identity the gates
reduce to plain element-wise products, recovering a linear variant closer
to Mamba's selective-scan formulation.  Setting σ=SiLU, σ₂=Sigmoid follows
the gated attention formulation.
"""

from typing import Optional

import torch
from einops import rearrange

from nvsubq_paper.lazy_config import LazyConfig, instantiate
from nvsubq_paper.modules.distributed_depthwise_conv_nd import (
    DistributedDepthwiseConv1d,
    DistributedDepthwiseConv2d,
    DistributedDepthwiseConv3d,
)
from nvsubquadratic.parallel.a2a_comms import AllToAllSingleFunction
from nvsubquadratic.utils import rope


class Hyena(torch.nn.Module):
    """Gated global convolutional mixer for ND signals.

    Two multiplicative gates sandwich a long-range (global) convolution:

        z = Q ⊙ σ(K)           — first gate
        h = GlobalConv(z)
        y = h ⊙ σ₂(V)          — second gate

    where σ is ``gate_nonlinear`` and σ₂ is ``gate_nonlinear_2`` (defaults
    to σ when not provided).  Setting both to Identity gives plain
    element-wise products, recovering a linear gating variant.

    Optional components (each disabled by passing Identity or None):
        - Short depthwise convolution on concatenated [Q, K, V]
        - Rotary positional encoding (RoPE) on Q and K (1D/2D/3D)
        - QK normalization (Q always; K only when σ = Identity)
        - PixelHyena normalization between first gate and global conv
        - Output normalization after second gate
        - Context parallelism via AllToAll communication
    """

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
        gate_nonlinear_2_cfg: Optional[LazyConfig] = None,
    ):
        """
        Args:
            global_conv_cfg: Global (long-range) convolutional layer.
            short_conv_cfg: Short depthwise conv applied to concatenated [Q, K, V].
                Must produce a ConvNd, DistributedDepthwiseConvNd, or Identity.
            gate_nonlinear_cfg: Activation for the first multiplicative gate (e.g. SiLU).
                Use Identity for linear gating.
            pixelhyena_norm_cfg: Normalization between first gate and global conv.
                Use Identity to disable.
            use_rope: Whether to apply rotary positional encoding to Q and K.
            qk_norm_cfg: Per-channel normalization for Q (and K when gate is Identity).
                None to disable.  Separate instances are created for Q and K to
                support stateful norms (e.g. RMSNorm with learnable scale).
            rope_base: Base frequency for RoPE (default: 10000.0).
            output_norm_cfg: Normalization after the second gate.  Defaults to Identity.
            gate_nonlinear_2_cfg: Activation for the second multiplicative gate.
                If None (default), reuses gate_nonlinear_cfg for both gates.
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

        # Nonlinear gates
        self.gate_nonlinear = instantiate(gate_nonlinear_cfg)
        if gate_nonlinear_2_cfg is not None:
            self.gate_nonlinear_2 = instantiate(gate_nonlinear_2_cfg)
        else:
            self.gate_nonlinear_2 = self.gate_nonlinear

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
        is_causal = getattr(self.global_conv, "is_causal", None)
        qk_norm_str = self.q_norm.__class__.__name__ if self.q_norm is not None else "None"
        parts = [f"qk_norm={qk_norm_str}", f"use_rope={self.use_rope}"]
        if self.gate_nonlinear is not self.gate_nonlinear_2:
            g1 = self.gate_nonlinear.__class__.__name__
            g2 = self.gate_nonlinear_2.__class__.__name__
            parts.append(f"gates={g1}/{g2}")
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
        **mixer_kwargs,
    ) -> torch.Tensor:
        """Compute  y = OutputNorm( GlobalConv( Norm( Q ⊙ σ(K) ) ) ⊙ σ(V) ).

        All tensors are channel-last on entry and exit.

        Args:
            query: [B, *spatial, C] query tensor (from linear projection of input).
            key: [B, *spatial, C] key tensor.
            value: [B, *spatial, C] value tensor.
            cp_group: Context-parallel process group.  None disables CP.
<<<<<<< HEAD:nvsubq_paper/modules/hyena_nd.py
=======
            **mixer_kwargs: Forwarded to the global conv (e.g. ``conditioning`` for FiLM).
>>>>>>> 2c15801 (DALI unification, WSD scheduler, checkpoint resume, reference-matching init, FiLM kernels (#61)):nvsubquadratic/modules/hyena_nd.py

        Returns:
            [B, *spatial, C] output tensor.
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

        # QK normalization (after RoPE).
        # Tensors are BHL: [B, C, *spatial]. Move C to last dim for norms that
        # expect channel-last (RMSNorm, LayerNorm, L2Norm with dim=-1).
        # K is only normalized when gate_nonlinear is Identity (linear gating),
        # because a nonlinear σ(K) already bounds the magnitude.
        if self.q_norm is not None:
            query = self.q_norm(query.movedim(1, -1)).movedim(-1, 1)
            if isinstance(self.gate_nonlinear, torch.nn.Identity):
                key = self.k_norm(key.movedim(1, -1)).movedim(-1, 1)

        # First gate: z = Q ⊙ σ(K)
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
        y = self.global_conv(query, is_bhl_input=True, cp_group=cp_group, **mixer_kwargs)

        # CP communication - scatter along first spatial dimension while gathering across channels/hidden dimension
        if cp_group is not None and cp_group.size() > 1:
            y = AllToAllSingleFunction.apply(y, cp_group, "full_to_split", True)

        # Second gate: y = h ⊙ σ₂(V)
        y = y * self.gate_nonlinear_2(value)

        # Output normalization (after the second gate).
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
