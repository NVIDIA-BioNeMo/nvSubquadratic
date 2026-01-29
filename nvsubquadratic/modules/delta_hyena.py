# TODO: Add license header here

"""Explicit Associative Delta-Hyena implementation inheriting from HyenaND."""

import torch
from einops import rearrange

from torch.utils.checkpoint import checkpoint
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.ops.delta_rule import delta_rule_scan
from nvsubquadratic.parallel.a2a_comms import AllToAllSingleFunction
from nvsubquadratic.utils import qk_norm, rope


class DeltaHyena(Hyena):
    """Associative Delta-Hyena module.
    
    This module inherits from Hyena to reuse RoPE caching and short-convolution logic,
    but implements its own forward pass to incorporate the Delta Rule recursive update.
    """

    def __init__(
        self,
        global_conv_cfg: LazyConfig,
        short_conv_cfg: LazyConfig,
        gate_nonlinear_cfg: LazyConfig,
        pixelhyena_norm_cfg: LazyConfig,
        apply_qk_norm: bool,
        use_rope: bool,
        rope_base: float = 10000.0,
        output_norm_cfg: LazyConfig = LazyConfig(torch.nn.Identity)(),
        # Delta-specific arguments
        hidden_dim: int = 0,
        num_heads: int = 8,
        gamma_init: float = 0.1,
    ):
        """Initialize the Associative Delta-Hyena mixer.

        Args:
            global_conv_cfg: LazyConfig - For the Hyena filter on the value branch.
            short_conv_cfg: LazyConfig - For the short convolutional projections.
            gate_nonlinear_cfg: LazyConfig - For the gate nonlinearity.
            pixelhyena_norm_cfg: LazyConfig - For the pixelhyena normalization.
            apply_qk_norm: bool - Whether to apply QK normalization.
            use_rope: bool - Whether to use RoPE.
            rope_base: float - RoPE base frequency.
            output_norm_cfg: LazyConfig - For the output normalization.
            hidden_dim: int - Total hidden dimension.
            num_heads: int - Number of associative memory heads.
            gamma_init: float - Initial value for learnable learning rate beta.
        """
        super().__init__(
            global_conv_cfg=global_conv_cfg,
            short_conv_cfg=short_conv_cfg,
            gate_nonlinear_cfg=gate_nonlinear_cfg,
            pixelhyena_norm_cfg=pixelhyena_norm_cfg,
            apply_qk_norm=apply_qk_norm,
            use_rope=use_rope,
            rope_base=rope_base,
            output_norm_cfg=output_norm_cfg,
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim > 0, "hidden_dim must be specifies and > 0"
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Learnable learning rate beta (per head, per head_dim)
        self.beta = torch.nn.Parameter(torch.ones(num_heads, self.head_dim) * gamma_init)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cp_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """Forward pass of Associative Delta-Hyena.
        
        Args:
            query: (B, *spatial, D)
            key: (B, *spatial, D)
            value: (B, *spatial, D)
            cp_group: Context parallel group.
            
        Returns:
            (B, *spatial, D)
        """
        orig_shape = query.shape
        B = orig_shape[0]
        
        # 1. Reshape for short convolutional projections [B, C, *spatial]
        query = rearrange(query, "b ... c -> b c ...")
        key = rearrange(key, "b ... c -> b c ...")
        value = rearrange(value, "b ... c -> b c ...")

        # Apply short convolutional projection (inherited logic)
        if not isinstance(self.short_conv, torch.nn.Identity):
            x = torch.cat([query, key, value], dim=1)  # [B, 3 * hidden_dim, *spatial_dims]

            if cp_group is not None and cp_group.size() > 1:
                x = AllToAllSingleFunction.apply(x, cp_group, "split_to_full", True)

            if hasattr(self.short_conv, "__class__") and "Distributed" in self.short_conv.__class__.__name__:
                x = self.short_conv(x, cp_group)
            else:
                x = self.short_conv(x)

            if cp_group is not None and cp_group.size() > 1:
                x = AllToAllSingleFunction.apply(x, cp_group, "full_to_split", True)

            query, key, value = x.split(query.shape[1], dim=1)
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # 2. Optional RoPE positional encoding (using inherited cache methods)
        if self.use_rope:
            dimensionality_input = query.ndim - 2
            if dimensionality_input == 1:
                _, hidden_dim, seq_len = query.shape
                rope_1d_cache = self._rope_cache_1d(seq_len, hidden_dim, query.device, query.dtype)
                query = rope.apply_rope_1d_bhl(query, rope_1d_cache)
                key = rope.apply_rope_1d_bhl(key, rope_1d_cache)
            elif dimensionality_input == 2:
                _, hidden_dim, height, width = query.shape
                rope_2d_cache = self._rope_cache_2d(height, width, hidden_dim // 2, query.device, query.dtype)
                query = rope.apply_rope_2d_bhl(query, rope_2d_cache)
                key = rope.apply_rope_2d_bhl(key, rope_2d_cache)
            elif dimensionality_input == 3:
                _, hidden_dim, depth, height, width = query.shape
                rope_3d_cache = self._rope_cache_3d(depth, height, width, hidden_dim // 3, query.device, query.dtype)
                query = rope.apply_rope_3d_bhl(query, rope_3d_cache)
                key = rope.apply_rope_3d_bhl(key, rope_3d_cache)
            else:
                raise NotImplementedError(f"RoPE is not implemented for {dimensionality_input}D inputs.")

        # 3. Apply QK normalization
        if self.apply_qk_norm:
            query, key = qk_norm.apply_qk_norm(query, key, dim=1)

        # 4. Hyena Filtering on the Value branch
        # Apply PixelHyena normalization to value branch if configured
        v_spatial = value
        if not isinstance(self.pixelhyena_norm, torch.nn.Identity):
            if isinstance(self.pixelhyena_norm, torch.nn.GroupNorm):
                v_spatial = self.pixelhyena_norm(v_spatial)
            else:  # torch.nn.LayerNorm
                v_spatial = rearrange(v_spatial, "b c ... -> b ... c")
                v_spatial = self.pixelhyena_norm(v_spatial)
                v_spatial = rearrange(v_spatial, "b ... c -> b c ...")

        # Global convolution on value branch to provide spatial context
        if not isinstance(self.global_conv, torch.nn.Identity):
            v_spatial = self.global_conv(v_spatial, is_bhl_input=True, cp_group=cp_group)

        # 5. Associative Memory Update (Delta Rule)
        # Flatten spatial dims and reshape to (B, L, H, D)
        q_flat = rearrange(query, "b c ... -> b (...) c").reshape(B, -1, self.num_heads, self.head_dim)
        k_flat = rearrange(key, "b c ... -> b (...) c").reshape(B, -1, self.num_heads, self.head_dim)
        v_flat = rearrange(v_spatial, "b c ... -> b (...) c").reshape(B, -1, self.num_heads, self.head_dim)
        
        # Key normalization for specific Delta rule stability
        eps = 1e-6
        k_flat = k_flat / (torch.norm(k_flat, dim=-1, keepdim=True) + eps)
        
        # Apply the recursive Delta rule logic
        if self.training:
            y_flat = checkpoint(delta_rule_scan, q_flat, k_flat, v_flat, self.beta, use_reentrant=False)
        else:
            y_flat = delta_rule_scan(q_flat, k_flat, v_flat, self.beta)
        
        # 6. Final Output Processing
        # Restore spatial shape [B, C, *spatial]
        y = y_flat.reshape(B, -1, orig_shape[-1])
        y = rearrange(y, "b l c -> b c l").reshape(B, orig_shape[-1], *orig_shape[1:-1])
        
        # Optional output normalization
        if not isinstance(self.output_norm, torch.nn.Identity):
            if isinstance(self.output_norm, torch.nn.GroupNorm):
                y = self.output_norm(y)
            else:  # torch.nn.LayerNorm / torch.nn.RMSNorm
                output_tmp = rearrange(y, "b c ... -> b ... c")
                output_tmp = self.output_norm(output_tmp)
                y = rearrange(output_tmp, "b ... c -> b c ...")


        # Final reshape back to [B, *spatial_dims, C]
        return rearrange(y, "b c ... -> b ... c")


class ReasoningDeltaHyena(Hyena):
    """Reasoning Delta-Hyena module with multi-pass iterative refinement.
    
    This module performs multiple passes over the same sequence, refining its
    associative memory state and output using the Reasoning Delta Rule.
    """

    def __init__(
        self,
        global_conv_cfg: LazyConfig,
        short_conv_cfg: LazyConfig,
        gate_nonlinear_cfg: LazyConfig,
        pixelhyena_norm_cfg: LazyConfig,
        apply_qk_norm: bool,
        use_rope: bool,
        rope_base: float = 10000.0,
        output_norm_cfg: LazyConfig = LazyConfig(torch.nn.Identity)(),
        # Delta-specific arguments
        hidden_dim: int = 0,
        num_heads: int = 8,
        gamma_init: float = 0.1,
        # Reasoning-specific arguments
        num_recurrence: int = 3,
    ):
        """Initialize the Reasoning Delta-Hyena mixer."""
        super().__init__(
            global_conv_cfg=global_conv_cfg,
            short_conv_cfg=short_conv_cfg,
            gate_nonlinear_cfg=gate_nonlinear_cfg,
            pixelhyena_norm_cfg=pixelhyena_norm_cfg,
            apply_qk_norm=apply_qk_norm,
            use_rope=use_rope,
            rope_base=rope_base,
            output_norm_cfg=output_norm_cfg,
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_recurrence = num_recurrence
        
        from nvsubquadratic.ops.delta_rule import reasoning_delta_rule_scan
        self._reasoning_scan = reasoning_delta_rule_scan

        # Learnable learning rate beta
        self.beta = torch.nn.Parameter(torch.ones(num_heads, self.head_dim) * gamma_init)
        
        # Linear refinement to update values based on previous pass output
        self.refinement_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.zeros_(self.refinement_proj.weight)
        torch.nn.init.zeros_(self.refinement_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cp_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """Forward pass of Reasoning Delta-Hyena."""
        orig_shape = query.shape
        B = orig_shape[0]
        
        # 1. Reshape and Short Conv (same as DeltaHyena)
        query = rearrange(query, "b ... c -> b c ...")
        key = rearrange(key, "b ... c -> b c ...")
        value = rearrange(value, "b ... c -> b c ...")

        if not isinstance(self.short_conv, torch.nn.Identity):
            x = torch.cat([query, key, value], dim=1)
            x = self.short_conv(x)
            query, key, value = x.split(query.shape[1], dim=1)

        # 2. RoPE
        if self.use_rope:
            dimensionality_input = query.ndim - 2
            if dimensionality_input == 1:
                rope_1d_cache = self._rope_cache_1d(query.shape[-1], self.hidden_dim, query.device, query.dtype)
                query = rope.apply_rope_1d_bhl(query, rope_1d_cache)
                key = rope.apply_rope_1d_bhl(key, rope_1d_cache)
            elif dimensionality_input == 2:
                rope_2d_cache = self._rope_cache_2d(query.shape[-2], query.shape[-1], self.hidden_dim // 2, query.device, query.dtype)
                query = rope.apply_rope_2d_bhl(query, rope_2d_cache)
                key = rope.apply_rope_2d_bhl(key, rope_2d_cache)

        # 3. QK Norm
        if self.apply_qk_norm:
            query, key = qk_norm.apply_qk_norm(query, key, dim=1)

        # 4. Global Conv on Value
        v_spatial = value
        if not isinstance(self.global_conv, torch.nn.Identity):
            v_spatial = self.global_conv(v_spatial, is_bhl_input=True, cp_group=cp_group)

        # 5. Reasoning Loop
        q_flat = rearrange(query, "b c ... -> b (...) c").reshape(B, -1, self.num_heads, self.head_dim)
        k_flat = rearrange(key, "b c ... -> b (...) c").reshape(B, -1, self.num_heads, self.head_dim)
        v_base = rearrange(v_spatial, "b c ... -> b (...) c").reshape(B, -1, self.num_heads, self.head_dim)
        
        # Key normalization
        eps = 1e-6
        k_flat = k_flat / (torch.norm(k_flat, dim=-1, keepdim=True) + eps)
        
        y_pass = None
        state = None
        
        for k_idx in range(self.num_recurrence):
            # Refine values based on previous pass output
            if y_pass is not None:
                # y_pass is (B, L, H*D)
                v_refined = v_base + self.refinement_proj(y_pass).reshape(B, -1, self.num_heads, self.head_dim)
            else:
                v_refined = v_base
                
            if self.training:
                y_pass, state = checkpoint(self._reasoning_scan, q_flat, k_flat, v_refined, self.beta, state, use_reentrant=False)
            else:
                y_pass, state = self._reasoning_scan(q_flat, k_flat, v_refined, self.beta, state)
            
            # Prepare y_pass for next iteration (reshape to hidden_dim)
            y_pass = y_pass.reshape(B, -1, self.hidden_dim)

        # 6. Post-processing
        y = y_pass.reshape(B, -1, self.hidden_dim)
        y = rearrange(y, "b l c -> b c l").reshape(B, self.hidden_dim, *orig_shape[1:-1])
        
        if not isinstance(self.output_norm, torch.nn.Identity):
            output_tmp = rearrange(y, "b c ... -> b ... c")
            output_tmp = self.output_norm(output_tmp)
            y = rearrange(output_tmp, "b ... c -> b c ...")

        return rearrange(y, "b c ... -> b ... c")
