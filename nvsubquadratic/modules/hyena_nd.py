# TODO: Add license header here


"""Hyena-ND: gated global convolutional mixer for 1D/2D/3D signals.

Computation graph (per-block):

    Q, K, V ← linear projections of input  (done outside this module)
         │
    short_conv([Q; K; V])                   depthwise short conv on concatenated QKV
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

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules._channels_first_utils import is_channels_first_norm
from nvsubquadratic.modules.distributed_depthwise_conv_nd import (
    DistributedDepthwiseConv1d,
    DistributedDepthwiseConv2d,
    DistributedDepthwiseConv3d,
)
from nvsubquadratic.parallel.a2a_comms import AllToAllSingleFunction


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
        qk_norm_cfg: Optional[LazyConfig] | None,
        output_norm_cfg: LazyConfig = LazyConfig(torch.nn.Identity)(),
        gate_nonlinear_2_cfg: Optional[LazyConfig] = None,
    ):
        """Constructor.

        Args:
            global_conv_cfg: Global (long-range) convolutional layer.
            short_conv_cfg: Short depthwise conv applied to concatenated [Q, K, V].
                Must produce a ConvNd, DistributedDepthwiseConvNd, or Identity.
            gate_nonlinear_cfg: Activation for the first multiplicative gate (e.g. SiLU).
                Use Identity for linear gating.
            pixelhyena_norm_cfg: Normalization between first gate and global conv.
                Use Identity to disable.
            qk_norm_cfg: Per-channel normalization for Q (and K when gate is Identity).
                None to disable.  Separate instances are created for Q and K to
                support stateful norms (e.g. RMSNorm with learnable scale).
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

        # QK Normalization (separate instances for Q and K to support stateful norms like RMSNorm).
        # K-norm is only useful when gating is linear (Identity); a nonlinear gate
        # (e.g. SiLU) already bounds K's magnitude, so we use Identity for k_norm.
        if qk_norm_cfg is not None:
            self.q_norm = instantiate(qk_norm_cfg)
            if isinstance(self.gate_nonlinear, torch.nn.Identity):
                self.k_norm = instantiate(qk_norm_cfg)
            else:
                self.k_norm = torch.nn.Identity()
        else:
            self.q_norm = None
            self.k_norm = None

    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        is_causal = getattr(self.global_conv, "is_causal", None)
        q_norm_str = self.q_norm.__class__.__name__ if self.q_norm is not None else "None"
        k_norm_str = self.k_norm.__class__.__name__ if self.k_norm is not None else "None"
        parts = [f"q_norm={q_norm_str}", f"k_norm={k_norm_str}"]
        if self.gate_nonlinear is not self.gate_nonlinear_2:
            g1 = self.gate_nonlinear.__class__.__name__
            g2 = self.gate_nonlinear_2.__class__.__name__
            parts.append(f"gates={g1}/{g2}")
        if is_causal is not None:
            parts.append(f"is_causal={is_causal}")
        return ", ".join(parts)

    def flop_count(self, spatial_dims: tuple[int, ...], inference: bool = False) -> int:
        """Count FLOPs for the Hyena gated global convolutional mixer.

        Let C = hidden_dim (per projection), S = prod(spatial_dims).

        FLOPs breakdown:
          1. Short depthwise conv on concatenated [Q, K, V] (3C channels):
             2 * 3C * S * k_prod,  where k_prod = product of kernel sizes.
             Each output element: k_prod MACs for 1 depthwise filter.
             Skipped when short_conv is Identity.
          2. QK-Norm (when ``self.q_norm is not None``):
             Q: 3 * C * S  (RMSNorm-like).
             K: 3 * C * S  only when ``self.gate_nonlinear`` is Identity
             (linear gating); a nonlinear σ(K) already bounds magnitude.
          3. First gate  Q ⊙ σ(K):  C * S (multiply).
             + C * S for activation on K if gate_nonlinear is not Identity.
          4. PixelHyena norm (if not Identity):  3 * C * S.
          5. Global convolution (CKConvND):
             Delegated to ``self.global_conv.flop_count(spatial_dims, inference)``.
          6. Second gate  h ⊙ σ₂(V):  C * S (multiply).
             + C * S for activation on V if gate_nonlinear_2 is not Identity.
          7. Output norm (if not Identity):  3 * C * S.

        Args:
            spatial_dims: Spatial dimensions of the input, e.g. (H, W) for 2D.
            inference: Passed through to CKConvND for kernel generation caching.

        Returns:
            Total FLOPs as an integer.
        """
        C = self.global_conv.hidden_dim
        S = 1
        for s in spatial_dims:
            S *= s

        flops = 0

        # 1. Short depthwise conv
        if not isinstance(self.short_conv, torch.nn.Identity):
            k_prod = 1
            for k in self.short_conv.kernel_size:
                k_prod *= k
            in_ch = self.short_conv.in_channels  # 3 * C
            groups = self.short_conv.groups
            out_ch = self.short_conv.out_channels
            flops += 2 * (in_ch // groups) * out_ch * S * k_prod

        # 2. QK-Norm (k_norm is Identity when gate is non-linear, so no extra FLOPs)
        if self.q_norm is not None:
            flops += 3 * C * S  # Q norm
            if not isinstance(self.k_norm, torch.nn.Identity):
                flops += 3 * C * S  # K norm (only for linear gating)

        # 3. First gate: Q * σ(K)
        flops += C * S  # elementwise multiply
        if not isinstance(self.gate_nonlinear, torch.nn.Identity):
            flops += C * S  # activation on K

        # 4. PixelHyena norm
        if not isinstance(self.pixelhyena_norm, torch.nn.Identity):
            flops += 3 * C * S

        # 5. Global convolution
        flops += self.global_conv.flop_count(spatial_dims, inference=inference)

        # 6. Second gate: h * σ₂(V)
        flops += C * S
        if not isinstance(self.gate_nonlinear_2, torch.nn.Identity):
            flops += C * S  # activation on V

        # 7. Output norm
        if not isinstance(self.output_norm, torch.nn.Identity):
            flops += 3 * C * S

        return flops

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
            query: ``[B, *spatial, C]`` query tensor (from linear projection of input).
            key: ``[B, *spatial, C]`` key tensor.
            value: ``[B, *spatial, C]`` value tensor.
            cp_group: Context-parallel process group.  None disables CP.
            **mixer_kwargs: Forwarded to the global conv (e.g. ``conditioning`` for FiLM).

        Returns:
            ``[B, *spatial, C]`` output tensor.
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

        # QK normalization.
        # Tensors are BHL: [B, C, *spatial]. Channel-first norms can operate
        # directly; channel-last norms need movedim(1, -1) / movedim(-1, 1).
        # K is only normalized when gate_nonlinear is Identity (linear gating),
        # because a nonlinear σ(K) already bounds the magnitude.
        if self.q_norm is not None:
            if is_channels_first_norm(self.q_norm):
                query = self.q_norm(query)
            else:
                query = self.q_norm(query.movedim(1, -1)).movedim(-1, 1)
            if isinstance(self.gate_nonlinear, torch.nn.Identity):
                if is_channels_first_norm(self.k_norm):
                    key = self.k_norm(key)
                else:
                    key = self.k_norm(key.movedim(1, -1)).movedim(-1, 1)

        # First gate: z = Q ⊙ σ(K)
        query = query * self.gate_nonlinear(key)

        # Apply PixelHyena normalization (use torch.nn.Identity for no normalization)
        if not isinstance(self.pixelhyena_norm, torch.nn.Identity):
            if is_channels_first_norm(self.pixelhyena_norm):
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
            if is_channels_first_norm(self.output_norm):
                y = self.output_norm(y)
            else:
                shape = y.shape  # [B, C, *spatial]
                y = y.movedim(1, -1).reshape(-1, shape[1])
                y = self.output_norm(y)
                y = y.view(shape[0], *shape[2:], shape[1]).movedim(-1, 1)

        # Reshape back to [B, * spatial_dims, C]
        return rearrange(y, "b c ... -> b ... c")
