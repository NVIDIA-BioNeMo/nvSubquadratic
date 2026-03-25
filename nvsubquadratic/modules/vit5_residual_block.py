"""ViT-5 Residual Block: Pre-norm + Attention/MLP with LayerScale and DropPath.

Architecture per the ViT-5 paper (Wang et al., 2026):
    x = x + DropPath(LayerScale(Attention(Norm(x))))
    x = x + DropPath(LayerScale(MLP(Norm(x))))
"""

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.drop_path import DropPath
from nvsubquadratic.modules.layer_scale import LayerScale


class ViT5ResidualBlock(nn.Module):
    """ViT-5 style residual block with LayerScale and stochastic depth.

    Optionally owns a ``RegisterPooling`` module: when configured, registers
    are extracted from the *normalized* input (after ``input_norm``) and pooled
    into a conditioning vector that is threaded through ``**mixer_kwargs`` to
    the sequence mixer (and ultimately to the SIREN kernel for FiLM).

    Args:
        sequence_mixer_cfg: LazyConfig for the sequence mixer (QKVSequenceMixer wrapping Attention).
        sequence_mixer_norm_cfg: LazyConfig for the pre-norm before the sequence mixer.
        mlp_cfg: LazyConfig for the MLP.
        mlp_norm_cfg: LazyConfig for the pre-norm before the MLP.
        hidden_dim: Channel dimension (needed for LayerScale).
        layer_scale_init: Initial value for LayerScale gammas. Set to 0 to disable LayerScale.
        drop_path_rate: Stochastic depth drop probability.
        register_pooling_cfg: Optional LazyConfig for RegisterPooling. When provided,
            register tokens are extracted from the normalized input and pooled.
        num_registers: Number of register tokens (needed for extraction). Only used
            when register_pooling_cfg is provided.
        register_start_idx: Start index of register tokens in the sequence.
            Default 1 for [CLS, regs, patches] layout; set to 0 for GAP models
            without CLS where registers are prepended directly.
        grn_cfg: Optional LazyConfig for GlobalResponseNorm (ConvNeXt V2).
            When provided, GRN is applied after the sequence mixer output
            to promote inter-channel feature competition.
    """

    def __init__(
        self,
        sequence_mixer_cfg: LazyConfig,
        sequence_mixer_norm_cfg: LazyConfig,
        mlp_cfg: LazyConfig,
        mlp_norm_cfg: LazyConfig,
        hidden_dim: int,
        layer_scale_init: float = 1e-4,
        drop_path_rate: float = 0.0,
        register_pooling_cfg: LazyConfig | None = None,
        num_registers: int = 0,
        register_start_idx: int = 1,
        grn_cfg: LazyConfig | None = None,
    ):
        """Instantiate norms, sequence mixer, MLP, and optional register pooling."""
        super().__init__()

        self.input_norm = instantiate(sequence_mixer_norm_cfg)
        for param in self.input_norm.parameters():
            param._no_weight_decay = True

        self.sequence_mixer = instantiate(sequence_mixer_cfg)

        self.mlp_norm = instantiate(mlp_norm_cfg)
        for param in self.mlp_norm.parameters():
            param._no_weight_decay = True

        self.mlp = instantiate(mlp_cfg)

        self.ls_attn = LayerScale(hidden_dim, init_value=layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.ls_mlp = LayerScale(hidden_dim, init_value=layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # Optional register-based FiLM conditioning
        self.num_registers = num_registers
        self.register_start_idx = register_start_idx
        if register_pooling_cfg is not None and num_registers > 0:
            self.register_pooling = instantiate(register_pooling_cfg)
        else:
            self.register_pooling = None

        # Optional GRN (Global Response Normalization) after mixer
        self.grn = instantiate(grn_cfg) if grn_cfg is not None else None

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        """Count FLOPs for one ViT-5 residual block.

        Architecture per branch:
          x = x + DropPath(LayerScale(Mixer(Norm(x))))
          x = x + DropPath(LayerScale(MLP(Norm(x))))

        Let T = num_tokens.

        FLOPs breakdown:
          1. input_norm (RMSNorm):   ``self.input_norm.flop_count(T)``
          2. register_pooling:       ``self.register_pooling.flop_count(D)``
             Only when FiLM conditioning is enabled (register_pooling is not None).
             D is inferred from ``self.input_norm.weight.shape[0]``.
          3. sequence_mixer:         ``self.sequence_mixer.flop_count(T, inference)``
             Dispatches to ViT5Attention or ViT5HyenaAdapter depending on config.
          4. ls_attn (LayerScale):   ``self.ls_attn.flop_count(T)``
             Skipped when LayerScale is replaced by Identity (init_value=0).
          5. drop_path:              0  (stochastic identity)
          6. mlp_norm (RMSNorm):     ``self.mlp_norm.flop_count(T)``
          7. mlp:                    ``self.mlp.flop_count(T)``
          8. ls_mlp (LayerScale):    ``self.ls_mlp.flop_count(T)``

        Args:
            num_tokens: Sequence length T.
            inference: Passed through to the sequence mixer.

        Returns:
            Total FLOPs as an integer.
        """
        flops = 0

        # input_norm
        flops += self.input_norm.flop_count(num_tokens)

        # Register pooling (FiLM conditioning)
        if self.register_pooling is not None:
            D = self.input_norm.weight.shape[0]
            flops += self.register_pooling.flop_count(D)

        # Sequence mixer
        flops += self.sequence_mixer.flop_count(num_tokens, inference=inference)

        # LayerScale (attention branch)
        if isinstance(self.ls_attn, LayerScale):
            flops += self.ls_attn.flop_count(num_tokens)

        # mlp_norm
        flops += self.mlp_norm.flop_count(num_tokens)

        # MLP
        flops += self.mlp.flop_count(num_tokens)

        # LayerScale (MLP branch)
        if isinstance(self.ls_mlp, LayerScale):
            flops += self.ls_mlp.flop_count(num_tokens)

        return flops

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] where T = num_tokens (cls + patches + registers).
            condition: Unused, kept for API compatibility with ResidualBlock.

        Returns:
            [B, T, C]
        """
        x_normed = self.input_norm(x)

        mixer_kwargs = {}
        if self.register_pooling is not None:
            s = self.register_start_idx
            regs = x_normed[:, s : s + self.num_registers, :]  # [B, num_registers, C]
            mixer_kwargs["conditioning"] = self.register_pooling(regs)  # [B, C]

        mixer_out = self.sequence_mixer(x_normed, **mixer_kwargs)
        if self.grn is not None:
            mixer_out = self.grn(mixer_out)
        x = x + self.drop_path(self.ls_attn(mixer_out))
        x = x + self.drop_path(self.ls_mlp(self.mlp(self.mlp_norm(x))))
        return x
