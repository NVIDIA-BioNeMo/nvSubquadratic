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
    ):
        """Initialize ViT5ResidualBlock."""
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
        if register_pooling_cfg is not None and num_registers > 0:
            self.register_pooling = instantiate(register_pooling_cfg)
        else:
            self.register_pooling = None

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
            # Registers at positions [1, 1+num_registers) in [CLS, regs, patches] layout
            regs = x_normed[:, 1 : 1 + self.num_registers, :]  # [B, num_registers, C]
            mixer_kwargs["conditioning"] = self.register_pooling(regs)  # [B, C]

        x = x + self.drop_path(self.ls_attn(self.sequence_mixer(x_normed, **mixer_kwargs)))
        x = x + self.drop_path(self.ls_mlp(self.mlp(self.mlp_norm(x))))
        return x
