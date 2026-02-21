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

    Args:
        sequence_mixer_cfg: LazyConfig for the sequence mixer (QKVSequenceMixer wrapping Attention).
        sequence_mixer_norm_cfg: LazyConfig for the pre-norm before the sequence mixer.
        mlp_cfg: LazyConfig for the MLP.
        mlp_norm_cfg: LazyConfig for the pre-norm before the MLP.
        hidden_dim: Channel dimension (needed for LayerScale).
        layer_scale_init: Initial value for LayerScale gammas. Set to 0 to disable LayerScale.
        drop_path_rate: Stochastic depth drop probability.
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
    ):
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

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] where T = num_tokens (cls + patches + registers).
            condition: Unused, kept for API compatibility with ResidualBlock.

        Returns:
            [B, T, C]
        """
        x = x + self.drop_path(self.ls_attn(self.sequence_mixer(self.input_norm(x))))
        x = x + self.drop_path(self.ls_mlp(self.mlp(self.mlp_norm(x))))
        return x
