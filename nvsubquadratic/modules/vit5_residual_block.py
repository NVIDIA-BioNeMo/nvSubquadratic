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
        register_start_idx: Start index of register tokens in the sequence. Default 1
            assumes [CLS, regs, patches] layout. Set to 0 for [regs, patches] (no CLS).
        distribute_registers: If True, registers are evenly interleaved among patches
            and extracted via precomputed indices instead of a contiguous slice.
        num_patches: Number of patch tokens. Required when distribute_registers=True,
            used to compute distributed register indices.
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
        distribute_registers: bool = False,
        num_patches: int = 0,
        grn_cfg: LazyConfig | None = None,
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

        # Optional register-based FiLM conditioning
        self.num_registers = num_registers
        self.register_start_idx = register_start_idx
        self.distribute_registers = distribute_registers
        if register_pooling_cfg is not None and num_registers > 0:
            self.register_pooling = instantiate(register_pooling_cfg)
        else:
            self.register_pooling = None

        # Precompute distributed register indices for FiLM extraction
        if distribute_registers and num_registers > 0:
            assert num_patches > 0, "num_patches required when distribute_registers=True"
            stride = num_patches // num_registers
            register_indices = torch.tensor(
                [stride * (i + 1) + i for i in range(num_registers)], dtype=torch.long
            )
            self.register_buffer("register_indices", register_indices)
        else:
            self.register_indices = None

        # Optional GRN (Global Response Normalization) after mixer
        self.grn = instantiate(grn_cfg) if grn_cfg is not None else None

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
            if self.register_indices is not None:
                # Distributed registers: gather from precomputed indices
                # Use gather instead of advanced indexing for torch.compile compatibility
                B_r, _, C_r = x_normed.shape
                reg_idx = self.register_indices.unsqueeze(0).unsqueeze(-1).expand(B_r, -1, C_r)
                regs = torch.gather(x_normed, 1, reg_idx)  # [B, num_registers, C]
            else:
                # Contiguous registers: slice from start index
                s = self.register_start_idx
                regs = x_normed[:, s : s + self.num_registers, :]  # [B, num_registers, C]
            mixer_kwargs["conditioning"] = self.register_pooling(regs)  # [B, C]

        mixer_out = self.sequence_mixer(x_normed, **mixer_kwargs)
        if self.grn is not None:
            mixer_out = self.grn(mixer_out)
        x = x + self.drop_path(self.ls_attn(mixer_out))
        x = x + self.drop_path(self.ls_mlp(self.mlp(self.mlp_norm(x))))
        return x
