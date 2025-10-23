# TODO: Add license header here


"""CKConv-based residual network tailored for diffusion noise prediction."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND


class DiffusionBlock(nn.Module):
    """Single residual block composed of a CKConv branch and an MLP branch."""

    def __init__(
        self,
        hidden_dim: int,
        data_dim: int,
        kernel_cfg: LazyConfig,
        mask_cfg: LazyConfig,
        grid_type: str,
        dropout: float,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        for param in self.norm1.parameters():
            param._no_wd = True
        # CKConvND implements the long-range Hyena-style convolutional kernel.
        self.conv = CKConvND(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            kernel_cfg=kernel_cfg,
            mask_cfg=mask_cfg,
            grid_type=grid_type,
        )
        self.activation = nn.SiLU()
        self.dropout_conv = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        for param in self.norm2.parameters():
            param._no_wd = True
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        # Standard MLP branch (SiLU -> linear) follows the convolutional residual.
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        if condition is not None:
            h = h + condition
        h = self.conv(h)
        h = self.activation(h)
        h = self.dropout_conv(h)
        x = residual + h

        residual = x
        h = self.norm2(x)
        h = self.ff(h)
        h = self.dropout_ff(h)
        return residual + h


class DiffusionResNet(nn.Module):
    """CKConv-based residual network tailored for diffusion noise prediction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        num_blocks: int,
        kernel_cfg: LazyConfig,
        mask_cfg: LazyConfig,
        grid_type: str,
        dropout_in: float,
        block_dropout: float,
        mlp_ratio: float = 2.0,
        data_dim: int = 2,
        positional_encoding_cfg: Optional[LazyConfig] = None,
        condition_proj_cfg: Optional[LazyConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.data_dim = data_dim

        # Initial linear lifts RGB (or channels-last) inputs into the model width.
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.in_dropout = nn.Dropout(dropout_in)

        self.positional_encoding = (
            instantiate(positional_encoding_cfg) if positional_encoding_cfg is not None else None
        )
        condition_proj = (
            instantiate(condition_proj_cfg) if condition_proj_cfg is not None else None
        )
        self.condition_proj = condition_proj if not isinstance(condition_proj, nn.Identity) else None

        # Stack identical diffusion blocks, each mixing spatial and conditional information.
        self.blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    hidden_dim=hidden_dim,
                    data_dim=data_dim,
                    kernel_cfg=kernel_cfg,
                    mask_cfg=mask_cfg,
                    grid_type=grid_type,
                    dropout=block_dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        for param in self.out_norm.parameters():
            param._no_wd = True
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def _expand_condition(self, condition: Optional[torch.Tensor], x: torch.Tensor) -> Optional[torch.Tensor]:
        if condition is None:
            return None
        if self.condition_proj is not None:
            condition = self.condition_proj(condition)
        if condition.ndim == 2:
            # Broadcast timestep embeddings along spatial dimensions for every pixel.
            condition = condition.view(condition.shape[0], *([1] * self.data_dim), condition.shape[-1])
        return condition.expand_as(x)

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = input_and_condition["input"]
        condition = input_and_condition.get("condition")

        x = self.in_dropout(x)
        x = self.in_proj(x)

        if self.positional_encoding is not None:
            x = x + self.positional_encoding(x)

        broadcast_condition = self._expand_condition(condition, x)
        for block in self.blocks:
            x = block(x, broadcast_condition)

        x = self.out_norm(x)
        x = self.out_proj(x)
        return {"prediction": x}
