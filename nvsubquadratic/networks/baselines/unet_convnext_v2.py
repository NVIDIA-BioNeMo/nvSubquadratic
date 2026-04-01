"""UNet-ConvNeXt V2 — fixed skip connections.

Fixes the upstream bug in ``unet_convnext.UNetConvNext`` where ``skips[0]``
(finest-resolution encoder features) is never used in the decoder.  With N
encoder stages the original decoder loop accesses ``skips[-1], ...,
skips[-(N-1)]`` for ``j = 1, ..., N-1``, skipping ``skips[0]`` entirely.

Changes vs :class:`~unet_convnext.UNetConvNext`:

- **All** encoder skips are consumed (``skips[0]`` feeds the last decoder).
- Every decoder stage uses ``skip_project=True`` (the original had
  ``skip_project=False`` for ``j == 0``).

All building blocks (``_Stage``, ``_Block``, ``_LayerNorm``, etc.) are
imported from ``unet_convnext`` to avoid duplication.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from nvsubquadratic.networks.baselines.unet_convnext import (
    _CHANNELS_FIRST_TO_LAST,
    _CHANNELS_LAST_TO_FIRST,
    _Stage,
    conv_modules,
)


class UNetConvNextV2(nn.Module):
    """UNet-ConvNeXt with corrected skip connections.

    Every decoder stage receives the corresponding encoder skip via
    concatenation, including the finest-resolution features that the
    upstream implementation silently drops.

    Args:
        dim_in: Number of input channels.
        dim_out: Number of output channels.
        n_spatial_dims: 2 for images, 3 for volumes.
        spatial_resolution: Unused, kept for API compatibility.
        stages: Number of encoder/decoder stages (default 4).
        blocks_per_stage: ConvNeXt blocks per stage (default 1).
        blocks_at_neck: ConvNeXt blocks at bottleneck (default 1).
        init_features: Feature map width at the first stage (default 32).
        gradient_checkpointing: Use activation checkpointing to save memory.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...] | None = None,
        stages: int = 4,
        blocks_per_stage: int = 1,
        blocks_at_neck: int = 1,
        init_features: int = 32,
        gradient_checkpointing: bool = False,
    ):
        """Build encoder-decoder with corrected skip wiring."""
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        features = init_features
        self.gradient_checkpointing = gradient_checkpointing

        encoder_dims = [features * 2**i for i in range(stages + 1)]
        decoder_dims = [features * 2**i for i in range(stages, -1, -1)]

        encoder = []
        decoder = []
        self.in_proj = conv_modules[n_spatial_dims](dim_in, features, kernel_size=3, padding=1)
        self.out_proj = conv_modules[n_spatial_dims](features, dim_out, kernel_size=3, padding=1)
        for i in range(stages):
            encoder.append(
                _Stage(
                    encoder_dims[i],
                    encoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="down",
                )
            )
            decoder.append(
                _Stage(
                    decoder_dims[i],
                    decoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="up",
                    skip_project=True,
                )
            )
        self.encoder = nn.ModuleList(encoder)
        self.neck = _Stage(
            encoder_dims[-1],
            encoder_dims[-1],
            n_spatial_dims,
            blocks_at_neck,
            mode="neck",
        )
        self.decoder = nn.ModuleList(decoder)

    def _optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all skip connections used.

        Args:
            x: Channels-first input tensor [B, C_in, *spatial].

        Returns:
            Channels-first output tensor [B, C_out, *spatial].
        """
        x = self.in_proj(x)
        skips = []
        for enc in self.encoder:
            skips.append(x)
            x = self._optional_checkpointing(enc, x)
        x = self.neck(x)
        # Consume skips in reverse: skips[-1] (coarsest) first, skips[0] (finest) last.
        for j, dec in enumerate(self.decoder):
            skip_idx = len(skips) - 1 - j
            x = torch.cat([x, skips[skip_idx]], dim=1)
            x = dec(x)
        x = self.out_proj(x)
        return x


class WellUNetConvNextV2(nn.Module):
    """Like :class:`~unet_convnext.WellUNetConvNext` but with fixed skip connections.

    Input:  ``{"input": [B, *spatial, C_in], "condition": None}``
    Output: ``{"logits": [B, *spatial, C_out]}``
    """

    def __init__(self, **kwargs):
        """Initialize by forwarding all kwargs to :class:`UNetConvNextV2`."""
        super().__init__()
        self.net = UNetConvNextV2(**kwargs)
        self._n_spatial_dims = self.net.n_spatial_dims

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transpose to channels-first, run UNet V2, transpose back."""
        x = input_and_condition["input"]
        x = rearrange(x, _CHANNELS_LAST_TO_FIRST[self._n_spatial_dims])
        y = self.net(x)
        y = rearrange(y, _CHANNELS_FIRST_TO_LAST[self._n_spatial_dims])
        return {"logits": y}
