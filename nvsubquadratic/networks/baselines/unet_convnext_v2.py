"""UNet-ConvNeXt V2 — fixed skip connections.

Fixes the upstream bug in ``unet_convnext.UNetConvNext`` where ``skips[0]``
(finest-resolution encoder features) is never used in the decoder.  With N
encoder stages the original decoder loop accesses ``skips[-1], ...,
skips[-(N-1)]`` for ``j = 1, ..., N-1``, skipping ``skips[0]`` entirely.

Changes vs :class:`~unet_convnext.UNetConvNext`:

- **All** encoder skips are consumed.  The decoder loop is kept identical
  (first stage upsamples without a skip), but after the last decoder stage
  ``skips[0]`` is concatenated and projected via a 1x1 conv before
  ``out_proj``.  This is the minimal change that recovers the missing
  finest-resolution information.

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

    Fixes the upstream bug where ``skips[0]`` (finest-resolution encoder
    features) is never consumed.  The decoder loop is identical to the
    original — first stage upsamples without a skip, stages 1..N-1 consume
    ``skips[-1], ..., skips[-(N-1)]`` — but after all decoder stages we
    concatenate ``skips[0]`` (full-resolution) and project before ``out_proj``.

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
                    skip_project=i != 0,
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

        # V2 addition: project the concatenated finest-resolution skip
        # (2 * features → features) so skips[0] is actually used.
        self.final_skip_proj = conv_modules[n_spatial_dims](2 * features, features, kernel_size=1)

    def _optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all skip connections used.

        The decoder loop is identical to the original (j=0 upsamples without
        skip, j=1..N-1 consume skips[-1]..skips[-(N-1)]).  The V2 fix adds
        a final concatenation with skips[0] at the original resolution.

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
        # Same decoder loop as original: first stage has no skip, rest consume
        # skips[-1], skips[-2], ... skips[-(N-1)].
        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)
            x = dec(x)
        # V2 fix: consume skips[0] (finest resolution) which the original drops.
        x = torch.cat([x, skips[0]], dim=1)
        x = self.final_skip_proj(x)
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
