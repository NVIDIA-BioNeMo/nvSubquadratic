# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Polymathic AI.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions ported from PolymathicAI/the_well (BSD-3-Clause), itself adapted from
# facebookresearch/ConvNeXt (MIT; Copyright (c) Meta Platforms, Inc. and affiliates).
# See LICENSE/third_party.txt for the full BSD-3-Clause and MIT license texts.

"""UNet-ConvNeXt baseline from The Well benchmark.

Mixed adaptation from:
    Liu et al. 2022, A ConvNet for the 2020s.
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Ronneberger et al., 2015. Convolutional Networks for Biomedical Image Segmentation.

Ported from the_well.benchmark.models.unet_convnext to avoid the neuralop
dependency that the_well.benchmark.models.__init__ eagerly imports.

Note:
    **Known bug (upstream):** ``skips[0]`` (finest-resolution encoder
    features) is never used.  With N encoder stages the decoder loop
    accesses ``skips[-1], skips[-2], ..., skips[-(N-1)]`` for
    ``j = 1, 2, ..., N-1``, skipping ``skips[0]`` entirely.  In a
    standard UNet the finest skip should connect to the last decoder
    stage.  This matches the reference implementation in
    ``the_well.benchmark.models.unet_convnext.UNetConvNext`` (v1.0.1)
    line-for-line, so we preserve it here for reproducibility.
    See :class:`UNetConvNextV2` for a corrected version.

If you use this implementation, please cite the original work above.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from torch.utils.checkpoint import checkpoint


conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

permute_channel_strings = {
    2: [
        "N C H W -> N H W C",
        "N H W C -> N C H W",
    ],
    3: [
        "N C D H W -> N D H W C",
        "N D H W C -> N C D H W",
    ],
}


class _LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first data formats."""

    def __init__(self, normalized_shape, n_spatial_dims, eps=1e-6, data_format="channels_last"):
        super().__init__()
        if data_format == "channels_last":
            padded_shape = (normalized_shape,)
        else:
            padded_shape = (normalized_shape,) + (1,) * n_spatial_dims
        self.weight = nn.Parameter(torch.ones(padded_shape))
        # channels_first mode only uses weight (L2-norm + scale); avoid
        # creating an unused bias that would break DDP strict mode.
        if data_format == "channels_last":
            self.bias = nn.Parameter(torch.zeros(padded_shape))
        self.n_spatial_dims = n_spatial_dims
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            x = F.normalize(x, p=2, dim=1, eps=self.eps) * self.weight
            return x


class _Upsample(nn.Module):
    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            _LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_transpose_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class _Downsample(nn.Module):
    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            _LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class _Block(nn.Module):
    """ConvNeXt Block.

    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv;
        all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear
        -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch.
    """

    def __init__(self, dim, n_spatial_dims, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.dwconv = conv_modules[n_spatial_dims](dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm(dim, n_spatial_dims, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][0])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][1])
        x = input + self.drop_path(x)
        return x


class _Stage(nn.Module):
    """ConvNeXt Stage: sequence of blocks + optional resampling."""

    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        depth=1,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        mode="down",
        skip_project=False,
    ):
        super().__init__()

        if skip_project:
            self.skip_proj = conv_modules[n_spatial_dims](2 * dim_in, dim_in, 1)
        else:
            self.skip_proj = nn.Identity()
        if mode == "down":
            self.resample = _Downsample(dim_in, dim_out, n_spatial_dims)
        elif mode == "up":
            self.resample = _Upsample(dim_in, dim_out, n_spatial_dims)
        else:
            self.resample = nn.Identity()

        self.blocks = nn.ModuleList(
            [_Block(dim_in, n_spatial_dims, drop_path, layer_scale_init_value) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.skip_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.resample(x)
        return x


class UNetConvNext(nn.Module):
    """UNet with ConvNeXt blocks — channels-first (NCHW / NCDHW) interface.

    This is a faithful port of the_well.benchmark.models.unet_convnext.UNetConvNext.
    Input/output are channels-first tensors. For channels-last dict interface see
    :class:`WellUNetConvNext`.

    Args:
        dim_in: Number of input channels.
        dim_out: Number of output channels.
        n_spatial_dims: 2 for images, 3 for volumes.
        spatial_resolution: Tuple of spatial sizes (used only for compat with BaseModel API).
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
        """Build encoder-decoder with the given depth and width."""
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

    def _optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Channels-first input tensor ``[B, C_in, *spatial]``.

        Returns:
            Channels-first output tensor ``[B, C_out, *spatial]``.

        Note:
            **Known bug (upstream):** ``skips[0]`` (finest-resolution encoder
            features) is never used.  With N encoder stages the decoder loop
            accesses ``skips[-1], skips[-2], ..., skips[-(N-1)]`` for
            ``j = 1, 2, ..., N-1``, skipping ``skips[0]`` entirely.  In a
            standard UNet the finest skip should connect to the last decoder
            stage.  This matches the reference implementation in
            ``the_well.benchmark.models.unet_convnext.UNetConvNext`` (v1.0.1)
            line-for-line, so we preserve it here for reproducibility.
            See :class:`UNetConvNextV2` for a corrected version.
        """
        x = self.in_proj(x)
        skips = []
        for enc in self.encoder:
            skips.append(x)
            x = self._optional_checkpointing(enc, x)
        x = self.neck(x)
        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)
            x = dec(x)
        x = self.out_proj(x)
        return x


# ─── Channels-last dict wrapper for the nvSubquadratic training infra ─────────


_CHANNELS_LAST_TO_FIRST = {
    2: "B H W C -> B C H W",
    3: "B D H W C -> B C D H W",
}
_CHANNELS_FIRST_TO_LAST = {
    2: "B C H W -> B H W C",
    3: "B C D H W -> B D H W C",
}


class WellUNetConvNext(nn.Module):
    """UNet-ConvNeXt with the dict-based channels-last interface expected by WELLRegressionWrapper.

    Input:  ``{"input": [B, *spatial, C_in], "condition": None}``
    Output: ``{"logits": [B, *spatial, C_out]}``

    All internal computation is channels-first; this wrapper only transposes at
    the boundary.

    Constructor args are forwarded to :class:`UNetConvNext`.
    """

    def __init__(self, **kwargs):
        """Initialize by forwarding all kwargs to :class:`UNetConvNext`."""
        super().__init__()
        self.net = UNetConvNext(**kwargs)
        self._n_spatial_dims = self.net.n_spatial_dims

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transpose to channels-first, run UNet, transpose back."""
        x = input_and_condition["input"]
        x = rearrange(x, _CHANNELS_LAST_TO_FIRST[self._n_spatial_dims])
        y = self.net(x)
        y = rearrange(y, _CHANNELS_FIRST_TO_LAST[self._n_spatial_dims])
        return {"logits": y}
