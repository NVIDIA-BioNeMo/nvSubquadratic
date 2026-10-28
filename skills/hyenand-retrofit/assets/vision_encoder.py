# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A small Hyena mixer for dense, row-major 2D patch tokens.

Copy this file into a host project with nvSubquadratic installed. This is a
tensor-only building block, not a universal attention replacement. The host
keeps its residual connection, normalization, MLP and readout. Inputs contain
patches only: CLS/register tokens, padding, packed images, dropped/reordered
patches, attention masks, causal mode and cross-attention require a separate
adapter. No attention weights are returned.

Example::

    mixer = Hyena2DTokenMixer(32, max_grid_hw=(8, 8), film_conditioning="patch_mean")
    patches = torch.randn(2, 3 * 5, 32)
    output = mixer(patches, grid_hw=(3, 5))

Use FP32 model parameters with autocast for mixed precision; blanket half/BF16
module casts also lower coordinate-buffer precision and are not supported here.
The default ``torch_fft`` backend is a reference path, not a speed claim.
Benchmark the intended backend, shapes, precision and training/inference mode.
"""

from typing import Literal

import torch
from torch import nn

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.rms_norm_channel_first import RMSNormChannelFirst
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.utils.qk_norm import L2Norm


class _AutocastCKConv2D(CKConvND):
    """Match the FFT shortcut dtype under autocast without replacing its parameter."""

    def apply_convolution(self, x, conv_kernel, shortcut, is_bhl_input):
        """Cast the local shortcut view so gradients still reach the FP32 parameter."""
        return super().apply_convolution(x, conv_kernel, shortcut.to(dtype=x.dtype), is_bhl_input)


class Hyena2DTokenMixer(nn.Module):
    """Mix ``[B, H*W, C]`` patch tokens and return a tensor of the same shape.

    ``film_conditioning="none"`` uses a shared input-independent kernel; the
    Hyena gates still depend on the input. ``"patch_mean"`` conditions each
    sample's kernel on its own mean patch embedding. This simple pooling choice
    is an explicit architecture option, not a replacement for register pooling.
    FiLM starts as identity modulation, so kernels initially ignore the control
    vector. Its earlier layers can have zero gradients on the first step.

    ``max_grid_hw`` fixes the per-axis coordinate cache and spacing. Every input
    grid must fit these bounds so forward calls never extend the cache. Save
    these bounds with the model configuration and restore the same values for
    trained checkpoints; changing them changes the kernel's coordinates, even
    when the learned parameters are identical. A state_dict does not store this
    constructor configuration.

    All parameters are registered during construction. Build the module and
    optionally copy projections before creating the optimizer or DDP wrapper.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        max_grid_hw: tuple[int, int],
        film_conditioning: Literal["none", "patch_mean"] = "none",
        fft_backend: str = "torch_fft",
        qkv_bias: bool = True,
        out_proj_bias: bool = True,
        kernel_hidden_dim: int = 32,
    ):
        """Construct projections, a 2D Hyena mixer and an optional FiLM generator."""
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("kernel_hidden_dim", kernel_hidden_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(max_grid_hw, (tuple, list)) or len(max_grid_hw) != 2:
            raise ValueError("max_grid_hw must contain exactly (max_height, max_width)")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 2 for n in max_grid_hw):
            raise ValueError("max_grid_hw dimensions must be integers >= 2")
        if film_conditioning not in ("none", "patch_mean"):
            raise ValueError("film_conditioning must be 'none' or 'patch_mean'")
        if fft_backend not in ("torch_fft", "subq_ops", "subq_ops_fused"):
            raise ValueError("fft_backend must be 'torch_fft', 'subq_ops' or 'subq_ops_fused'")

        self.hidden_dim = hidden_dim
        self.max_grid_hw = tuple(max_grid_hw)
        self.film_conditioning = film_conditioning
        use_film = film_conditioning == "patch_mean"
        num_kernel_layers = 3
        film_cfg = None
        if use_film:
            film_cfg = LazyConfig(KernelFiLMGenerator)(
                cond_dim=hidden_dim,
                kernel_hidden_dim=kernel_hidden_dim,
                # Two hidden SIREN layers plus the positional embedding.
                num_film_layers=num_kernel_layers,
                film_hidden_dim=kernel_hidden_dim,
                init_type="identity",
            )

        self.mixer = QKVSequenceMixer(
            hidden_dim=hidden_dim,
            qkv_bias=qkv_bias,
            out_proj_bias=out_proj_bias,
            mixer_cfg=LazyConfig(Hyena)(
                global_conv_cfg=LazyConfig(_AutocastCKConv2D)(
                    data_dim=2,
                    hidden_dim=hidden_dim,
                    kernel_cfg=LazyConfig(SIRENKernelND)(
                        out_dim=hidden_dim,
                        data_dim=2,
                        mlp_hidden_dim=kernel_hidden_dim,
                        num_layers=num_kernel_layers,
                        embedding_dim=kernel_hidden_dim,
                        omega_0=10.0,
                        L_cache=self.max_grid_hw,
                        use_bias=True,
                        hidden_omega_0=1.0,
                        film_cfg=film_cfg,
                        film_after_pos_embed=use_film,
                    ),
                    mask_cfg=LazyConfig(nn.Identity)(),
                    grid_type="double",
                    fft_padding="zero",
                    is_causal=False,
                    fft_backend=fft_backend,
                ),
                short_conv_cfg=LazyConfig(nn.Conv2d)(
                    in_channels=3 * hidden_dim,
                    out_channels=3 * hidden_dim,
                    kernel_size=3,
                    padding=1,
                    groups=3 * hidden_dim,
                    bias=False,
                ),
                gate_nonlinear_cfg=LazyConfig(nn.SiLU)(),
                gate_nonlinear_2_cfg=LazyConfig(nn.Sigmoid)(),
                pixelhyena_norm_cfg=LazyConfig(RMSNormChannelFirst)(dim=hidden_dim, use_quack=False),
                qk_norm_cfg=LazyConfig(L2Norm)(dim=1),
                output_norm_cfg=LazyConfig(RMSNormChannelFirst)(dim=hidden_dim, use_quack=False),
            ),
        )

    def forward(self, x: torch.Tensor, *, grid_hw: tuple[int, int]) -> torch.Tensor:
        """Mix a complete grid; unsupported attention arguments raise TypeError.

        ``grid_hw`` must describe the true row-major patch layout for every
        sample. Length checks reject additional prefix tokens, but cannot detect
        incorrect caller-supplied geometry or a permutation of the patches.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a tensor with shape [B, H*W, C]")
        if not isinstance(grid_hw, (tuple, list)) or len(grid_hw) != 2:
            raise ValueError("grid_hw must contain exactly (height, width)")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in grid_hw):
            raise ValueError("grid_hw dimensions must be positive integers")
        height, width = grid_hw
        if height > self.max_grid_hw[0] or width > self.max_grid_hw[1]:
            raise ValueError(
                f"grid_hw={grid_hw} exceeds max_grid_hw={self.max_grid_hw}; "
                "use a grid within the configured bounds. Larger bounds change SIREN coordinates "
                "and require a separately validated model/checkpoint configuration."
            )
        batch, tokens, channels = x.shape
        if batch < 1 or channels != self.hidden_dim:
            raise ValueError(f"x must have a nonempty batch and C={self.hidden_dim}")
        if tokens != height * width:
            raise ValueError(
                f"Expected {height * width} patch tokens for grid_hw={grid_hw}, got {tokens}; "
                "prefix tokens, padding and packed images are unsupported"
            )
        if not torch.is_floating_point(x):
            raise TypeError("x must contain floating-point patch embeddings")

        grid = x.reshape(batch, height, width, channels)
        if self.film_conditioning == "patch_mean":
            output = self.mixer(grid, conditioning=x.mean(dim=1))
        else:
            output = self.mixer(grid)
        return output.reshape(batch, tokens, channels)

    def copy_projections_from(self, qkv: nn.Linear, out_proj: nn.Linear) -> "Hyena2DTokenMixer":
        """Copy packed Q/K/V and output projection values without replacing parameters.

        The source QKV rows must be ordered Q, K, V. Both dimensions and bias
        presence must match. Dtypes must match to avoid silently rounding saved
        weights; move this module to the source dtype before calling if needed.
        Devices may differ. Both sources are validated before any copy, and the
        source modules are unchanged. This transfers projections only; it does
        not preserve the original attention function or its trained quality.
        """
        pairs = (("qkv", qkv, self.mixer.qkv_proj), ("out_proj", out_proj, self.mixer.out_proj))
        for name, source, target in pairs:
            if not isinstance(source, nn.Linear):
                raise TypeError(f"{name} must be torch.nn.Linear")
            if (source.in_features, source.out_features) != (target.in_features, target.out_features):
                raise ValueError(f"{name} must map {target.in_features} to {target.out_features} features")
            if (source.bias is None) != (target.bias is None):
                raise ValueError(f"{name} bias presence must match the mixer configuration")
            if source.weight.dtype != target.weight.dtype:
                raise ValueError(f"{name} weight dtype must match; convert the mixer dtype first")
            if source.bias is not None and source.bias.dtype != target.bias.dtype:
                raise ValueError(f"{name} bias dtype must match; convert the mixer dtype first")
        with torch.no_grad():
            for _, source, target in pairs:
                target.weight.copy_(source.weight)
                if source.bias is not None:
                    target.bias.copy_(source.bias)
        return self
