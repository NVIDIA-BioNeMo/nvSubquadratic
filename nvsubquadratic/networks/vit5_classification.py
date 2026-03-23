"""ViT-5 Classification Network.

Implements the full ViT-5 architecture for ImageNet classification:
- Patch embedding via Conv2d (Patchify)
- Learnable absolute positional embeddings (APE) on patch tokens
- Prepended CLS token, appended register tokens
- N x ViT5ResidualBlock (pre-norm, attention, LayerScale, DropPath, MLP)
- Final norm on CLS token -> linear head

Reference: Wang et al., "ViT-5: Vision Transformers for The Mid-2020s", 2026.
"""

from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5ClassificationNet(nn.Module):
    """ViT-5 classification network.

    Args:
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output classes.
        hidden_dim: Transformer hidden dimension.
        num_blocks: Number of transformer blocks.
        patch_size: Patch size for patchification.
        image_size: Input image size (assumes square).
        num_registers: Number of learnable register tokens.
        block_cfg: LazyConfig for ViT5ResidualBlock.
        norm_cfg: LazyConfig for the normalization layer (RMSNorm).
            When ``readout="register_concat"``, this norm is applied to the
            concatenated compressed register vector (dim = ``num_registers * neck_dim``),
            so the caller must pass ``dim=num_registers * neck_dim`` in the config.
        dropout_rate: Dropout rate applied before the classification head.
        readout: Classification readout strategy.
            ``"cls"``: prepend a learnable CLS token and read it out.
            ``"gap"``: global average pooling over patch tokens.
            ``"register_concat"``: gather register tokens after all blocks,
            compress each via a shared neck linear, concatenate, and project.
        neck_compression_ratio: Compression ratio for ``register_concat`` readout.
            Each register is projected from ``hidden_dim`` to
            ``hidden_dim // neck_compression_ratio``.  The classification head
            input dimension becomes
            ``num_registers * (hidden_dim // neck_compression_ratio)``.
            Required when ``readout="register_concat"``.
        prepend_registers: If True, register tokens are placed before patch tokens.
            With CLS: [CLS, regs, patches]. Without CLS: [regs, zero_pad, patches]
            where zero_pad fills the register row to grid width (image_size // patch_size)
            for clean 2D reshape in spatial mixers like Hyena.
        reg_init: Initialization strategy for register tokens.
            "trunc_normal" (default) uses truncated normal with std=0.02.
            "zeros" initializes registers to zero.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int,
        num_blocks: int,
        patch_size: int,
        image_size: int,
        num_registers: int,
        block_cfg: LazyConfig,
        norm_cfg: LazyConfig,
        readout: Literal["cls", "gap", "register_concat"],
        dropout_rate: float = 0.0,
        neck_compression_ratio: int | None = None,
        prepend_registers: bool = False,
        reg_init: Literal["trunc_normal", "zeros"] = "trunc_normal",
    ):
        """Initialize ViT-5 classification network."""
        super().__init__()
        self._reg_init = reg_init
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.patch_size = patch_size
        self.image_size = image_size
        self.prepend_registers = prepend_registers

        self.readout = readout
        self.use_cls_token = readout == "cls"

        if self.readout == "register_concat":
            if neck_compression_ratio is None:
                raise ValueError("neck_compression_ratio is required when readout='register_concat'")
            if num_registers == 0:
                raise ValueError("num_registers must be > 0 for register_concat readout")
            if hidden_dim % neck_compression_ratio != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by neck_compression_ratio ({neck_compression_ratio})"
                )

        num_patches_h = image_size // patch_size
        num_patches_w = image_size // patch_size
        self.num_patches = num_patches_h * num_patches_w

        # Patch embedding (non-overlapping Conv2d, no bias — pos_embed absorbs the offset)
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=False,
        )

        # Learnable tokens
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.cls_token._no_weight_decay = True
        else:
            self.cls_token = None

        # Absolute positional embeddings for patch tokens only (not cls, not registers)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.pos_embed._no_weight_decay = True

        if num_registers > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            self.reg_token._no_weight_decay = True
        else:
            self.reg_token = None

        # Zero-padding buffer for GAP models: fills the register row to grid width
        assert num_registers <= num_patches_w, (
            f"num_registers ({num_registers}) > grid width ({num_patches_w}); "
            "registers must fit in a single row for 2D reshape"
        )
        self._register_row_width = num_patches_w
        if num_registers > 0 and prepend_registers and not self.use_cls_token:
            pad_size = num_patches_w - num_registers
            if pad_size > 0:
                self.register_buffer("reg_zero_pad", torch.zeros(1, pad_size, hidden_dim), persistent=False)
            else:
                self.reg_zero_pad = None
        else:
            self.reg_zero_pad = None

        # Transformer blocks
        self.blocks = nn.ModuleList([instantiate(block_cfg) for _ in range(num_blocks)])

        # Register-concat readout: shared neck compression
        if self.readout == "register_concat":
            self.neck_dim = hidden_dim // neck_compression_ratio
            self.register_neck = nn.Linear(hidden_dim, self.neck_dim, bias=False)
            head_dim = num_registers * self.neck_dim
        else:
            self.neck_dim = None
            self.register_neck = None
            head_dim = hidden_dim

        # Output norm and head
        self.out_norm = instantiate(norm_cfg)
        for param in self.out_norm.parameters():
            param._no_weight_decay = True

        self.out_proj = nn.Linear(head_dim, num_classes, bias=False)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.reg_token is not None:
            if self._reg_init == "zeros":
                nn.init.zeros_(self.reg_token)
            else:
                nn.init.trunc_normal_(self.reg_token, std=0.02)

        # Initialize patch embed and head with truncated normal
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        if self.register_neck is not None:
            nn.init.trunc_normal_(self.register_neck.weight, std=0.02)

    def flop_count(self, inference: bool = False) -> int:
        """Count FLOPs for a full ViT-5 classification forward pass (one sample).

        Pipeline:
          1. Patch embedding (Conv2d(in_channels, D, kernel_size=P, stride=P)):
               2 * in_channels * D * P² * num_patches
             Each of the ``num_patches`` output positions has a receptive field
             of P x P x in_channels → P² * in_channels MACs = 2 * P² * in_ch * D FLOPs.
          2. Positional embedding addition:  num_patches * D  (elementwise add).
          3. Transformer blocks:  sum of ``block.flop_count(T, inference)``
             where T is the total token count (see below).
          4. Output norm (RMSNorm on 1 CLS token or GAP result):
               ``self.out_norm.flop_count(1)``
             For GAP models, the averaging itself costs num_patches * D adds.
          5. Classification head (Linear(D, num_classes)):
               2 * D * num_classes

        Token count T:
          - With CLS + prepend_registers: 1 + num_registers + num_patches
          - With CLS + append_registers: 1 + num_patches + num_registers
          - Without CLS + prepend_registers: register_row_width + num_patches
          - Without CLS/registers: num_patches
          The ordering doesn't affect FLOP count — only T matters.

        Args:
            inference: Passed through to each block for kernel caching decisions.

        Returns:
            Total FLOPs as an integer.
        """
        D = self.hidden_dim
        P = self.patch_size
        num_patches = self.num_patches
        in_channels = self.patch_embed.in_channels

        flops = 0

        # 1. Patch embedding: Conv2d(in_ch, D, P, stride=P) on (image_size, image_size)
        flops += 2 * in_channels * D * P * P * num_patches

        # 2. Positional embedding addition
        flops += num_patches * D

        # 3. Total token count
        T = num_patches
        if self.use_cls_token:
            T += 1
        if self.num_registers > 0:
            if self.prepend_registers and not self.use_cls_token:
                T += self._register_row_width
            else:
                T += self.num_registers

        # 4. Transformer blocks
        for block in self.blocks:
            flops += block.flop_count(T, inference=inference)

        # 5. Output norm + readout
        if self.readout == "register_concat":
            # Neck linear: R * (2 * D * neck_dim)
            flops += self.num_registers * 2 * D * self.neck_dim
            head_dim = self.num_registers * self.neck_dim
            flops += self.out_norm.flop_count(1)
        elif not self.use_cls_token:
            flops += num_patches * D  # GAP: mean over patch tokens
            head_dim = D
            flops += self.out_norm.flop_count(1)
        else:
            head_dim = D
            flops += self.out_norm.flop_count(1)

        # 6. Classification head
        flops += 2 * head_dim * self.num_classes

        return flops

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_and_condition: Dict with keys "input" (images [B, H, W, C]) and "condition" (unused).

        Returns:
            Dict with key "logits" of shape [B, num_classes].
        """
        x = input_and_condition["input"]  # [B, H, W, C] channels-last

        # Channels-last -> channels-first for Conv2d
        x = rearrange(x, "b h w c -> b c h w")
        x = self.patch_embed(x)  # [B, hidden_dim, H', W']
        x = rearrange(x, "b c h w -> b (h w) c")  # [B, num_patches, hidden_dim]

        # Add absolute positional embeddings
        x = x + self.pos_embed

        B = x.shape[0]

        # Prepend CLS token (when enabled)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + num_patches, C]

        # Insert register tokens
        if self.reg_token is not None:
            reg_tokens = self.reg_token.expand(B, -1, -1)
            if self.prepend_registers and self.cls_token is not None:
                # [CLS, regs, patches] — enables direct 2D reshape for spatial mixers
                x = torch.cat([x[:, :1, :], reg_tokens, x[:, 1:, :]], dim=1)
            elif self.prepend_registers:
                # [regs, zero_pad, patches] — GAP model without CLS
                if self.reg_zero_pad is not None:
                    pad = self.reg_zero_pad.expand(B, -1, -1)
                    x = torch.cat([reg_tokens, pad, x], dim=1)
                else:
                    x = torch.cat([reg_tokens, x], dim=1)
            else:
                x = torch.cat([x, reg_tokens], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        if self.readout == "register_concat":
            # Gather register tokens, compress via neck, concatenate
            if self.prepend_registers and self.cls_token is not None:
                reg_start = 1  # [CLS, regs, patches]
            elif self.prepend_registers:
                reg_start = 0  # [regs, zero_pad, patches]
            else:
                reg_start = x.shape[1] - self.num_registers  # [..., regs]
            regs = x[:, reg_start : reg_start + self.num_registers, :]
            out = self.register_neck(regs).flatten(start_dim=1)  # [B, R * neck_dim]
        elif self.use_cls_token:
            out = x[:, 0]
        else:
            # Global average pool over patch tokens, excluding register tokens
            if self.prepend_registers and self.num_registers > 0 and self.cls_token is not None:
                # [CLS, regs, patches] — skip CLS + registers
                out = x[:, 1 + self.num_registers :].mean(dim=1)
            elif self.prepend_registers and self.num_registers > 0:
                # [regs, zero_pad, patches] — skip entire register row
                out = x[:, self._register_row_width :].mean(dim=1)
            elif self.num_registers > 0:
                out = x[:, : -self.num_registers].mean(dim=1)
            else:
                out = x.mean(dim=1)

        out = self.out_norm(out)
        out = self.dropout(out)
        logits = self.out_proj(out)

        return {"logits": logits}
