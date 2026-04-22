"""ViT-5 Classification Network.

Implements the full ViT-5 architecture for ImageNet classification:
- Patch embedding via Conv2d (Patchify)
- Learnable absolute positional embeddings (APE) on patch tokens
- Token layout: [patches, CLS, registers, padding]
- N x ViT5ResidualBlock (pre-norm, attention, LayerScale, DropPath, MLP)
- Final norm on CLS token -> linear head

Token ordering convention:
  [patches (H*W), CLS (1), registers (R), padding (P)]
where padding has length ``(-num_non_pad) % grid_w`` to make the total
sequence length divisible by ``grid_w`` for 2D spatial mixers (Hyena).
Attention blocks receive the sequence with padding stripped; Hyena blocks
receive the full padded sequence.

Supports **hybrid** architectures with interleaved block types via
``layer_pattern`` + ``layer_types`` (e.g. ``"HA" * 6`` for alternating
Hyena/Attention).

Reference: Wang et al., "ViT-5: Vision Transformers for The Mid-2020s", 2026.
"""

from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


def _compute_drop_path_rates(max_rate: float, num_blocks: int, schedule: str) -> list[float]:
    """Compute per-layer drop path rates according to the given schedule.

    Args:
        max_rate: Maximum stochastic depth drop probability.
        num_blocks: Total number of blocks.
        schedule: ``"constant"`` (same rate for all layers) or ``"linear"``
            (ramp from 0 to ``max_rate`` across depth).

    Returns:
        List of per-layer drop path rates (length ``num_blocks``).
    """
    if schedule == "constant":
        return [max_rate] * num_blocks
    elif schedule == "linear":
        if num_blocks <= 1:
            return [max_rate]
        return [max_rate * i / (num_blocks - 1) for i in range(num_blocks)]
    else:
        raise ValueError(f"Unknown drop_path_schedule: {schedule!r}. Expected 'constant' or 'linear'.")


class ViT5ClassificationNet(nn.Module):
    """ViT-5 classification network.

    Token layout: ``[patches (H*W), CLS (1), registers (R), padding (P)]``.

    Padding makes ``T % grid_w == 0`` for 2D spatial mixers (Hyena).
    Attention blocks receive the sequence with padding stripped; Hyena blocks
    receive the full padded sequence.

    Supports two block-stacking modes (mutually exclusive):

    1. **Homogeneous** (default): a single ``block_cfg`` is replicated
       ``num_blocks`` times.

    2. **Hybrid / interleaved**: ``layer_pattern`` + ``layer_types`` define
       per-layer block types.  For example, ``layer_pattern="HA" * 6`` with
       ``layer_types={"H": hyena_cfg, "A": attn_cfg}`` creates 12 blocks
       alternating between Hyena and Attention.

    Args:
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output classes.
        hidden_dim: Transformer hidden dimension.
        num_blocks: Number of transformer blocks.
        patch_size: Patch size for patchification.
        image_size: Input image size (assumes square).
        num_registers: Number of learnable register tokens.
        block_cfg: LazyConfig for ViT5ResidualBlock (homogeneous mode).
            Mutually exclusive with ``layer_pattern``/``layer_types``.
        norm_cfg: LazyConfig for the normalization layer (RMSNorm).
        dropout_rate: Dropout rate applied before the classification head.
        readout: Classification readout strategy.
            ``"cls"``: append a learnable CLS token after patches and read it out.
            ``"gap"``: global average pooling over patch tokens.
            ``"register_concat"``: gather register tokens after all blocks,
            compress each via a shared neck linear, concatenate, and project.
        neck_compression_ratio: Compression ratio for ``register_concat`` readout.
            Required when ``readout="register_concat"``.
        reg_init: Initialization strategy for register tokens.
            ``"trunc_normal"`` (default) or ``"zeros"``.
        layer_pattern: Pattern string defining per-layer block types (hybrid
            mode).  Each character maps to a key in ``layer_types``.
            Length must equal ``num_blocks``.  Example: ``"HA" * 6``.
        layer_types: Dict mapping pattern characters to block LazyConfigs.
            Required when ``layer_pattern`` is set.
        padding_types: Set of ``layer_pattern`` characters whose blocks need
            the full padded sequence (e.g. Hyena).  Blocks whose character is
            NOT in this set receive the sequence with padding stripped.
            Only relevant when ``layer_pattern`` is used *and* ``pad_size > 0``.
            Default: ``{"H"}``.
        max_drop_path_rate: Maximum stochastic depth drop probability.
            Per-layer rates are computed according to ``drop_path_schedule``
            and injected into each block config at construction time.
        drop_path_schedule: How drop path rates are distributed across depth.
            ``"constant"``: every layer gets ``max_drop_path_rate``.
            ``"linear"``: ramp from 0 to ``max_drop_path_rate`` across depth.
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
        norm_cfg: LazyConfig,
        readout: Literal["cls", "gap", "register_concat"],
        block_cfg: LazyConfig | None = None,
        dropout_rate: float = 0.0,
        neck_compression_ratio: int | None = None,
        reg_init: Literal["trunc_normal", "zeros"] = "trunc_normal",
        layer_pattern: str | None = None,
        layer_types: dict[str, LazyConfig] | None = None,
        padding_types: set[str] | None = None,
        max_drop_path_rate: float = 0.0,
        drop_path_schedule: Literal["constant", "linear"] = "constant",
    ):
        """Initialize ViT-5 classification network."""
        super().__init__()
        self._reg_init = reg_init
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.patch_size = patch_size
        self.image_size = image_size
        self.layer_pattern = layer_pattern

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
        self._grid_w = num_patches_w

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

        # Absolute positional embeddings for patch tokens only (not CLS, not registers)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.pos_embed._no_weight_decay = True

        if num_registers > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            self.reg_token._no_weight_decay = True
        else:
            self.reg_token = None

        # Token layout: [patches (H*W), CLS (1?), registers (R), padding (P)]
        # Padding makes T divisible by grid_w for 2D spatial mixers.
        num_non_pad = self.num_patches + (1 if self.use_cls_token else 0) + num_registers
        pad_size = (-num_non_pad) % num_patches_w
        self._pad_size = pad_size
        self._num_non_pad = num_non_pad

        if pad_size > 0:
            self.register_buffer("_zero_pad", torch.zeros(1, pad_size, hidden_dim), persistent=False)
        else:
            self._zero_pad = None

        # Per-block padding flag: True = block needs the full padded sequence
        if padding_types is None:
            padding_types = {"H"}
        if layer_pattern is not None:
            self._block_needs_padding = [ch in padding_types for ch in layer_pattern]
        else:
            self._block_needs_padding = [False] * num_blocks

        # Build per-layer block configs: either from layer_pattern or by
        # replicating a single block_cfg.  Both paths produce the same
        # list[LazyConfig] of length num_blocks fed to a single loop.
        if layer_pattern is not None:
            assert layer_types is not None, "layer_types is required when layer_pattern is set"
            assert len(layer_pattern) == num_blocks, (
                f"layer_pattern length ({len(layer_pattern)}) != num_blocks ({num_blocks})"
            )
            assert block_cfg is None, "block_cfg and layer_pattern are mutually exclusive"
            for ch in layer_pattern:
                assert ch in layer_types, f"Unknown layer type '{ch}' in layer_pattern; available: {set(layer_types)}"
            block_cfgs = [layer_types[ch] for ch in layer_pattern]
        else:
            assert block_cfg is not None, "Either block_cfg or (layer_pattern + layer_types) must be provided"
            block_cfgs = [block_cfg] * num_blocks

        # Auto-compute register_start_idx for the new layout:
        # [patches, CLS, registers, ...] -> registers start after patches + CLS
        register_start_idx = self.num_patches + (1 if self.use_cls_token else 0)

        drop_path_rates = _compute_drop_path_rates(max_drop_path_rate, num_blocks, drop_path_schedule)

        self.blocks = nn.ModuleList(
            [
                instantiate(cfg, drop_path_rate=drop_path_rates[i], register_start_idx=register_start_idx)
                for i, cfg in enumerate(block_cfgs)
            ]
        )

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
          1. Patch embedding (Conv2d):  2 * in_ch * D * P^2 * num_patches
          2. Positional embedding add:  num_patches * D
          3. Transformer blocks:        sum of block.flop_count(T)
          4. Output norm:               self.out_norm.flop_count(1)
          5. Classification head:       2 * head_dim * num_classes

        Token count T = num_non_pad + pad_size.  Attention blocks see
        T_attn = num_non_pad (padding stripped).

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

        # 1. Patch embedding
        flops += 2 * in_channels * D * P * P * num_patches

        # 2. Positional embedding addition
        flops += num_patches * D

        # 3. Transformer blocks (attention blocks see fewer tokens)
        T_full = self._num_non_pad + self._pad_size
        T_no_pad = self._num_non_pad
        for i, block in enumerate(self.blocks):
            T_block = T_full if self._block_needs_padding[i] else T_no_pad
            flops += block.flop_count(T_block, inference=inference)

        # 4. Output norm + readout
        if self.readout == "register_concat":
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

        # 5. Classification head
        flops += 2 * head_dim * self.num_classes

        return flops

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Token layout: ``[patches (H*W), CLS (1?), registers (R), padding (P)]``.
        Attention blocks see ``[patches, CLS, registers]`` (padding stripped).
        Hyena blocks see the full padded sequence.

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

        # Add absolute positional embeddings (patches only)
        x = x + self.pos_embed

        B = x.shape[0]

        # Build token sequence: [patches, CLS?, registers?, padding?]
        parts = [x]
        if self.cls_token is not None:
            parts.append(self.cls_token.expand(B, -1, -1))
        if self.reg_token is not None:
            parts.append(self.reg_token.expand(B, -1, -1))
        if self._zero_pad is not None:
            parts.append(self._zero_pad.expand(B, -1, -1))
        if len(parts) > 1:
            x = torch.cat(parts, dim=1)

        # Apply transformer blocks
        pad_size = self._pad_size
        for i, block in enumerate(self.blocks):
            if pad_size > 0 and not self._block_needs_padding[i]:
                x_no_pad = x[:, :-pad_size]
                x_no_pad = block(x_no_pad)
                x = torch.cat([x_no_pad, x[:, -pad_size:]], dim=1)
            else:
                x = block(x)

        # Readout (indices based on [patches, CLS, registers, padding] layout)
        if self.readout == "register_concat":
            reg_start = self.num_patches + (1 if self.use_cls_token else 0)
            regs = x[:, reg_start : reg_start + self.num_registers, :]
            out = self.register_neck(regs).flatten(start_dim=1)  # [B, R * neck_dim]
        elif self.use_cls_token:
            out = x[:, self.num_patches]  # CLS is right after patches
        else:
            out = x[:, : self.num_patches].mean(dim=1)  # GAP over patches only

        out = self.out_norm(out)
        out = self.dropout(out)
        logits = self.out_proj(out)

        return {"logits": logits}
