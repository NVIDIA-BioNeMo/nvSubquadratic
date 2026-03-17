"""ViT-5 Classification Network.

Implements the full ViT-5 architecture for ImageNet classification:
- Patch embedding via Conv2d (Patchify)
- Learnable absolute positional embeddings (APE) on patch tokens
- Prepended CLS token, appended register tokens
- N x ViT5ResidualBlock (pre-norm, attention, LayerScale, DropPath, MLP)
- Final norm on CLS token -> linear head

Reference: Wang et al., "ViT-5: Vision Transformers for The Mid-2020s", 2026.
"""

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
        dropout_rate: Dropout rate applied to the CLS token before head.
        use_cls_token: If True (default), prepend a learnable CLS token and read it
            out for classification. If False, skip the CLS token and use global
            average pooling over patch tokens instead.
        prepend_registers: If True, register tokens are placed between the CLS token
            and patch tokens ([CLS, regs, patches]) instead of after patches
            ([CLS, patches, regs]). This allows the full sequence to be reshaped to a
            contiguous 2D grid for spatial mixers like Hyena. Only takes effect when
            both use_cls_token and num_registers > 0.
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
        dropout_rate: float = 0.0,
        use_cls_token: bool = True,
        prepend_registers: bool = False,
    ):
        """Initialize ViT5ClassificationNet."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_cls_token = use_cls_token
        self.prepend_registers = prepend_registers

        num_patches_h = image_size // patch_size
        num_patches_w = image_size // patch_size
        self.num_patches = num_patches_h * num_patches_w

        # Patch embedding (non-overlapping Conv2d)
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Learnable tokens
        if use_cls_token:
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

        # Transformer blocks
        self.blocks = nn.ModuleList([instantiate(block_cfg) for _ in range(num_blocks)])

        # Output norm and head
        self.out_norm = instantiate(norm_cfg)
        for param in self.out_norm.parameters():
            param._no_weight_decay = True

        self.out_proj = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.reg_token is not None:
            nn.init.trunc_normal_(self.reg_token, std=0.02)

        # Initialize patch embed and head with truncated normal
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

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
            else:
                x = torch.cat([x, reg_tokens], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        if self.use_cls_token:
            out = x[:, 0]
        else:
            # Global average pool over patch tokens, excluding register tokens
            if self.prepend_registers and self.num_registers > 0:
                out = x[:, self.num_registers :].mean(dim=1)
            elif self.num_registers > 0:
                out = x[:, : -self.num_registers].mean(dim=1)
            else:
                out = x.mean(dim=1)

        out = self.out_norm(out)
        out = self.dropout(out)
        logits = self.out_proj(out)

        return {"logits": logits}
