from math import pi
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from torch import nn


def broadcat(tensors, dim=-1):
    """Broadcast-concatenate a sequence of tensors along *dim* (ported from VARC)."""
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))  # noqa: C414, C417
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]  # noqa: RUF015
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))  # noqa: C417
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (  # noqa: C417
        "invalid dimensions for broadcastable concatentation"
    )
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))  # noqa: C417
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))  # noqa: C417
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))  # noqa: C417
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))  # noqa: C417
    return torch.cat(tensors, dim=-1)


def rotate_half(x):
    """Rotate the last dimension of *x* by 90° in the complex plane (used by RoPE)."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    """2-D Rotary Positional Embedding (RoPE) for vision transformers (ported from VARC)."""

    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        no_rope=0,
    ):
        """Initialize 2-D RoPE frequency buffers."""
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        self.no_rope = no_rope

    def forward(self, t):
        """Apply 2-D RoPE rotation to the last ``dim - no_rope`` channels of *t*."""
        ret = t[:, :, self.no_rope :] * self.freqs_cos + rotate_half(t[:, :, self.no_rope :]) * self.freqs_sin
        if self.no_rope == 0:
            return ret
        return torch.cat((t[:, :, : self.no_rope], ret), dim=2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with 2-D RoPE positional encoding."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 1,
    ) -> None:
        """Initialize MHSA with QKV projection, RoPE, and output projection."""
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require the head dimension to be even")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2
        # Use only max_seq_len items excluding the task tokens for RoPE lengths
        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=max_seq_len,  # Intentionally pass actual H/W which we reconstruct later
            no_rope=no_rope,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute self-attention with RoPE and optional key-padding mask."""
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA expects True = attend; key_padding_mask has True = ignore → invert
            attn_mask = ~key_padding_mask[:, None, None, :].to(dtype=torch.bool)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context


class ARCTransformerEncoderLayer(nn.Module):
    """Single post-norm transformer encoder layer (matches VARC architecture)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
    ) -> None:
        """Initialize encoder layer with MHSA, MLP, and LayerNorm components."""
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply post-norm MHSA and MLP sublayers."""
        # Post-norm architecture (matches VARC original)
        residual = x
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = residual + self.dropout3(x)
        x = self.norm2(x)

        return x


class ARCTransformerEncoder(nn.Module):
    """Stack of post-norm transformer encoder layers with final LayerNorm."""

    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
    ) -> None:
        """Initialize encoder with *depth* stacked layers and a final norm."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ARCTransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    max_seq_len=max_seq_len,
                    no_rope=no_rope,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pass *x* through all encoder layers and the final norm."""
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class ARCViT(nn.Module):
    """Vision Transformer for ARC-AGI with per-task learnable Task Tokens.

    Replicates the VARC architecture ("ARC is a Vision Problem!"):
    colour embedding → patch embedding → task token prepend → transformer → pixel-shuffled head.
    """

    def __init__(
        self,
        num_tasks: int,
        max_size: int = 32,
        num_colors: int = 12,  # 10 colors + IGNORE(10) + PAD(11)
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_task_tokens: int = 1,
        patch_size: int = 2,
    ) -> None:
        """Initialize ARCViT with embeddings, transformer encoder, and prediction head."""
        super().__init__()

        self.max_size = max_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_task_tokens = num_task_tokens

        grid_size = max_size // patch_size
        self.seq_length = grid_size * grid_size

        from nvsubquadratic.networks.arc_embedding import ARCColorTaskEmbedding

        self.embedding = ARCColorTaskEmbedding(
            num_colors=num_colors, num_tasks=num_tasks, hidden_dim=embed_dim, num_task_tokens=num_task_tokens
        )
        self.patch_embed = PatchEmbed(
            img_size=max_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, bias=True
        )

        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))

        self.encoder = ARCTransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=grid_size,  # Pass grid dimension for RoPE
            no_rope=num_task_tokens,
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(embed_dim, num_colors * patch_size * patch_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_and_condition: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass: embed pixels, prepend task token, encode, and decode to logits."""
        pixel_values = input_and_condition["input"]
        condition = input_and_condition["condition"]
        task_ids = condition["task_id"]
        attention_mask = condition.get("attention_mask", None)

        if pixel_values.dim() != 3:
            raise ValueError("`input` must be (batch, height, width).")

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # Embed colors. Clamp so that padding sentinels (IGNORE_INDEX=10, PAD_INDEX=11)
        # don't cause out-of-bounds embedding lookups when num_colors=10. Their
        # embeddings are irrelevant because attention masks them out.
        x, task_tokens = self.embedding(pixel_values, task_ids)
        # x is currently channels-last; PatchEmbed expects [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # Create patches -> [B, num_patches, embed_dim]
        tokens = self.patch_embed(x)
        tokens = tokens + self.positional_embed[:, : tokens.size(1), :]

        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)

        hidden_states = torch.cat([task_tokens, tokens], dim=1)
        hidden_states = self.dropout(hidden_states)

        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask is [B, H, W]. Downsample roughly by max-pooling patch_size
            h, w = attention_mask.shape[1], attention_mask.shape[2]
            mask_reshaped = attention_mask.reshape(
                batch_size, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size
            )
            mask_patched = torch.max(torch.max(mask_reshaped, dim=2)[0], dim=3)[0]  # [B, h//p, w//p]
            flat_mask = mask_patched.view(batch_size, -1)

            # 1 implies ignore padding in standard PyTorch, but check usage above.
            pad_mask = ~flat_mask.bool()
            pad_mask = torch.cat(
                [torch.zeros(batch_size, self.num_task_tokens, device=device, dtype=torch.bool), pad_mask],
                dim=1,
            )
            key_padding_mask = pad_mask

        encoded = self.encoder(hidden_states, key_padding_mask=key_padding_mask)
        pixel_states = encoded[:, self.num_task_tokens :, :]  # [B, num_patches, embed_dim]

        logits_patched = self.head(pixel_states)  # [B, num_patches, num_colors * p * p]

        grid_dim = self.max_size // self.patch_size
        # Unflatten to [B, num_colors, H, W]
        # logits_patched is [B, grid_dim * grid_dim, num_colors * patch_size * patch_size]
        logits = logits_patched.reshape(-1, grid_dim, grid_dim, self.patch_size, self.patch_size, self.num_colors)
        logits = logits.permute((0, 1, 3, 2, 4, 5))  # [B, grid_dim, patch_size, grid_dim, patch_size, num_colors]
        logits = logits.reshape(batch_size, self.max_size, self.max_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)  # [B, num_colors, H, W]

        return {"logits": logits}
