# TODO: Add license header here

"""JiT utility functions.

Ported from https://github.com/LTH14/JiT.
"""

from math import pi

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn


def broadcat(tensors, dim=-1):
    """Concatenate tensors with broadcasting on non-concatenated dimensions."""
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = next(iter(shape_lens))
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*(list(t.shape) for t in tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(values)) <= 2 for _, values in expandable_dims), (
        "invalid dimensions for broadcastable concatenation"
    )
    max_dims = [(axis, max(values)) for axis, values in expandable_dims]
    expanded_dims = [(axis, (axis_size,) * num_tensors) for axis, axis_size in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*(sizes for _, sizes in expanded_dims)))
    expanded_tensors = [tensor.expand(*shape) for tensor, shape in zip(tensors, expandable_shapes)]
    return torch.cat(expanded_tensors, dim=dim)


def rotate_half(x):
    """Rotate tensor pairs in the last dimension for rotary embeddings."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    """2D rotary embedding helper with configurable frequency construction."""

    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        """Initialize a full 2D rotary embedding table."""
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

        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t, start_index=0):
        """Apply rotary embedding to the selected slice of ``t``."""
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], (
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    """Flattened 2D rotary embedding helper optimized for sequence inputs."""

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
        num_cls_token=0,
    ):
        """Initialize flattened cosine and sine tables for rotary embedding."""
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

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            _N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).contiguous()  # [N_cls+N_img, D]
            self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).contiguous()
        else:
            self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).contiguous()
            self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).contiguous()

    def forward(self, t):
        """Apply precomputed flattened rotary embedding values to ``t``."""
        # Ensure devices match
        if t.device != self.freqs_cos.device:
            self.freqs_cos = self.freqs_cos.to(t.device)
            self.freqs_sin = self.freqs_sin.to(t.device)

        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by JiT blocks."""

    def __init__(self, hidden_size, eps=1e-6):
        """Initialize RMSNorm parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """Normalize ``hidden_states`` with RMS statistics on the last dimension."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Build 2D sine-cosine positional embeddings for a square grid."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Build 2D sine-cosine positional embeddings from a provided grid."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Build 1D sine-cosine positional embeddings from a position array."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
