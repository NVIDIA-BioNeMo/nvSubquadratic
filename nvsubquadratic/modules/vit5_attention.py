"""ViT-5 Attention: Multi-head self-attention with RMSNorm QK-Norm and register-aware 2D/3D RoPE.

Key differences from the base Attention module:
- QK-Norm uses RMSNorm (learnable, per-head) instead of L2 normalization.
- RoPE is applied only to patch tokens (not cls token).
- Register tokens get their own high-frequency RoPE.
- Operates on flattened sequences [B, T, C] where T = 1 (cls) + N (patches) + R (registers).
- Supports 2D RoPE (default) and 3D RoPE (pass num_patches_d).

Optimizations vs naive implementation:
- RoPE cos/sin are precomputed as registered buffers (CUDA-graph safe, no graph breaks).
- RoPE applied via a single broadcast multiply on [B, T, H, D] — no reshape to (B*H, T, D).
- SDPA backend auto-selected by PyTorch (CuDNN preferred on H100).
- No redundant dtype casts around SDPA (autocast handles precision).
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsubquadratic.lazy_config import LazyConfig, instantiate


def _build_2d_rope_flat(
    height: int,
    width: int,
    head_dim: int,
    rope_base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute flattened 2D RoPE cos/sin for a (height x width) grid.

    The original codebase (``nvsubquadratic.utils.rope``) computed RoPE on the
    fly during every forward pass.  We precompute the cos/sin tables at init
    time and store them as ``register_buffer`` (persistent=False) for two
    reasons:

    1. **CUDA-graph safety** — dynamic tensor creation inside forward() causes
       graph-capture failures with ``torch.compile(mode="max-autotune")``.
       Buffers are allocated once and reused across replays.
    2. **Automatic device/dtype movement** — ``register_buffer`` ensures the
       tables follow the module to the correct device/dtype via ``.to()``
       without manual bookkeeping.

    Channel layout (matches ``_rotate_half_per_axis``):
        [Y_half | X_half], each of size head_dim/2.
        Within each half, frequencies are ``repeat_interleave(2)`` so that the
        paired-swap rotation in ``_rotate_half_per_axis`` operates on matching
        frequency pairs.

    Returns:
        (cos, sin) each of shape [height * width, head_dim].
    """
    dim_half = head_dim // 2
    theta = 1.0 / (rope_base ** (torch.arange(0, dim_half, 2).float() / dim_half))

    pos_y = torch.arange(height).float()
    pos_x = torch.arange(width).float()

    angles_y = pos_y[:, None] * theta[None, :]  # [H, dim_half//2]
    angles_x = pos_x[:, None] * theta[None, :]  # [W, dim_half//2]

    angles_y = angles_y.repeat_interleave(2, dim=-1)  # [H, dim_half]
    angles_x = angles_x.repeat_interleave(2, dim=-1)  # [W, dim_half]

    # Broadcast to [H, W, dim_half] each, then cat to [H, W, head_dim]
    angles_2d = torch.cat(
        [
            angles_y[:, None, :].expand(height, width, dim_half),
            angles_x[None, :, :].expand(height, width, dim_half),
        ],
        dim=-1,
    )

    flat = angles_2d.reshape(height * width, head_dim)
    return flat.cos(), flat.sin()


def _build_3d_rope_flat(
    depth: int,
    height: int,
    width: int,
    head_dim: int,
    rope_base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute flattened 3D RoPE cos/sin for a (depth x height x width) grid.

    Channel layout: [Z_third | Y_third | X_third], each of size head_dim/3.
    Within each third, frequencies are ``repeat_interleave(2)`` to match the
    split-half rotation in ``_rotate_half_3d``.

    Requires head_dim % 6 == 0 so each third is even.

    Returns:
        (cos, sin) each of shape [depth * height * width, head_dim].
    """
    dim_third = head_dim // 3
    theta = 1.0 / (rope_base ** (torch.arange(0, dim_third, 2).float() / dim_third))

    pos_z = torch.arange(depth).float()
    pos_y = torch.arange(height).float()
    pos_x = torch.arange(width).float()

    angles_z = (pos_z[:, None] * theta[None, :]).repeat_interleave(2, dim=-1)  # [D, dim_third]
    angles_y = (pos_y[:, None] * theta[None, :]).repeat_interleave(2, dim=-1)  # [H, dim_third]
    angles_x = (pos_x[:, None] * theta[None, :]).repeat_interleave(2, dim=-1)  # [W, dim_third]

    angles_3d = torch.cat(
        [
            angles_z[:, None, None, :].expand(depth, height, width, dim_third),
            angles_y[None, :, None, :].expand(depth, height, width, dim_third),
            angles_x[None, None, :, :].expand(depth, height, width, dim_third),
        ],
        dim=-1,
    )

    flat = angles_3d.reshape(depth * height * width, head_dim)
    return flat.cos(), flat.sin()


def _rotate_half_per_axis(x: torch.Tensor) -> torch.Tensor:
    """Split-half rotation applied independently to Y and X channel halves.

    **Why not the standard interleaved ``rotate_half``?**
    The original training run used ``rotate_half_blh`` from
    ``nvsubquadratic.utils.rope``, which splits each axis-half at the midpoint
    and swaps with negation ([-x2, x1]).  The commonly-seen interleaved
    rotation ([-x1, x0, -x3, x2, ...]) is numerically *incompatible* with
    checkpoints trained under the split-half convention — using it causes a
    ~4 pp accuracy drop at validation.  This function preserves exact numerical
    parity with the original rotation so existing checkpoints remain valid.

    Channel layout: [Y_half | X_half], each half of size D/2.
    Within each half: split at D/4 and swap with negation.
    """
    d = x.shape[-1]
    d_half = d // 2
    d_quarter = d // 4
    x_y1 = x[..., :d_quarter]
    x_y2 = x[..., d_quarter:d_half]
    x_x1 = x[..., d_half : d_half + d_quarter]
    x_x2 = x[..., d_half + d_quarter :]
    return torch.cat([-x_y2, x_y1, -x_x2, x_x1], dim=-1)


def _rotate_half_3d(x: torch.Tensor) -> torch.Tensor:
    """Split-third rotation applied independently to Z, Y, and X channel thirds.

    Channel layout: [Z_third | Y_third | X_third], each of size D/3.
    Within each third: split at D/6 and swap with negation ([-x2, x1]).
    Numerically consistent with ``rotate_half_blh`` from ``nvsubquadratic.utils.rope``.
    """
    d = x.shape[-1]
    d_third = d // 3
    d_sixth = d // 6
    x_z1 = x[..., :d_sixth]
    x_z2 = x[..., d_sixth:d_third]
    x_y1 = x[..., d_third : d_third + d_sixth]
    x_y2 = x[..., d_third + d_sixth : 2 * d_third]
    x_x1 = x[..., 2 * d_third : 2 * d_third + d_sixth]
    x_x2 = x[..., 2 * d_third + d_sixth :]
    return torch.cat([-x_z2, x_z1, -x_y2, x_y1, -x_x2, x_x1], dim=-1)


class ViT5Attention(nn.Module):
    """ViT-5 multi-head self-attention with RMSNorm QK-Norm and register-aware RoPE.

    Expects input as [B, T, C] where T = num_patches + (1 if has_cls) + num_registers.
    Token layout: [patches, (CLS), registers] -- no padding (stripped by the network).
    The module includes its own QKV and output projections (unlike the base Attention
    which is wrapped in QKVSequenceMixer).

    Supports 2D RoPE (default) and 3D RoPE (set num_patches_d).
    - 2D requires head_dim % 4 == 0.
    - 3D requires head_dim % 6 == 0.

    Token layout and RoPE buffer ordering depends on ``use_cls_token`` and
    ``prepend_registers``, which must match the ViT5GeneralPurposeNet config:

    - ``prepend_registers=False`` (default, ImageNet-style):
        [CLS (opt)] + [patches] + [registers]
        Patch tokens get spatial RoPE; CLS and registers get identity or high-freq RoPE.
    - ``prepend_registers=True`` (3D PDE-style, with zero-pad):
        [registers] + [zero_pad] + [patches]
        Registers and zero-pad tokens get identity RoPE (cos=1, sin=0); patches get
        spatial RoPE. Pass ``num_zero_pad`` to account for the alignment padding added
        by ViT5GeneralPurposeNet (= spatial_slice_size - num_registers).

    Args:
        hidden_dim: Total hidden dimension.
        num_heads: Number of attention heads.
        num_patches_h: Height of the patch grid.
        num_patches_w: Width of the patch grid.
        num_patches_d: Depth of the patch grid. When set, switches to 3D RoPE.
        num_registers: Number of register tokens.
        use_cls_token: Whether a CLS token is prepended (only used when prepend_registers=False).
        prepend_registers: Whether registers are prepended (True) or appended (False).
        num_zero_pad: Number of zero-pad tokens inserted after registers (only when
            prepend_registers=True). Must equal spatial_slice_size - num_registers.
        qk_norm: LazyConfig for the QK normalization layer, or None to disable.
        rope_base: Base frequency for patch RoPE.
        reg_rope_base: Base frequency for register RoPE (used when prepend_registers=False).
        attn_dropout: Attention dropout rate.
        proj_dropout: Output projection dropout rate.
        qkv_bias: Whether to use bias in QKV projection.
        out_proj_bias: Whether to use bias in the output projection.
        scale: Attention scaling factor. When None, defaults to ``head_dim ** -0.5``.
        init_fn_qkv_proj: Optional callable ``fn(tensor) -> None`` applied to the
            QKV projection weights. When None, weights keep PyTorch's default init.
        has_cls: Whether the sequence contains a CLS token (between patches and
            registers in the token layout). Defaults to True.
        init_fn_out_proj: Optional callable ``fn(tensor) -> None`` applied to the
            output projection weights. When None, weights keep PyTorch's default init.
    """

    def __init__(  # noqa: D107
        self,
        hidden_dim: int,
        num_heads: int,
        num_patches_h: int,
        num_patches_w: int,
        num_patches_d: Optional[int] = None,
        num_registers: int = 4,
        use_cls_token: bool = True,
        prepend_registers: bool = False,
        num_zero_pad: int = 0,
        qk_norm: Optional[LazyConfig] = None,
        rope_base: float = 10000.0,
        reg_rope_base: float = 100.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = False,
        out_proj_bias: bool = False,
        scale: Optional[float] = None,
        init_fn_qkv_proj: Optional[Callable[[torch.Tensor], None]] = None,
        init_fn_out_proj: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = scale if scale is not None else self.head_dim**-0.5
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.num_patches_d = num_patches_d
        self.data_dim = 3 if num_patches_d is not None else 2
        self.num_registers = num_registers
        self.use_cls_token = use_cls_token
        self.prepend_registers = prepend_registers
        self.num_zero_pad = num_zero_pad
        self.attn_dropout = attn_dropout

        if self.data_dim == 3:
            assert self.head_dim % 6 == 0, f"3D RoPE requires head_dim % 6 == 0, got head_dim={self.head_dim}."
        else:
            assert self.head_dim % 4 == 0, f"2D RoPE requires head_dim % 4 == 0, got head_dim={self.head_dim}."

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=out_proj_bias)
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        if init_fn_qkv_proj is not None:
            init_fn_qkv_proj(self.qkv.weight)
            if self.qkv.bias is not None:
                nn.init.zeros_(self.qkv.bias)
        if init_fn_out_proj is not None:
            init_fn_out_proj(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

        if qk_norm is not None:
            self.q_norm = instantiate(qk_norm)
            self.k_norm = instantiate(qk_norm)
            self.qk_norm = True
        else:
            self.qk_norm = False

        self.rope_base = rope_base
        self.reg_rope_base = reg_rope_base

        # ── Precomputed RoPE cos/sin buffers ──────────────────────────────
        #
        # Stored as non-persistent registered buffers so they:
        #   - survive .to(device) / .half() without manual bookkeeping,
        #   - are visible to CUDA-graph capture (no dynamic allocation in forward),
        #   - are NOT serialised into checkpoints (persistent=False), since they
        #     are deterministically reconstructed from __init__ args.
        #
        # Buffer layout must match the actual token order produced by ViT5GeneralPurposeNet.
        #
        # prepend_registers=False (ImageNet-style): [CLS?] + [patches] + [registers]
        #   - CLS gets identity (cos=1, sin=0).
        #   - Patches get spatial RoPE.
        #   - Registers get high-frequency RoPE (reg_rope_base).
        #
        # prepend_registers=True (3D PDE-style):   [registers] + [zero_pad] + [patches]
        #   - Registers and zero-pad get identity (cos=1, sin=0) — non-spatial tokens.
        #   - Patches get spatial RoPE.
        #   - num_zero_pad must equal spatial_slice_size - num_registers.

        # Build patch RoPE
        if self.data_dim == 3:
            patch_cos, patch_sin = _build_3d_rope_flat(
                num_patches_d, num_patches_h, num_patches_w, self.head_dim, rope_base
            )
        else:
            patch_cos, patch_sin = _build_2d_rope_flat(num_patches_h, num_patches_w, self.head_dim, rope_base)

        if prepend_registers:
            # Layout: [registers (identity)] + [zero_pad (identity)] + [patches (spatial)]
            n_prefix = num_registers + num_zero_pad
            parts_cos = [torch.ones(n_prefix, self.head_dim), patch_cos]
            parts_sin = [torch.zeros(n_prefix, self.head_dim), patch_sin]
        else:
            # Layout: [CLS (identity, optional)] + [patches (spatial)] + [registers (high-freq)]
            parts_cos = []
            parts_sin = []
            if use_cls_token:
                parts_cos.append(torch.ones(1, self.head_dim))
                parts_sin.append(torch.zeros(1, self.head_dim))
            parts_cos.append(patch_cos)
            parts_sin.append(patch_sin)
            if num_registers > 0:
                if self.data_dim == 3:
                    reg_d = reg_h = reg_w = max(1, round(num_registers ** (1 / 3)))
                    reg_cos, reg_sin = _build_3d_rope_flat(reg_d, reg_h, reg_w, self.head_dim, reg_rope_base)
                else:
                    reg_h = reg_w = max(1, int(num_registers**0.5))
                    reg_cos, reg_sin = _build_2d_rope_flat(reg_h, reg_w, self.head_dim, reg_rope_base)
                parts_cos.append(reg_cos)
                parts_sin.append(reg_sin)

        self.register_buffer("rope_cos", torch.cat(parts_cos, dim=0), persistent=False)
        self.register_buffer("rope_sin", torch.cat(parts_sin, dim=0), persistent=False)

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        """Count FLOPs for multi-head self-attention on ``num_tokens`` tokens.

        The ``inference`` flag is accepted for API consistency but does not
        change the count — attention has no cacheable precomputation analogous
        to SIREN kernels.

        Let T = num_tokens, D = ``self.hidden_dim``.

        FLOPs breakdown:
          1. QKV projection (Linear(D, 3D)):       6 * T * D²
             Three projections packed into one:  2 * T * D * 3D.
          2. QK-Norm (2x RMSNorm on Q and K):      Delegated to self.q_norm / self.k_norm.
             Only counted when ``self.qk_norm`` is True; 0 otherwise.
          3. RoPE on Q and K:                       4 * T * D
             Each of Q, K: x * cos + rotate(x) * sin = 2 elementwise
             multiplies per element, over T * D elements, for both Q and K.
             This assumes **full RoPE** (all ``head_dim`` dimensions rotated),
             which is the case here: the cos/sin buffers have shape
             ``[T, head_dim]`` and broadcast across all heads.
             For partial RoPE (only the first ``rope_dim`` of each head
             rotated, remainder passed through), the count would instead be
             ``4 * T * num_heads * rope_dim``.
          4. SDPA (Q@K^T + attn@V):                 4 * T² * D
             Q@K^T: 2 * T * T * D.  attn@V: 2 * T * T * D.
             (Softmax cost ~3 * T * H is negligible and omitted.)
          5. Output projection (Linear(D, D)):      2 * T * D²

        Total: 8 * T * D² + 4 * T² * D + 4 * T * D + qk_norm_flops.

        Args:
            num_tokens: Total sequence length T (cls + patches + registers).
            inference: Accepted for API consistency; does not affect the count.

        Returns:
            Total FLOPs as an integer.
        """
        T = num_tokens
        D = self.hidden_dim

        flops = 0
        # QKV projection:  2 * T * D * 3D
        flops += 2 * T * D * 3 * D
        # QK-Norm (delegate to per-head norm instances)
        if self.qk_norm:
            flops += self.q_norm.flop_count(T)
            flops += self.k_norm.flop_count(T)
        # RoPE: 2 elementwise ops on Q (T*D) + 2 on K (T*D)
        flops += 4 * T * D
        # SDPA: Q@K^T + attn@V
        flops += 4 * T * T * D
        # Output projection
        flops += 2 * T * D * D
        return flops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] where T = num_patches + (1 if has_cls) + num_registers.
                Token layout: [patches, (CLS), registers].

        Returns:
            [B, T, C]
        """
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, num_heads, head_dim]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE: x' = x * cos + rotate(x) * sin
        # Buffers are [T, D]; unsqueeze to [1, T, 1, D] for broadcast over B and H.
        cos = self.rope_cos[None, :, None, :]
        sin = self.rope_sin[None, :, None, :]
        _rotate_half = _rotate_half_3d if self.data_dim == 3 else _rotate_half_per_axis
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # Transpose for SDPA: [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self) -> str:  # noqa: D102
        patches = (
            f"{self.num_patches_d}x{self.num_patches_h}x{self.num_patches_w}"
            if self.data_dim == 3
            else f"{self.num_patches_h}x{self.num_patches_w}"
        )
        return (
            f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, "
            f"qk_norm={self.qk_norm}, num_registers={self.num_registers}, "
            f"patches=({patches}), data_dim={self.data_dim}, "
            f"prepend_registers={self.prepend_registers}, num_zero_pad={self.num_zero_pad}, "
            f"rope_base={self.rope_base}"
        )
