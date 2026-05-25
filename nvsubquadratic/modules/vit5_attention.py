"""ViT-5 Attention: Multi-head self-attention with RMSNorm QK-Norm and register-aware 2D RoPE.

This module implements the specialised self-attention block used throughout the
ViT-5 family of hierarchical vision transformers.  It is a *self-contained*
alternative to the generic :class:`~nvsubquadratic.modules.attention.Attention`
module and is **not** interchangeable with it through the
:class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer` dispatch layer.
The key ViT-5-specific design choices are described below.

**Structural differences vs.** :class:`~nvsubquadratic.modules.attention.Attention`

1. **Self-contained projections** — :class:`ViT5Attention` owns its own QKV
   projection (``nn.Linear(C, 3C)``) and output projection (``nn.Linear(C, C)``),
   plus an optional output dropout.  The generic :class:`Attention` is a *pure*
   attention kernel that receives pre-projected Q, K, V tensors from the
   surrounding :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`.
   :class:`ViT5Attention` is therefore consumed directly by
   :class:`~nvsubquadratic.modules.vit5_residual_block.ViT5ResidualBlock` without
   any outer projection wrapper.

2. **RMSNorm QK-Norm (per-head, learnable)** — When ``qk_norm`` is provided,
   :class:`ViT5Attention` instantiates two independent norm modules (one for Q,
   one for K) via :func:`~nvsubquadratic.lazy_config.instantiate`.  The generic
   :class:`Attention` uses a *shared* L2 (cosine) normalisation function
   (:func:`~nvsubquadratic.utils.qk_norm.apply_qk_norm`) without learnable
   parameters.  The ViT-5 norm is applied after the per-head reshape so that
   each head's Q and K are normalised independently.

3. **Register-aware, dual-base 2D RoPE** — Patch tokens receive 2D RoPE with
   base frequency ``rope_base`` (default 10000).  Register tokens receive
   their own 2D RoPE with a *different* base ``reg_rope_base`` (default 100,
   i.e., much higher frequency), reflecting their role as global context carriers
   rather than spatially localised patch representations.  The CLS token, when
   present, receives an identity rotation (cos=1, sin=0).  The generic
   :class:`Attention` applies a single shared RoPE base to all tokens uniformly.

4. **Fixed token layout** — The input sequence ``[B, T, C]`` must follow the
   strict ViT-5 token layout ``[patches, (CLS,) registers]`` where
   ``T = H*W + (1 if has_cls else 0) + R``.  There is no support for arbitrary
   spatial shapes or causal masking.  The generic :class:`Attention` accepts
   1D/2D/3D channels-last tensors and supports both causal and non-causal modes.

5. **CUDA-graph-safe precomputed RoPE buffers** — Both patch and register
   cos/sin tables are concatenated into a single ``[T, head_dim]`` buffer pair
   (``rope_cos``, ``rope_sin``) stored as non-persistent ``register_buffer``
   entries, making the forward pass free of dynamic tensor creation and safe
   for ``torch.compile(mode="max-autotune")`` and CUDA-graph capture.

6. **RoPE rotation convention** — :func:`_rotate_half_per_axis` uses the
   *split-half* convention from ``nvsubquadratic.utils.rope.rotate_half_blh``
   rather than the interleaved rotation used in some external codebases.
   Existing checkpoints trained with this convention will produce ~4 pp accuracy
   drops if the rotation function is swapped.  See :func:`_rotate_half_per_axis`
   for the full explanation.

**Cross-references**

* :class:`~nvsubquadratic.modules.attention.Attention` — generic 1D/2D/3D
  scaled dot-product attention, used via ``QKVSequenceMixer``.
* :class:`~nvsubquadratic.modules.vit5_residual_block.ViT5ResidualBlock` —
  consumes :class:`ViT5Attention` directly as its ``sequence_mixer``.
* ``_build_2d_rope_flat`` — constructs the flattened 2D RoPE tables (module-private).
* ``_rotate_half_per_axis`` — split-half rotation operator used in ``forward`` (module-private).
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
    r"""Precompute flattened 2D RoPE cos/sin for a (height x width) grid.

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

    **Frequency schedule**

    For each spatial axis, frequencies follow the standard RoPE schedule:

    .. math::

        \theta_j = \text{rope\_base}^{-2j / (d_k/2)},
        \quad j = 0, 1, \ldots, d_k/4 - 1

    where :math:`d_k` = ``head_dim``.  Angles for position :math:`p` are
    :math:`\phi_{p,j} = p \cdot \theta_j`, then ``repeat_interleave(2)``
    doubles each frequency so the paired-swap rotation in
    :func:`_rotate_half_per_axis` operates on matching frequency pairs.

    **Channel layout** (matches :func:`_rotate_half_per_axis`):
    ``[Y_half | X_half]``, each of size ``head_dim / 2``.
    Within each half, frequencies are ``repeat_interleave(2)`` so that the
    paired-swap rotation in :func:`_rotate_half_per_axis` operates on matching
    frequency pairs.

    Note:
        ``head_dim`` must be divisible by 2 so that ``dim_half = head_dim // 2``
        is an integer and ``torch.arange(0, dim_half, 2)`` produces a valid
        frequency vector.  The stricter divisibility-by-4 requirement belongs
        to ``_rotate_half_per_axis``, which needs the quarter-split.

    Args:
        height: Number of patch rows ``H`` in the 2D grid.
        width: Number of patch columns ``W`` in the 2D grid.
        head_dim: Per-head channel dimension ``d_k``.  Must be divisible by 2.
        rope_base: Base frequency for the geometric frequency schedule.
            Typical values: ``10000.0`` for patch tokens, ``100.0`` for
            register tokens (lower base → higher frequency → denser angular spacing).

    Returns:
        A tuple ``(cos, sin)`` where each tensor has shape
        ``[height * width, head_dim]``.  These are meant to be stored as
        non-persistent ``register_buffer`` entries and concatenated with the
        CLS and register entries in :class:`ViT5Attention.__init__`.

        Note: this function is also called for the register-token RoPE grid with
        ``height = reg_rope_h``, ``width = reg_rope_w``.  If ``num_registers`` is
        not a perfect square, ``reg_rope_h * reg_rope_w < num_registers`` and the
        returned table will have fewer rows than expected, causing a shape mismatch
        in the subsequent ``torch.cat``.  The caller is responsible for ensuring
        ``height * width`` equals the intended token count.
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


def _rotate_half_per_axis(x: torch.Tensor) -> torch.Tensor:
    r"""Split-half rotation applied independently to Y and X channel halves.

    This implements the rotation component of the RoPE formula
    ``x' = x * cos + rotate(x) * sin``.  The rotation maps each channel vector
    to its 90-degree rotated counterpart using a *split-half* convention:

    .. math::

        \text{rotate}(x) = \text{Concat}\bigl(
            -x_{y,2},\, x_{y,1},\, -x_{x,2},\, x_{x,1}
        \bigr)

    where :math:`x_{y,1}` and :math:`x_{y,2}` are the first and second
    quarters of the head-dim vector (the Y-axis half, each of size
    ``head_dim / 4``), and similarly :math:`x_{x,1}`, :math:`x_{x,2}` for
    the X-axis half.

    **Why not the standard interleaved** ``rotate_half``?

    The original training run used ``rotate_half_blh`` from
    ``nvsubquadratic.utils.rope``, which splits each axis-half at the midpoint
    and swaps with negation (``[-x2, x1]``).  The commonly-seen interleaved
    rotation (``[-x1, x0, -x3, x2, ...]``) is numerically *incompatible* with
    checkpoints trained under the split-half convention — using it causes a
    ~4 pp accuracy drop at validation.  This function preserves exact numerical
    parity with the original rotation so existing checkpoints remain valid.

    **Channel layout**: ``[Y_half | X_half]``, each half of size ``D/2``.
    Within each half: split at ``D/4`` and swap with negation.

    Args:
        x: Query or key tensor of shape ``[B, T, H, D]`` (before the SDPA
            transpose) or any shape whose last dimension is the head dimension
            ``D``.  ``D`` must be divisible by 4 to allow the quarter-split.

    Returns:
        torch.Tensor: Rotated tensor of the same shape as ``x``, representing
        the 90-degree rotation of each frequency pair within the Y and X
        channel halves.

    Note:
        The ``D % 4 == 0`` constraint is **not** checked at runtime.  Passing a
        ``head_dim`` that is not divisible by 4 will cause silent integer
        truncation in the quarter-splits, producing incorrect rotations.  The
        caller (:class:`ViT5Attention`) is responsible for ensuring this holds.
    """
    d = x.shape[-1]
    d_half = d // 2
    d_quarter = d // 4
    x_y1 = x[..., :d_quarter]
    x_y2 = x[..., d_quarter:d_half]
    x_x1 = x[..., d_half : d_half + d_quarter]
    x_x2 = x[..., d_half + d_quarter :]
    return torch.cat([-x_y2, x_y1, -x_x2, x_x1], dim=-1)


class ViT5Attention(nn.Module):
    r"""ViT-5 multi-head self-attention with RMSNorm QK-Norm and register-aware RoPE.

    This module is the primary sequence-mixing operator for the ViT-5 family of
    hierarchical vision transformers.  It computes standard scaled dot-product
    attention:

    .. math::

        \text{head}_i = \text{softmax}\!\left(
            \frac{Q_i K_i^\top}{\sqrt{d_k}}
        \right) V_i, \quad
        \text{out} = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O

    where :math:`d_k = C / H` is the per-head dimension and :math:`W_O` is the
    output projection.  Q, K, V are obtained from a single fused linear
    projection :math:`[Q, K, V] = x W_{QKV}`.

    **Token layout**

    Input shape: ``[B, T, C]`` where
    ``T = num_patches_h * num_patches_w + (1 if has_cls else 0) + num_registers``.
    Token ordering within the sequence axis:

    .. code-block:: text

        [ patch_0, patch_1, ..., patch_{H*W-1}, (CLS,) reg_0, ..., reg_{R-1} ]
          <----- H*W patch tokens --------->   <--1-->  <---- R registers ---->

    This ordering must be consistent with the token layout produced by the
    network's patchify + register-injection layers (see
    :class:`~nvsubquadratic.networks.vit5_classification.ViT5Classifier`).

    **Positional encoding**

    Three distinct positional encodings are applied:

    * **Patch tokens** — 2D RoPE with base frequency ``rope_base`` (default
      10000).  The H×W grid is linearised in row-major (Y-then-X) order.
    * **CLS token** — identity rotation: cos=1, sin=0.  No positional bias is
      imposed on the class token.
    * **Register tokens** — 2D RoPE with base ``reg_rope_base`` (default 100),
      treating the ``R`` registers as a ``sqrt(R) × sqrt(R)`` grid.  A lower base
      value (``reg_rope_base=100`` vs ``rope_base=10000``) yields higher rotation
      frequencies (theta decays more slowly across head-dim pairs), giving denser
      angular spacing for register positions.  This reflects their role as global
      context carriers without fixed spatial meaning.

    All three tables are concatenated into a single buffer pair (``rope_cos``,
    ``rope_sin``) of shape ``[T, head_dim]`` and applied with a single broadcast
    multiply in :meth:`forward`.

    **QK normalisation**

    When ``qk_norm`` is provided, two independent norm modules (``q_norm``,
    ``k_norm``) are instantiated and applied to Q and K after ``qkv.unbind()``
    produces tensors of shape ``[B, T, H, d_k]``, and *before* RoPE.  The norm
    is expected to be a learnable RMSNorm or equivalent (accepting input of shape
    ``[B, T, H, d_k]`` and normalising along the last axis).  Unlike the generic
    :class:`~nvsubquadratic.modules.attention.Attention` module which uses a fixed
    L2 (cosine) normalisation, the learnable per-head norm here allows the model
    to control the scale of the dot products.

    Note:
        Norm is applied **before** RoPE in this module (order: ``unbind →
        q_norm/k_norm → rope → SDPA``), whereas the generic :class:`Attention`
        applies RoPE before L2-norm.  The order matters for checkpoint
        compatibility — swapping the two will change the effective positional
        encoding applied to normalised queries and keys.

    **Differences vs.** :class:`~nvsubquadratic.modules.attention.Attention`

    * Self-contained QKV + output projections (generic uses outer
      ``QKVSequenceMixer``).
    * RMSNorm QK-Norm instead of L2 normalisation.
    * Dual-base register-aware RoPE instead of single-base uniform RoPE.
    * Fixed ``[B, T, C]`` input — no multi-dimensional spatial support, no
      causal masking, no context-parallelism guard.

    Attributes:
        hidden_dim (int): Total channel dimension ``C``.
        num_heads (int): Number of attention heads ``H``.
        head_dim (int): Per-head dimension ``d_k = C / H``.
        scale (float): Attention logit scale, default ``head_dim ** -0.5``.
        num_patches_h (int): Height of the patch grid used for 2D RoPE.
        num_patches_w (int): Width of the patch grid used for 2D RoPE.
        num_registers (int): Number of register tokens ``R``.
        has_cls (bool): Whether the token sequence includes a CLS token
            between the patch tokens and the register tokens.
        attn_dropout (float): Dropout probability applied to attention weights
            during training; set to 0.0 at inference.
        qkv (nn.Linear): Fused QKV projection: ``Linear(C, 3C, bias=qkv_bias)``.
        proj (nn.Linear): Output projection: ``Linear(C, C, bias=out_proj_bias)``.
        proj_drop (nn.Dropout | nn.Identity): Dropout on the projected output.
        q_norm (nn.Module): Per-head query normaliser.  Present only when
            ``qk_norm`` is provided (i.e. ``self.qk_norm is True``).
        k_norm (nn.Module): Per-head key normaliser.  Present only when
            ``qk_norm`` is provided.
        qk_norm (bool): Flag indicating whether QK normalisation is active.
        rope_base (float): Base frequency for patch-token RoPE.
        reg_rope_base (float): Base frequency for register-token RoPE.
        reg_rope_h (int): Height dimension of the register RoPE grid
            (``int(num_registers ** 0.5)``).
        reg_rope_w (int): Width dimension of the register RoPE grid
            (``int(num_registers ** 0.5)``).
        rope_cos (torch.Tensor): Non-persistent buffer of shape ``[T, head_dim]``
            containing the concatenated patch + CLS + register cosine tables.
        rope_sin (torch.Tensor): Non-persistent buffer of shape ``[T, head_dim]``
            containing the concatenated patch + CLS + register sine tables.

    Args:
        hidden_dim: Total hidden dimension ``C``. Must be divisible by
            ``num_heads``.
        num_heads: Number of attention heads ``H``.
        num_patches_h: Height of the patch grid (number of patch rows).
            Used to build the patch 2D RoPE table.
        num_patches_w: Width of the patch grid (number of patch columns).
            Used to build the patch 2D RoPE table.
        num_registers: Number of register tokens ``R`` appended after the
            (optional) CLS token.  Should be a perfect square when > 0 so
            that the register RoPE grid is exactly ``sqrt(R) × sqrt(R)``.
            If ``R`` is not a perfect square, ``reg_rope_h = reg_rope_w =
            int(R**0.5)`` silently truncates, producing only
            ``reg_rope_h * reg_rope_w < R`` RoPE rows and causing a
            ``torch.cat`` shape mismatch at init time.  Defaults to ``4``.
        has_cls: If ``True``, the token sequence contains one CLS token
            immediately after the patch tokens.  The CLS token receives
            identity RoPE (cos=1, sin=0).  Defaults to ``True``.
        qk_norm: :class:`~nvsubquadratic.lazy_config.LazyConfig` for the
            per-head QK normalisation module (e.g. ``RMSNorm(head_dim)``).
            When ``None``, QK normalisation is disabled.  Defaults to ``None``.
        rope_base: Base frequency :math:`\\theta_0` for the patch RoPE
            frequency schedule.  Defaults to ``10000.0``.
        reg_rope_base: Base frequency for the register-token RoPE schedule.
            A lower base (higher frequency) gives denser angular spacing.
            Defaults to ``100.0``.
        attn_dropout: Dropout rate on attention weights, applied only during
            training (``module.training is True``).  Defaults to ``0.0``.
        proj_dropout: Dropout rate on the output projection.  When ``0.0``,
            ``proj_drop`` is an ``nn.Identity``.  Defaults to ``0.0``.
        qkv_bias: Whether to include a bias term in the fused QKV projection.
            Defaults to ``False``.
        out_proj_bias: Whether to include a bias term in the output projection.
            Defaults to ``False``.
        scale: Explicit attention logit scale.  When ``None``, the scale
            defaults to ``head_dim ** -0.5``.  Defaults to ``None``.
        init_fn_qkv_proj: Optional callable ``fn(weight: Tensor) -> None``
            applied to ``self.qkv.weight`` after construction.  The bias, if
            present, is zero-initialised.  When ``None``, PyTorch's default
            Xavier uniform initialisation is used.  Defaults to ``None``.
        init_fn_out_proj: Optional callable ``fn(weight: Tensor) -> None``
            applied to ``self.proj.weight`` after construction.  The bias, if
            present, is zero-initialised.  When ``None``, PyTorch's default
            initialisation is used.  Defaults to ``None``.

    Raises:
        AssertionError: If ``hidden_dim % num_heads != 0``.

    Example::

        import torch
        from nvsubquadratic.modules.vit5_attention import ViT5Attention

        # 2D patch grid of 14x14 with 4 register tokens and 1 CLS token, no QK norm
        attn = ViT5Attention(
            hidden_dim=384,
            num_heads=6,
            num_patches_h=14,
            num_patches_w=14,
            num_registers=4,
            has_cls=True,
        )
        T = 14 * 14 + 1 + 4  # patches + CLS + registers = 201
        x = torch.randn(2, T, 384)  # [B, T, C]
        out = attn(x)               # [B, T, C]
        assert out.shape == x.shape
        # To enable QK-norm, pass a LazyConfig targeting any norm module
        # that accepts [B, T, H, d_k] tensors and normalises along the last axis.
    """

    def __init__(  # noqa: D107
        self,
        hidden_dim: int,
        num_heads: int,
        num_patches_h: int,
        num_patches_w: int,
        num_registers: int = 4,
        has_cls: bool = True,
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
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = scale if scale is not None else self.head_dim**-0.5
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.num_registers = num_registers
        self.attn_dropout = attn_dropout

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
        self.reg_rope_h = int(num_registers**0.5)
        self.reg_rope_w = int(num_registers**0.5)

        # ── Precomputed RoPE cos/sin buffers ──────────────────────────────
        #
        # Stored as non-persistent registered buffers so they:
        #   - survive .to(device) / .half() without manual bookkeeping,
        #   - are visible to CUDA-graph capture (no dynamic allocation in forward),
        #   - are NOT serialised into checkpoints (persistent=False), since they
        #     are deterministically reconstructed from __init__ args.
        #
        # Token layout: [patches, (CLS), registers]
        # Layout per token position: [Y_frequencies | X_frequencies]
        # CLS token (when present) gets cos=1, sin=0 (identity — no positional bias).
        # Register tokens get their own high-frequency RoPE (theta=100).
        self.has_cls = has_cls

        patch_cos, patch_sin = _build_2d_rope_flat(
            num_patches_h,
            num_patches_w,
            self.head_dim,
            rope_base,
        )

        parts_cos = [patch_cos]
        parts_sin = [patch_sin]

        if has_cls:
            parts_cos.append(torch.ones(1, self.head_dim))  # CLS: cos=1 (no rotation)
            parts_sin.append(torch.zeros(1, self.head_dim))  # CLS: sin=0 (no rotation)

        if num_registers > 0:
            reg_cos, reg_sin = _build_2d_rope_flat(
                self.reg_rope_h,
                self.reg_rope_w,
                self.head_dim,
                reg_rope_base,
            )
            parts_cos.append(reg_cos)
            parts_sin.append(reg_sin)

        # [T, head_dim] where T = H*W + (1 if has_cls) + R
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
             Each norm module must expose ``flop_count(num_tokens: int) -> int``
             returning the cost for a sequence of ``num_tokens`` tokens across
             all heads (i.e. for the full ``[B, T, H, d_k]`` shaped input).
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

        Note:
            ``num_tokens`` should equal
            ``num_patches_h * num_patches_w + (1 if has_cls else 0) + num_registers``
            to match the actual sequence length seen during :meth:`forward`.
            Passing a different value will give a proportionally scaled estimate.

        Args:
            num_tokens: Total sequence length T
                (cls + patches + registers).  Should equal
                ``num_patches_h * num_patches_w + (1 if has_cls else 0) + num_registers``.
            inference: Accepted for API consistency with other sequence-mixer
                modules (e.g. Hyena); does not affect the FLOP count.

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
        """Apply ViT-5 multi-head self-attention to a token sequence.

        Executes the following pipeline:

        1. **QKV projection** — ``x W_{QKV}`` reshaped to
           ``[B, T, 3, H, d_k]``, then split into Q, K, V each of shape
           ``[B, T, H, d_k]``.
        2. **(Optional) QK normalisation** — ``q_norm(Q)`` and ``k_norm(K)``
           applied independently along the last (head-dim) axis.
        3. **RoPE** — ``Q' = Q * cos + rotate(Q) * sin`` and
           ``K' = K * cos + rotate(K) * sin``, where ``cos`` / ``sin`` are the
           precomputed ``[T, head_dim]`` buffers broadcast to
           ``[1, T, 1, head_dim]`` over the batch and head axes.  Uses
           :func:`_rotate_half_per_axis` (split-half convention).
        4. **Transpose for SDPA** — rearrange to ``[B, H, T, d_k]``.
        5. **Scaled dot-product attention** — delegates to
           ``F.scaled_dot_product_attention``; PyTorch auto-selects the best
           backend (CuDNN on H100, FlashAttention on A100, etc.).  The
           ``dropout_p`` is set to ``self.attn_dropout`` during training and
           0.0 at inference.
        6. **Merge heads** — ``out.transpose(1, 2).reshape(B, T, C)``.
        7. **Output projection + dropout** — ``proj_drop(proj(out))``.

        Args:
            x: Input token sequence of shape ``[B, T, C]`` where:

                * ``B`` — batch size,
                * ``T = num_patches_h * num_patches_w + (1 if has_cls else 0)
                  + num_registers`` — total token count following the ViT-5
                  layout ``[patches, (CLS,) registers]``,
                * ``C = hidden_dim`` — channel dimension.

                The spatial dimensions of the patch grid are baked into the
                precomputed ``rope_cos`` / ``rope_sin`` buffers; ``T`` must
                match ``rope_cos.shape[0]`` exactly.

        Returns:
            torch.Tensor: Output tensor of shape ``[B, T, C]``, the same shape
            as the input.

        Raises:
            RuntimeError: If ``T`` does not match ``rope_cos.shape[0]``,
                causing a shape mismatch in the broadcast multiply
                ``q * cos``.  The expected value is
                ``num_patches_h * num_patches_w + int(has_cls) + num_registers``
                as set at construction time.
        """
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, num_heads, head_dim]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE: x' = x * cos + rotate(x) * sin
        # Buffers are [T, D]; unsqueeze to [1, T, 1, D] for broadcast over B and H.
        # Uses _rotate_half_per_axis (split-half) for checkpoint compatibility —
        # see its docstring for why the standard interleaved rotation doesn't work.
        cos = self.rope_cos[None, :, None, :]
        sin = self.rope_sin[None, :, None, :]
        q = q * cos + _rotate_half_per_axis(q) * sin
        k = k * cos + _rotate_half_per_axis(k) * sin

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

    def extra_repr(self) -> str:
        """Return a concise string summary of this module's configuration.

        Note:
            ``has_cls`` and ``scale`` are consequential hyperparameters that
            are **not** included in the output string.  Use
            ``module.has_cls`` and ``module.scale`` to inspect them directly.

        Returns:
            str: Comma-separated key=value pairs covering ``hidden_dim``,
            ``num_heads``, ``qk_norm``, ``num_registers``, patch grid size,
            ``rope_base``, and ``reg_rope_base``.
        """
        return (
            f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, "
            f"qk_norm={self.qk_norm}, num_registers={self.num_registers}, "
            f"patches=({self.num_patches_h}x{self.num_patches_w}), "
            f"rope_base={self.rope_base}, reg_rope_base={self.reg_rope_base}"
        )
