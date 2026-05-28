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

r"""Standard scaled dot-product attention for ND spatial inputs.

Background
----------
This module implements the classic scaled dot-product attention (Vaswani et al.,
"Attention Is All You Need", NeurIPS 2017):

.. math::

    \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V

where :math:`d_k` is the per-head dimension.  Multi-head attention computes
:math:`H` independent attention heads in parallel by splitting the channel
dimension :math:`C` into :math:`H` heads each of size :math:`d_k = C / H`,
then concatenating the results:

.. math::

    \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_H)

    \text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i

The total FLOP count for a sequence of length :math:`L` with :math:`H` heads and
per-head dimension :math:`d_k` is dominated by the attention matrix products:

.. math::

    \text{FLOPs} \approx 4 \cdot B \cdot H \cdot L^2 \cdot d_k

(two matrix multiplications of shape :math:`[L, d_k] \times [d_k, L]` each for
the logit computation and the value aggregation, summed over all heads and the
batch dimension; **attention kernel only** — QKV input/output projections in
:class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer` add
:math:`\sim 6 \cdot B \cdot L \cdot C^2` additional FLOPs).  This is
:math:`O(L^2)` in sequence length, which limits
scalability to long sequences — hence the availability of
:class:`~nvsubquadratic.modules.hyena_nd.Hyena` and
:class:`~nvsubquadratic.modules.ckconv_nd.CKConvND` as :math:`O(L \log L)`
alternatives.

Role in the dispatch pattern
----------------------------
:class:`Attention` is one of the concrete inner mixers recognised by
:class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`.  It receives
pre-projected Q, K, V tensors in **channels-last** layout ``[B, *spatial, C]``
and returns an output of the same shape.  Swapping :class:`Attention` for
:class:`~nvsubquadratic.modules.hyena_nd.Hyena` or
:class:`~nvsubquadratic.modules.ckconv_nd.CKConvND` requires only a config
change — the surrounding residual block is agnostic to the choice of mixer.

The inner mixer contract is: ``forward(q, k, v, cp_group, **kwargs)`` where all
tensors are channels-last ``[B, *spatial, C]`` and ``cp_group`` is the fourth
positional argument.

Positional encodings
--------------------
Rotary Positional Embeddings (RoPE, Su et al., "RoFormer", 2021) can be
applied before the attention computation.  For a pair of positions :math:`m` and
:math:`n`, RoPE encodes their relative distance through rotation matrices:

.. math::

    q_m^\top k_n = \text{Re}\!\left[\sum_{j} q_{m,j} k_{n,j}^* e^{i(m-n)\theta_j}\right]

where :math:`\theta_j = \text{base}^{-2j/d_k}`.  The cos/sin tables are
precomputed once at ``__init__`` time and stored as non-persistent registered
buffers so they survive ``.to(device)`` / ``.half()`` calls and are compatible
with ``torch.compile`` and CUDA-graph capture.  1D, 2D, and 3D spatial layouts
are supported with the following head-dim divisibility requirements:

- 1D: ``head_dim`` divisible by 2
- 2D: ``head_dim`` divisible by 4 (two half-dim tables, one per spatial axis)
- 3D: ``head_dim`` divisible by 6 (three one-third-dim tables, one per axis)

Optional QK normalisation (cosine attention) is also supported; when enabled
the scale factor is set to 1.0 since the norms are already unit-normalised.

Flash / memory-efficient attention
-----------------------------------
The forward pass delegates to ``torch.nn.functional.scaled_dot_product_attention``,
which automatically selects the most efficient backend available on the current
device (FlashAttention on A100/A10, cuDNN SDPA on H100, a memory-efficient
fallback otherwise).  No manual dtype casting is performed; AMP / ``autocast``
contexts are respected transparently.

The ``dropout_p`` argument is passed only during training (``module.training``
is ``True``); it is set to 0.0 at inference so dropout is never applied when
the model is in eval mode.

Context parallelism
-------------------
A ``cp_group`` argument is accepted for context-parallel training.  The current
implementation raises ``ValueError`` immediately — the zigzag all-gather/split
pathway is sketched for future compatibility but does not implement ring-attention
and would therefore materialise the full sequence on every rank.  This limitation
means the memory cost scales as :math:`O(L^2 / \text{cp\_size})` per rank
rather than the desired :math:`O(L^2 / \text{cp\_size}^2)`.  See
:class:`Attention` class docstring for details.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.parallel.utils import zigzag_gather_from_group_ranks, zigzag_split_across_group_ranks
from nvsubquadratic.utils import qk_norm, rope


class Attention(torch.nn.Module):
    r"""Multi-head scaled dot-product self-attention for 1D/2D/3D spatial inputs.

    Computes standard multi-head attention:

    .. math::

        \text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i

        \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_H)

    where :math:`d_k = C / H` is the per-head dimension, :math:`H` is the
    number of heads, and :math:`C` is the hidden (channel) dimension.

    Spatial layout
    --------------
    Inputs and outputs use **channels-last** layout:

    - 1D sequences: ``[B, T, C]``
    - 2D images:    ``[B, H, W, C]``
    - 3D volumes:   ``[B, D, H, W, C]``

    Internally, spatial dimensions are flattened to a single sequence axis
    ``L = prod(spatial_dims)`` for the SDPA kernel, then unflattened on output.

    Multi-head splitting
    --------------------
    The channel axis ``C`` is split into ``H`` heads of size ``d_k = C / H``.
    Internally the module works with the merged batch-head axis
    ``(B * H, L, d_k)`` before the SDPA call and re-merges after.

    QK normalisation (cosine attention)
    ------------------------------------
    When ``apply_qk_norm=True``, queries and keys are L2-normalised per head
    along the last dimension before the attention logits are formed.  This
    replaces the ``1/sqrt(d_k)`` scaling with a fixed scale of 1.0 to avoid
    flattening the already-normalised logits.

    Rotary Positional Embeddings (RoPE)
    ------------------------------------
    RoPE is applied to Q and K before QK-normalisation and before the SDPA
    call.  The cos/sin buffers are precomputed once at ``__init__`` from
    ``rope_spatial_dims`` and stored as non-persistent registered buffers
    (``persistent=False``) so they are reconstructed from ``__init__`` args
    and never serialised to checkpoints.  Head-dim divisibility requirements:

    - 1D: ``head_dim`` divisible by 2
    - 2D: ``head_dim`` divisible by 4 (two half-dim RoPE tables, one per axis)
    - 3D: ``head_dim`` divisible by 6 (three one-third-dim RoPE tables)

    Context parallelism (CP)
    -------------------------
    **Not yet functional.**  Passing a ``cp_group`` with ``size() > 1`` to
    ``forward`` immediately raises ``ValueError("Context parallelism must be
    revisited.")``.  The zigzag all-gather/split code below the ``raise`` is
    dead code retained as a sketch for a future ring-attention implementation.
    Pass ``cp_group=None`` (the default) for all current use cases.

    Backend selection
    -----------------
    Attention is computed with ``torch.nn.functional.scaled_dot_product_attention``,
    which auto-selects FlashAttention (A100), cuDNN SDPA (H100), or a
    memory-efficient fallback based on device capability.

    Attributes:
        hidden_dim (int): Total channel dimension ``C``.
        num_heads (int): Number of attention heads ``H``.  In the current
            implementation all heads are computed on every rank (there is no
            head-parallel CP split).  A ``# TODO(@farhad)`` in ``forward``
            flags that ``local_num_heads`` is always equal to ``num_heads``,
            which may need revisiting for tensor-parallel training.
        head_dim (int): Per-head dimension ``d_k = C / H``.
        scale (float): Attention logit scale ``1 / sqrt(d_k)``; set to 1.0
            when ``apply_qk_norm=True``.
        apply_qk_norm (bool): Whether L2 QK normalisation is active.
        use_rope (bool): Whether RoPE positional encoding is active.
        rope_base (float): Geometric base for RoPE frequency bands.
        is_causal (bool): Whether to apply a causal (auto-regressive) mask.
        attn_dropout (float): Dropout probability applied to attention weights
            during training.  Set to 0.0 automatically at inference regardless
            of this value.
        _rope_ndim (int): Spatial rank for which RoPE was initialised (1, 2,
            or 3).  Present only when ``use_rope=True``; not defined
            otherwise.  Used in ``forward`` to dispatch to the correct RoPE
            apply function.

    Args:
        hidden_dim (int): Total hidden-state dimension ``C``. Must be
            divisible by ``num_heads``.
        num_heads (int): Number of parallel attention heads ``H``.
        apply_qk_norm (bool): If ``True``, L2-normalise Q and K per head
            along the last dimension (cosine attention).
        use_rope (bool): If ``True``, apply Rotary Positional Embeddings to
            Q and K before the attention logits.
        is_causal (bool): If ``True``, apply a causal attention mask so each
            position attends only to earlier positions. Defaults to ``False``.
        attn_dropout (float): Dropout rate on attention weights (active only
            during training). Defaults to ``0.0``.
        rope_base (float): Base frequency for RoPE; controls how fast the
            rotation frequency decays across head-dim pairs.
            Defaults to ``10000.0``.
        rope_spatial_dims (tuple[int, ...] | None): Spatial grid shape used
            to precompute RoPE tables.  Required when ``use_rope=True``.
            Examples: ``(4096,)`` for 1D, ``(64, 64)`` for 2D,
            ``(8, 64, 64)`` for 3D.  Must match the spatial shape seen
            during ``forward``.

    Example::

        import torch
        from nvsubquadratic.modules.attention import Attention

        # 2D image attention with 8 heads, RoPE, and cosine-attention QK norm
        attn = Attention(
            hidden_dim=256,
            num_heads=8,
            apply_qk_norm=True,
            use_rope=True,
            rope_spatial_dims=(32, 32),
        )
        q = k = v = torch.randn(2, 32, 32, 256)  # [B, H, W, C]
        out = attn(q, k, v)  # [B, H, W, C]
        assert out.shape == q.shape
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        apply_qk_norm: bool,
        use_rope: bool,
        is_causal: bool = False,
        attn_dropout: float = 0.0,
        rope_base: float = 10000.0,
        rope_spatial_dims: tuple[int, ...] | None = None,
    ):
        """Initialise the Attention module and precompute RoPE buffers.

        Args:
            hidden_dim (int): Total channel dimension ``C``. Must be
                divisible by ``num_heads``.
            num_heads (int): Number of attention heads ``H``.
            apply_qk_norm (bool): Whether to L2-normalise Q and K per head.
            use_rope (bool): Whether to apply Rotary Positional Embeddings.
            is_causal (bool): Whether to use a causal attention mask.
                Defaults to ``False``.
            attn_dropout (float): Attention-weight dropout probability.
                Defaults to ``0.0``.
            rope_base (float): RoPE base frequency. Defaults to ``10000.0``.
            rope_spatial_dims (tuple[int, ...] | None): Spatial grid shape
                for RoPE table precomputation.  Required when
                ``use_rope=True``.  **Not stored** as an instance attribute;
                the caller is responsible for tracking the spatial dims if they
                need to recover them after construction (e.g. for serialisation
                or ``extra_repr``).  The corresponding cos/sin buffers are
                stored as non-persistent registered buffers (``rope_cos``,
                ``rope_sin``, etc.).

        Raises:
            AssertionError: If ``hidden_dim % num_heads != 0``.
            AssertionError: If ``use_rope=True`` and ``rope_spatial_dims``
                is ``None``.
            AssertionError: If RoPE head-dim divisibility requirements are
                not met (divisible by 2 for 1D, 4 for 2D, 6 for 3D).
            ValueError: If ``rope_spatial_dims`` has length other than 1, 2,
                or 3.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.apply_qk_norm = apply_qk_norm
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.is_causal = is_causal
        self.attn_dropout = attn_dropout

        # ── Precomputed RoPE cos/sin buffers ──────────────────────────────
        #
        # Following the ViT5Attention pattern: cos/sin tables are built once
        # at init and stored as non-persistent registered buffers so they:
        #   - survive .to(device) / .half() without manual bookkeeping,
        #   - are visible to CUDA-graph capture (no dynamic allocation in forward),
        #   - cause no graph breaks with torch.compile (no dict lookups,
        #     no torch.no_grad/inference_mode context managers in forward),
        #   - are NOT serialised into checkpoints (persistent=False), since
        #     they are deterministically reconstructed from __init__ args.
        if self.use_rope:
            assert rope_spatial_dims is not None, (
                "rope_spatial_dims is required when use_rope=True. "
                "Pass a tuple of spatial dimensions, e.g. (4096,) for 1D, "
                "(64, 64) for 2D, (8, 64, 64) for 3D."
            )
            self._rope_ndim = len(rope_spatial_dims)

            if self._rope_ndim == 1:
                (seq_len,) = rope_spatial_dims
                assert self.head_dim % 2 == 0, f"1D RoPE requires head_dim divisible by 2, got {self.head_dim}"
                cos, sin = rope.construct_rope_1d_cache_blh(
                    seq_len, self.head_dim, torch.device("cpu"), torch.float32, self.rope_base
                )
                self.register_buffer("rope_cos", cos, persistent=False)
                self.register_buffer("rope_sin", sin, persistent=False)

            elif self._rope_ndim == 2:
                height, width = rope_spatial_dims
                assert self.head_dim % 4 == 0, f"2D RoPE requires head_dim divisible by 4, got {self.head_dim}"
                cos_y, sin_y, cos_x, sin_x = rope.construct_rope_2d_cache_blh(
                    height, width, self.head_dim // 2, torch.device("cpu"), torch.float32, self.rope_base
                )
                self.register_buffer("rope_cos_y", cos_y, persistent=False)
                self.register_buffer("rope_sin_y", sin_y, persistent=False)
                self.register_buffer("rope_cos_x", cos_x, persistent=False)
                self.register_buffer("rope_sin_x", sin_x, persistent=False)

            elif self._rope_ndim == 3:
                depth, height, width = rope_spatial_dims
                assert self.head_dim % 6 == 0, f"3D RoPE requires head_dim divisible by 6, got {self.head_dim}"
                cos_z, sin_z, cos_y, sin_y, cos_x, sin_x = rope.construct_rope_3d_cache_blh(
                    depth, height, width, self.head_dim // 3, torch.device("cpu"), torch.float32, self.rope_base
                )
                self.register_buffer("rope_cos_z", cos_z, persistent=False)
                self.register_buffer("rope_sin_z", sin_z, persistent=False)
                self.register_buffer("rope_cos_y", cos_y, persistent=False)
                self.register_buffer("rope_sin_y", sin_y, persistent=False)
                self.register_buffer("rope_cos_x", cos_x, persistent=False)
                self.register_buffer("rope_sin_x", sin_x, persistent=False)

            else:
                raise ValueError(f"rope_spatial_dims must be 1D, 2D, or 3D, got {self._rope_ndim}D")

    def extra_repr(self) -> str:
        """Return a concise string summary of this module's configuration.

        Returns:
            str: Comma-separated key=value pairs for ``num_heads``,
                ``apply_qk_norm``, ``is_causal``, ``attn_dropout``,
                ``use_rope``, and ``rope_base``.
        """
        return f"num_heads={self.num_heads}, apply_qk_norm={self.apply_qk_norm}, is_causal={self.is_causal}, attn_dropout={self.attn_dropout}, use_rope={self.use_rope}, rope_base={self.rope_base}"

    def _flatten_spatial(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten all spatial dimensions into a single sequence axis.

        Converts a channels-last tensor with 1–3 spatial dimensions into a
        flat sequence tensor ``[B, L, C]`` where ``L = prod(spatial_dims)``.
        The original spatial shape is returned so it can be restored by
        :meth:`_unflatten_spatial`.

        Args:
            x (torch.Tensor): Input tensor of shape ``[B, *spatial_dims, C]``
                where ``len(spatial_dims)`` is 1, 2, or 3.

                .. note::

                    In :meth:`forward`, this is called after the channel → head
                    split, so the leading dimension is ``B * H`` (not ``B``) and
                    ``C`` is ``d_k = head_dim`` (not the full ``hidden_dim``).

        Returns:
            tuple[torch.Tensor, tuple[int, ...]]:
                - Flattened tensor of shape ``[B, L, C]`` with
                  ``L = prod(spatial_dims)``.
                - ``spatial_shape``: the original spatial dimensions as a
                  tuple, needed to invert the operation.

        Raises:
            AssertionError: If ``x.ndim`` is not 3, 4, or 5 (i.e., if the
                number of spatial dimensions is not 1, 2, or 3).
        """
        assert x.ndim in [3, 4, 5], (
            "Input must be tensors with shape (batch_size, *spatial_dims, hidden_dim), where len(spatial_dims) can be 1, 2, or 3."
        )
        spatial_shape = x.shape[1:-1]
        x = rearrange(x, "b ... c -> b (...) c")
        return x, spatial_shape

    def _unflatten_spatial(self, x: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        """Restore spatial dimensions after attention, inverting :meth:`_flatten_spatial`.

        This is the inverse of :meth:`_flatten_spatial`: given a flat ``[B, L, C]``
        tensor and the saved ``spatial_shape``, it reshapes the sequence axis back
        into the original spatial grid.

        Args:
            x (torch.Tensor): Flat sequence tensor of shape ``[B, L, C]``
                where ``L = prod(spatial_shape)``.
            spatial_shape (tuple[int, ...]): Original spatial dimensions
                ``(T,)``, ``(H, W)``, or ``(D, H, W)``.  Must have length
                1, 2, or 3.  Obtained from the second return value of
                :meth:`_flatten_spatial`.

        Returns:
            torch.Tensor: Tensor restored to ``[B, *spatial_shape, C]``.
                For 1D inputs (``len(spatial_shape) == 1``) the tensor is
                returned unchanged since the shape is already ``[B, T, C]``.

        Raises:
            AssertionError: If ``len(spatial_shape)`` is not 1, 2, or 3.
        """
        assert len(spatial_shape) in [1, 2, 3], "Spatial shape must be a tuple of length 1, 2, or 3."
        if len(spatial_shape) == 1:
            return x
        elif len(spatial_shape) == 2:
            return rearrange(x, "b (h w) c -> b h w c", h=spatial_shape[0], w=spatial_shape[1])
        else:
            return rearrange(x, "b (d h w) c -> b d h w c", d=spatial_shape[0], h=spatial_shape[1], w=spatial_shape[2])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cp_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        r"""Apply multi-head scaled dot-product attention.

        Computes:

        .. math::

            \text{out} = \text{Concat}_{i=1}^{H}
                \left[
                    \text{softmax}\!\left(
                        \frac{Q_i K_i^\top}{\sqrt{d_k}}
                    \right) V_i
                \right]

        where :math:`H` = ``num_heads`` and :math:`d_k` = ``head_dim``.
        When ``apply_qk_norm=True``, Q and K are L2-normalised before the
        logits are formed and the scale is 1.0 instead of ``1/sqrt(d_k)``.

        The forward pipeline is:

        1. (CP guard) Raises ``ValueError`` if ``cp_group.size() > 1``;
           pass ``cp_group=None`` for all current use cases.
        2. Split channel dim into heads: ``[B, *spatial, C] → [B*H, *spatial, d_k]``.
        3. (Optional) Apply RoPE to Q and K.
        4. (Optional) L2-normalise Q and K per head.
        5. Flatten spatial dims: ``[B*H, *spatial, d_k] → [B*H, L, d_k]``.
        6. Reshape to SDPA layout: ``[B*H, L, d_k] → [B, H, L, d_k]``.
        7. ``F.scaled_dot_product_attention`` (FlashAttention / cuDNN / fallback).
        8. Merge heads: ``[B, H, L, d_k] → [B, L, C]``.
        9. Unflatten spatial dims: ``[B, L, C] → [B, *spatial, C]``.
        10. (Optional CP) Zigzag-split output back to the local spatial slice.

        Args:
            query (torch.Tensor): Query tensor of shape
                ``[B, *spatial_dims, C]``.  ``spatial_dims`` may be
                ``(T,)``, ``(H, W)``, or ``(D, H, W)``.
            key (torch.Tensor): Key tensor of shape
                ``[B, *spatial_dims, C]``.  Must match ``query`` shape.
            value (torch.Tensor): Value tensor of shape
                ``[B, *spatial_dims, C]``.  Must match ``query`` shape.
            cp_group (torch.distributed.ProcessGroup | None): Context-parallel
                process group.  When not ``None`` and ``cp_group.size() > 1``,
                the full spatial sequence is gathered before attention and
                split back afterwards.  **Currently raises ``ValueError``
                as ring-attention is not yet implemented; provided for
                future compatibility.**  Defaults to ``None``.

        Returns:
            torch.Tensor: Output of shape ``[B, *spatial_dims, C]``, the
                same layout as the inputs.

        Raises:
            ValueError: If ``cp_group`` is provided and has size > 1 (context
                parallelism is not yet supported).
        """
        # CP communication for self-attention (Megatron-style):
        # CP only splits sequence/spatial dimension, not hidden_dim or heads
        # Pattern: All-gather K/V along sequence, keep Q local (or gather Q too for non-causal)

        if cp_group is not None and cp_group.size() > 1:
            raise ValueError("Context parallelism must be revisited.")
            # For non-causal attention, gather full sequence for Q, K, V
            # Input: [B, *spatial_partial, hidden_dim] where spatial_partial = spatial/cp_size
            # Output: [B, *spatial_full, hidden_dim]
            query = zigzag_gather_from_group_ranks(query, group=cp_group, seq_dim=1)
            key = zigzag_gather_from_group_ranks(key, group=cp_group, seq_dim=1)
            value = zigzag_gather_from_group_ranks(value, group=cp_group, seq_dim=1)

        # [B, * spatial_dims, hidden_dim] -> [B * num_heads, * spatial_dims, head_dim]
        # All heads are computed locally (no head splitting with CP)
        query = rearrange(query, "b ... (h d) -> (b h) ... d", h=self.num_heads)
        key = rearrange(key, "b ... (h d) -> (b h) ... d", h=self.num_heads)
        value = rearrange(value, "b ... (h d) -> (b h) ... d", h=self.num_heads)
        local_num_heads = self.num_heads  # TODO(@farhad): This looks to me like an error.

        # Optional RoPE positional encoding (before normalization)
        # Cos/sin buffers were precomputed at init — no dynamic allocation,
        # no dict lookups, no context managers; safe for torch.compile.
        if self.use_rope:
            if self._rope_ndim == 1:
                cache = (self.rope_cos, self.rope_sin)
                query = rope.apply_rope_1d_blh(query, cache)
                key = rope.apply_rope_1d_blh(key, cache)
            elif self._rope_ndim == 2:
                cache = (self.rope_cos_y, self.rope_sin_y, self.rope_cos_x, self.rope_sin_x)
                query = rope.apply_rope_2d_blh(query, cache)
                key = rope.apply_rope_2d_blh(key, cache)
            elif self._rope_ndim == 3:
                cache = (
                    self.rope_cos_z,
                    self.rope_sin_z,
                    self.rope_cos_y,
                    self.rope_sin_y,
                    self.rope_cos_x,
                    self.rope_sin_x,
                )
                query = rope.apply_rope_3d_blh(query, cache)
                key = rope.apply_rope_3d_blh(key, cache)

        # Optional QK normalization (after RoPE)
        if self.apply_qk_norm:
            query, key = qk_norm.apply_qk_norm(query, key, dim=-1)

        # Flatten spatial dims if present
        query, spatial_shape = self._flatten_spatial(query)
        key, _ = self._flatten_spatial(key)
        value, _ = self._flatten_spatial(value)

        # Reshape from [B*H, T, D] to [B, H, T, D] for SDPA
        query = rearrange(query, "(b h) t d -> b h t d", h=local_num_heads)
        key = rearrange(key, "(b h) t d -> b h t d", h=local_num_heads)
        value = rearrange(value, "(b h) t d -> b h t d", h=local_num_heads)

        # Scaled dot-product attention — let PyTorch auto-select the best
        # backend (CuDNN on H100, FlashAttention on A100, etc.).
        # No manual dtype cast: autocast handles precision.
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.is_causal,
            # When QK-norm is applied (cosine attention), disable the default
            # 1/sqrt(d) scaling — it would flatten the normalised logits.
            scale=self.scale if not self.apply_qk_norm else 1.0,
        )

        # Merge heads: [B, H, T, D] -> [B, T, H*D]
        out = rearrange(out, "b h t d -> b t (h d)", h=local_num_heads)

        # Unflatten back to spatial dims
        out = self._unflatten_spatial(out, spatial_shape)

        # CP communication: split sequence back to original distribution
        if cp_group is not None and cp_group.size() > 1:
            # Each rank has: [B, *spatial_full, hidden_dim] (full sequence, full hidden_dim)
            # Split sequence back: [B, *spatial_full, hidden_dim] -> [B, *spatial/cp, hidden_dim]
            out = zigzag_split_across_group_ranks(out, group=cp_group, seq_dim=1)

        return out
