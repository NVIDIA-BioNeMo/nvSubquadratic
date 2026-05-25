"""Adapter that plugs 2-D sequence mixers (e.g. Hyena) into the ViT-5 token-sequence architecture.

Drop-in replacement interface for :class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention`.
**Important**: unlike ``ViT5Attention``, the adapter owns no QKV or output projections.
All projection dimensions (``hidden_dim``, ``num_heads``, etc.) must be configured
inside ``inner_mixer_cfg`` (e.g. as part of ``QKVSequenceMixer``).

Why an adapter is needed
------------------------
:class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention` expects a flat
``[B, T, C]`` token sequence where ``T = num_patches + (1 if has_cls) + num_registers``.
It owns its own QKV and output projections and produces ``[B, T, C]`` output — the
residual block calls ``mixer(x)`` and adds the result back to ``x``.

2-D operators such as Hyena (wrapped in
:class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`) expect a spatial
grid ``[B, H, W, C]`` — they cannot consume the flat sequence directly. Moreover,
``QKVSequenceMixer`` provides its own QKV and output projections that are already
part of the inner mixer; duplicating projections in the adapter would waste memory
and parameters.

This module solves both issues with a *thin, stateless* reshape adapter:

1. Receives ``[B, T, C]`` from the residual block.
2. Reshapes to ``[B, T // grid_w, grid_w, C]`` (a 2-D spatial grid).
3. Delegates entirely to the inner mixer (any ``[B, H, W, C]``-in / ``[B, H, W, C]``-out
   module, e.g. ``QKVSequenceMixer(Hyena)``).  The inner mixer must be
   **shape-preserving** — its output must have the same ``[B, H, W, C]`` shape as its
   input; ``reshape`` will raise a cryptic error if the shape changes.
4. Reshapes back to ``[B, T, C]`` and returns.

The adapter itself adds **no parameters** — all learnable weights (input projection,
output projection, Hyena kernel generator) live inside the inner mixer.  Projection
dimensions (``hidden_dim``, ``num_heads``, etc.) must be configured inside
``inner_mixer_cfg``; the adapter accepts no ``hidden_dim`` argument itself.

Register tokens and the CLS token are **not** handled specially here: they are treated
as ordinary spatial positions in the grid.  The calling network
(:class:`~nvsubquadratic.networks.vit5_classification.ViT5ClassificationNet`) is
responsible for padding ``T`` so that it is exactly divisible by ``grid_w`` and for
arranging tokens into a layout that makes spatial sense to the mixer.  In the
hierarchical (:class:`~nvsubquadratic.networks.vit5_hierarchical_classification.ViT5HierarchicalClassificationNet`)
setting, ``grid_w`` must be consistent with the spatial width **after any patch-merging
stage** — the network supplies the correct ``grid_w`` at each stage.

Interface contract (same as ``ViT5Attention``)
----------------------------------------------
``forward(x, **mixer_kwargs) -> Tensor``

* Input:  ``x`` of shape ``[B, T, C]``.
* Output: tensor of shape ``[B, T, C]``.
* The inner mixer must return a tensor of the same shape ``[B, H, W, C]`` it received;
  downsampling or strided mixers are not supported.
* Optional kwargs (e.g. ``conditioning``) are forwarded verbatim to the inner mixer.

The module also exposes a ``flop_count(num_tokens, inference)`` method that
delegates to the inner mixer's ``flop_count``, matching the API used by the network
for FLOPs accounting.
"""

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5HyenaAdapter(nn.Module):
    """Bridges ViT-5's ``[B, T, C]`` token sequences and Hyena's ``[B, H, W, C]`` spatial interface.

    The adapter is a **parameter-free reshape wrapper**: it does not own any QKV
    projection, output projection, or positional encoding.  All learnable components
    live inside ``inner_mixer`` (typically a
    :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer` wrapping a
    :class:`~nvsubquadratic.modules.hyena_nd.Hyena` instance).

    Data flow::

        x: [B, T, C]
            │
            ▼  reshape  (T → H × grid_w, where H = T // grid_w)
        x: [B, H, grid_w, C]
            │
            ▼  inner_mixer  (any [B, H, W, C]-preserving mixer)
        x: [B, H, grid_w, C]
            │
            ▼  reshape  back
        x: [B, T, C]

    What the adapter handles vs. what the inner mixer handles:

    * **Adapter**: shape contract (flat ↔ 2-D), ``flop_count`` delegation.
    * **Inner mixer**: input projection (C → 3C), Hyena global convolution,
      gating, output projection (C → C), any normalisation, and optional
      FiLM / AdaLN conditioning.  The mixer receives tensors in channels-last
      layout ``[B, H, W, C]``; if it uses channels-first convolution internally
      (as ``QKVSequenceMixer`` does) it handles the permutation itself.

    Register-token handling:
        Register tokens (and the CLS token, if present) are treated as ordinary
        spatial positions within the reshaped grid — no masking or special-casing
        is applied.  The upstream network is responsible for:

        1. Padding the sequence so that ``T % grid_w == 0``.
        2. Choosing a ``grid_w`` that places register/CLS tokens in a predictable
           row (e.g. a dedicated "register row" at the bottom of the grid), so
           that the spatial convolution inside Hyena sees a consistent layout.
        3. In the hierarchical case, supplying the correct ``grid_w`` at each
           stage after patch merging changes the spatial width.

    Attributes:
        inner_mixer (nn.Module): The instantiated 2-D sequence mixer.  Accepts
            and returns ``[B, H, W, C]`` tensors in channels-last layout.
            Typically a :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`
            wrapping :class:`~nvsubquadratic.modules.hyena_nd.Hyena`.
        grid_w (int): Width of the 2-D spatial grid.  The height is inferred
            at runtime as ``T // grid_w``.
    """

    def __init__(
        self,
        inner_mixer_cfg: LazyConfig,
        grid_w: int,
    ):
        """Instantiate the adapter and its inner 2-D mixer.

        Args:
            inner_mixer_cfg: :class:`~nvsubquadratic.lazy_config.LazyConfig`
                describing the 2-D sequence mixer to instantiate (e.g.
                ``QKVSequenceMixer`` wrapping ``Hyena``).  The instantiated module
                must accept ``(x: Tensor[B, H, W, C], **kwargs)`` in channels-last
                layout and return a tensor of the same shape.  Any inner mixer
                that uses channels-first convolution (like ``QKVSequenceMixer``)
                handles the permutation internally.  Projection dimensions
                (``hidden_dim``, ``num_heads``, etc.) must be set inside this config;
                the adapter itself accepts no ``hidden_dim`` argument.
            grid_w: Width of the 2-D spatial grid.  Every call to ``forward``
                must supply a sequence length ``T`` that satisfies
                ``T % grid_w == 0``; the grid height is computed as
                ``H = T // grid_w``.  In a hierarchical network, pass the
                correct ``grid_w`` for each stage (after patch merging).  After
                a 2× patch-merging step, ``grid_w`` halves; the network's stage
                configuration (e.g. ``ViT5HierarchicalClassificationNet``) is the
                source of truth for each stage's ``grid_w``.
        """
        super().__init__()
        self.inner_mixer = instantiate(inner_mixer_cfg)
        self.grid_w = grid_w

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        """Delegate FLOPs accounting to the inner mixer.

        The adapter's reshape operations are pure metadata re-strides — zero
        arithmetic FLOPs — so the total cost is entirely determined by
        ``inner_mixer.flop_count``.

        Note:
            ``flop_count`` is a de-facto protocol, not enforced by a formal
            interface.  To guard against missing implementations use
            ``hasattr(adapter.inner_mixer, "flop_count")``.

        Args:
            num_tokens: Total flat sequence length ``T``.  Must satisfy
                ``T % grid_w == 0``.  The 2-D spatial dimensions passed to the
                inner mixer are ``(T // grid_w, grid_w)``.
            inference: Forwarded to the inner mixer.  Some mixers (e.g. those
                with cached Hyena kernels) report fewer FLOPs at inference time.

        Returns:
            Total FLOPs reported by the inner mixer for a ``(T // grid_w, grid_w)``
            spatial grid.

        Raises:
            AttributeError: If ``inner_mixer`` does not implement ``flop_count``.
        """
        spatial_dims = (num_tokens // self.grid_w, self.grid_w)
        return self.inner_mixer.flop_count(spatial_dims, inference=inference)

    def forward(self, x: torch.Tensor, **mixer_kwargs) -> torch.Tensor:
        """Reshape to 2-D grid, apply the inner mixer, reshape back.

        Args:
            x: Input token sequence of shape ``[B, T, C]`` where

                * ``B`` — batch size.
                * ``T`` — total sequence length (must satisfy ``T % grid_w == 0``).
                  Typical layout (set by the network, not enforced here):
                  ``[patch_tokens (H_patch * W_patch), CLS (0 or 1),
                  register_tokens (R), padding (P)]``.
                * ``C`` — channel / hidden dimension.

            **mixer_kwargs: Keyword arguments forwarded verbatim to
                ``inner_mixer.forward``.  Common keys include:

                * ``conditioning`` — FiLM/AdaLN conditioning tensor used by
                  some Hyena configurations.
                * ``cp_group`` — process group for context-parallel (AllToAll)
                  sharding inside the Hyena operator.

                Any additional kwargs accepted by the concrete inner mixer are
                also forwarded; consult the inner mixer's docstring for the full
                list.

        Returns:
            Tensor of shape ``[B, T, C]`` — the token sequence after 2-D
            Hyena mixing.  The first ``reshape`` (to ``[B, H, W, C]``) is a
            zero-copy view when ``x`` is contiguous.  If ``inner_mixer`` returns
            a non-contiguous tensor, the final ``reshape`` (back to ``[B, T, C]``)
            triggers a contiguous copy; this does not affect correctness but can
            affect memory traffic in CUDA-graph or ``torch.compile`` contexts.
            In practice, ``QKVSequenceMixer`` returns a contiguous tensor (its
            output projection is a ``Linear`` on the last axis), so the final
            ``reshape`` is typically a free view.

        Raises:
            RuntimeError: Raised by ``torch.Tensor.reshape`` if
                ``T % grid_w != 0``, with a message reporting the mismatched
                total element count.
        """
        B, T, C = x.shape
        x = x.reshape(B, T // self.grid_w, self.grid_w, C)
        x = self.inner_mixer(x, **mixer_kwargs)
        x = x.reshape(B, T, C)
        return x

    def extra_repr(self) -> str:
        """Return the string ``'grid_w=<value>'`` inserted into PyTorch's module repr."""
        return f"grid_w={self.grid_w}"
