"""LayerScale: learnable per-channel scaling of residual branch outputs.

LayerScale (Touvron et al., "Going deeper with Image Transformers", arXiv:2103.17239,
ICCV 2021) is a lightweight technique for stabilising the training of very deep
vision transformers.  In a standard residual network the update rule is

    x ← x + F(x)

which means that the residual branch F can grow arbitrarily large at
initialisation, making depth 36+ networks difficult to train.  LayerScale
modifies the update to

    x ← x + λ ⊙ F(x)

where ``λ ∈ ℝ^C`` is a *learnable*, *per-channel* scalar vector initialised to
a small positive constant (e.g. ``1e-4``).  At the start of training the gated
residual updates are therefore nearly zero, so the effective depth of the network
is small.  As training proceeds ``λ`` grows, progressively incorporating the
residual branches until the full network capacity is exploited.

The module is used by
:class:`~nvsubquadratic.modules.vit5_residual_block.ViT5ResidualBlock` where
**two independent** :class:`LayerScale` instances — one for the sequence-mixer
branch and one for the MLP branch — wrap each residual update:

.. code-block:: text

    x = x + drop_path(ls_attn(mixer(norm(x))))
    x = x + drop_path(ls_mlp(mlp(mlp_norm(x))))

Reference:
    Touvron, H., et al. "Going deeper with Image Transformers."
    ICCV 2021.  arXiv:2103.17239.
"""

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """Learnable per-channel scalar gate for residual branch outputs.

    **Operation**

    Given an input tensor ``x`` of arbitrary leading batch / spatial dimensions
    followed by a channel dimension ``C``, LayerScale computes

        output = x * γ

    where ``γ ∈ ℝ^C`` is broadcast element-wise along all axes except the last
    one.  Concretely, if ``x`` has shape ``(B, T, C)`` (the ViT-5 layout) then
    ``γ`` is of shape ``(C,)`` and is broadcast to ``(1, 1, C)`` automatically
    by PyTorch.  The same broadcast rule applies to any channels-last layout:
    ``(B, H, W, C)``, ``(B, T, H, W, C)``, etc.

    **Training dynamics**

    The ``init_value`` argument controls the initial magnitude of every element
    of ``γ``.  A small value (e.g. ``1e-4``) means the residual update is
    almost entirely suppressed at the start of training, which:

    * Prevents gradient explosion in very deep networks (depth ≥ 24).
    * Lets the skip connections carry most of the signal early on, and allows
      the residual branches to activate gradually once they have learned useful
      features.

    Using a larger ``init_value`` (e.g. ``1.0``) is appropriate when
    fine-tuning from a checkpoint where the residual branches are already
    well-trained and suppressing them would slow convergence.

    The parameter ``γ`` is tagged with ``_no_weight_decay = True`` so that
    optimiser weight-decay regularisation (L2) is **not** applied to it.  This
    is standard practice and matches the original CaiT training recipe.

    **How it differs from a plain nn.Linear / scalar gate**

    Unlike a ``nn.Linear(C, C)`` projection (which mixes channels), LayerScale
    applies an independent scalar per channel.  This is equivalent to a
    *diagonal* linear map ``diag(γ)`` and requires only ``C`` parameters rather
    than ``C²``.

    Attributes:
        gamma (nn.Parameter): Learnable scale vector of shape ``(dim,)``,
            initialised to ``init_value``.  Tagged ``_no_weight_decay = True``
            to exclude it from L2 weight-decay in the optimiser.

    Example::

        ls = LayerScale(dim=768, init_value=1e-4)
        x = torch.randn(2, 196, 768)   # [B, T, C]
        out = ls(x)                    # [B, T, C], same shape as x
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
        """Initialise the learnable scale vector.

        Args:
            dim: Channel dimension ``C``.  Determines the length of the
                ``gamma`` parameter vector.  Must match the channel (last)
                dimension of tensors passed to :meth:`forward`.
            init_value: Initial value for every element of ``gamma``.
                All elements are set to this scalar at construction time.
                Typical choices:

                * ``1e-4`` — recommended for training from scratch on deep
                  networks; effectively suppresses residual branches initially.
                * ``1e-5`` — used in the original CaiT paper for the deepest
                  (depth-48) variants.
                * ``1.0`` — effectively disables the gating at initialisation;
                  useful when fine-tuning from a strong pre-trained checkpoint.
        """
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        self.gamma._no_weight_decay = True

    def flop_count(self, num_tokens: int) -> int:
        """Count floating-point multiply operations for one forward pass.

        Each element of ``x`` is multiplied by the corresponding element of
        ``gamma``, so the total number of scalar multiplications is

            FLOPs = num_tokens × dim

        where ``dim = self.gamma.shape[0]``.  The broadcast of ``gamma`` is
        free (no arithmetic), and the element-wise multiply is counted as one
        FLOP per output element (following the convention used throughout this
        codebase of counting multiply-only, not multiply-add pairs).

        Args:
            num_tokens: Number of token (or spatial) positions ``T`` in the
                input tensor.  The full tensor has ``T × dim`` elements, each
                requiring one multiply.

        Returns:
            Integer FLOP count equal to ``num_tokens * dim``.
        """
        dim = self.gamma.shape[0]
        return num_tokens * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel scaling to the input tensor.

        Multiplies every channel slice of ``x`` by the corresponding scalar in
        ``gamma``.  PyTorch broadcasts ``gamma`` of shape ``(C,)`` across all
        leading dimensions of ``x`` automatically.

        Args:
            x: Input tensor of shape ``(*leading_dims, C)`` where the last
               dimension must equal ``dim`` passed to the constructor.
               Common shapes:

               * ``(B, T, C)`` — ViT-5 / transformer token sequences.
               * ``(B, H, W, C)`` — channels-last 2-D feature maps.
               * ``(B, T, H, W, C)`` — channels-last 3-D (video) tensors.

        Returns:
            torch.Tensor: Scaled tensor with the same shape and dtype as ``x``,
            where ``output[..., c] = x[..., c] * gamma[c]`` for each channel
            index ``c``.
        """
        return x * self.gamma
