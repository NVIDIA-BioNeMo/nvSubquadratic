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

# TODO: Add license header here


r"""Multi-head Continuous Kernel Convolution (CKConv) for 2D signals.

Background
----------
This module implements a **multi-head** extension of the CKConv operator
(Romero et al., "CKConv: Continuous Kernel Convolution With Arbitrary
Resolution", ICLR 2022, arXiv:2102.02611).  For the single-head (depthwise)
variant see :mod:`nvsubquadratic.modules.ckconv_nd`.

In the single-head variant, every channel is convolved independently with its
own implicit kernel — the kernel has shape ``[C, K_x, K_y]`` and there is no
cross-channel mixing in the convolution layer.  The multi-head variant enables
*dense channel mixing within each head*, analogous to multi-head attention but
for convolutions:

- The hidden dimension ``C`` is split into ``H`` heads of ``d = C / H``
  channels each.
- Within head ``h``, the convolution kernel ``K^h`` has shape
  ``[d, d, K_x, K_y]``, so every output channel sees all ``d`` input channels
  of that head through a spatially-varying weight.
- Across heads, channels remain isolated (no cross-head mixing).

Formally, for head ``h`` and a 2D input :math:`x^h \in \mathbb{R}^{d \times H \times W}`:

.. code-block:: none

    y^h = K^h * x^h + shortcut^h ⊙ x^h

where ``*`` denotes a dense 2D convolution (not depthwise), and the
concatenated output across all heads is:

.. code-block:: none

    y = [y^0 | y^1 | ... | y^{H-1}]   (concatenated along channel axis)

Each per-head kernel :math:`K^h_\theta` is produced by a shared SIREN network
evaluated on a continuous positional grid normalised to ``[-1, 1]^2``:

.. code-block:: none

    K^h_\theta(p) = MLP_\theta(pos_enc(p))

Because the MLP is small and the same for all heads, the number of parameters
scales with ``H * d^2 = C^2 / H`` (dense) rather than ``C^2`` (full
unstructured mixing), giving a parameter count between the depthwise and fully
dense extremes.

Low-rank / factored kernel
--------------------------
For large ``d``, the dense kernel ``[d, d, K_x, K_y]`` per head can be
expensive.  An optional low-rank factorisation (``kernel_rank=r``) decomposes
the kernel as:

.. code-block:: none

    K^h ≈ U^h · V^h,    U^h ∈ R^{d × r × K_x × K_y},
                          V^h ∈ R^{r × d × K_x × K_y}

The SIREN outputs ``num_heads * 2 * r * d`` values per position rather than
``num_heads * d^2``, and the frequency-domain contraction is split into two
cheaper steps:

.. code-block:: none

    z = V^h x     (rank-to-d_in contraction)
    y = U^h z     (d_out-to-rank contraction)

This reduces the per-head compute from :math:`O(d^2)` to :math:`O(2 d r)` and
the kernel parameter count by the same factor, while preserving the ``d × d``
mixing capacity at rank ``r``.

Boundary conditions
-------------------
Two boundary conditions are supported:

* **Zero-padding** (``fft_padding="zero"``): standard linear convolution with
  "same" output size.  Kernel size is ``2*N``, covering the full receptive
  field without wrap-around.
* **Circular / periodic** (``fft_padding="circular"``): wrap-around
  convolution.  The kernel size equals the input size (requires
  ``grid_type="single"``).

Shortcut (skip connection)
--------------------------
Every forward pass adds a per-channel learnable shortcut term:

.. code-block:: none

    y ← y + shortcut ⊙ x

where ``shortcut`` is a ``[hidden_dim]`` parameter vector initialised with
Kaiming-uniform scale, identical in design to the shortcut in
:class:`nvsubquadratic.modules.ckconv_nd.CKConvND`.

Current limitations
-------------------
* Only 2D inputs are supported (``data_dim`` must equal 2).
* Context parallelism (``cp_group``) is not yet implemented.
* Only ``"torch_fft"`` backend (no ``"subq_ops"`` backend support).

Related modules
---------------
* :mod:`nvsubquadratic.modules.ckconv_nd` — single-head (depthwise) CKConv,
  supporting 1D / 2D / 3D inputs, mixed boundary conditions, and causal mode.
* :mod:`nvsubquadratic.modules.kernels_nd` — implicit kernel parametrisation
  (``SIRENKernelND``, FiLM-conditioned variants) consumed by both variants.
* :mod:`nvsubquadratic.ops.fftconv_multihead` — the FFT convolution primitives
  that implement the dense per-head spatial mixing called from ``forward``.

References:
----------
* Romero et al. (2022). *CKConv: Continuous Kernel Convolution With Arbitrary
  Resolution*. ICLR 2022. https://arxiv.org/abs/2102.02611
* Sitzmann et al. (2020). *Implicit Neural Representations with Periodic
  Activation Functions*. NeurIPS 2020. (SIREN kernel used for ``MLP_θ``)
"""

import math
from typing import Literal

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.ops.fftconv_multihead import (
    fftconv2d_multihead_bhl,
    fftconv2d_multihead_circular_bhl,
    fftconv2d_multihead_lowrank_bhl,
    fftconv2d_multihead_lowrank_circular_bhl,
)


class CKConvMultiheadND(torch.nn.Module):
    r"""Multi-head CKConv for 2D signals with dense within-head channel mixing.

    Extends :class:`nvsubquadratic.modules.ckconv_nd.CKConvND` from depthwise
    convolution to *dense within-head* convolution by splitting the channel
    dimension into ``num_heads`` groups and applying a separate
    ``[head_dim × head_dim]`` implicit kernel to each group.

    **Mathematical description**

    Let :math:`x \\in \\mathbb{R}^{B \times C \times H \times W}` with
    :math:`C = H_\text{heads} \\cdot d` (``num_heads × head_dim``).  The input
    is partitioned into heads:

    .. code-block:: none

        x^h ∈ R^{B × d × H × W},   h = 0, ..., H_heads - 1

    Per head ``h``, the implicit kernel network produces
    :math:`K^h_\theta \\in \\mathbb{R}^{d \times d \times K_x \times K_y}`
    (full-rank) or its low-rank factorisation
    :math:`U^h \\in \\mathbb{R}^{d \times r \times K_x \times K_y}`,
    :math:`V^h \\in \\mathbb{R}^{r \times d \times K_x \times K_y}` (low-rank,
    when ``kernel_rank`` is set).  The output for each head is:

    .. code-block:: none

        y^h = K^h_θ * x^h + shortcut^h ⊙ x^h

    where ``*`` is a dense 2D linear (or circular) convolution computed via
    FFT.  The final output concatenates all head outputs:

    .. code-block:: none

        y = concat([y^0, y^1, ..., y^{H-1}], dim=channel)

    **Key differences from CKConvND (single-head)**

    * *Dense within-head mixing*: whereas :class:`CKConvND` uses a depthwise
      kernel of shape ``[C, K_x, K_y]`` (one scalar kernel per channel),
      ``CKConvMultiheadND`` uses a dense kernel of shape
      ``[H, d, d, K_x, K_y]`` (a :math:`d \times d` mixing matrix per head
      per spatial frequency).
    * *No context parallelism*: CP support is not yet implemented; passing
      ``cp_group`` to ``forward`` raises ``NotImplementedError``.
    * *2D only*: the current implementation only supports ``data_dim=2``
      (images with two spatial axes).  The single-head variant supports 1D,
      2D, and 3D inputs.
    * *No causal mode, no fp16 FFT, no chunked FFT*: these features from the
      single-head variant are not carried over here.
    * *Optional low-rank kernel*: the ``kernel_rank`` parameter (absent in
      ``CKConvND``) enables a factored ``U · V`` kernel decomposition that
      reduces SIREN output size and FFT cost by approximately ``2r / d``.

    Attributes:
        data_dim (int): Spatial rank of the input.  Always 2 for this class.
        hidden_dim (int): Total number of channels ``C = num_heads * head_dim``.
        num_heads (int): Number of independent heads ``H``.
        head_dim (int): Channels per head ``d = hidden_dim // num_heads``.
        fft_padding (str): Boundary condition — ``"zero"`` (linear conv) or
            ``"circular"`` (periodic conv).
        grid_type (str): Kernel grid size mode — ``"single"`` (kernel size
            equals input size, for circular conv) or ``"double"`` (kernel
            size is ``2N``, for zero-padded conv).
        kernel_rank (int or None): Rank of the low-rank kernel factorisation.
            ``None`` means full-rank ``[d, d, K_x, K_y]`` kernels are used.
        kernel (nn.Module): Implicit kernel generator (SIREN or similar).
            Called as ``kernel(grid_lens, conditioning=...)`` and returns
            ``(kernel_values, grid)`` where ``kernel_values`` has shape
            ``[1_or_B, K_x, K_y, num_heads * d * d]`` (full-rank) or
            ``[1_or_B, K_x, K_y, num_heads * 2 * r * d]`` (low-rank).
        mask (nn.Module): Attenuation mask applied to kernel values after
            generation.  ``nn.Identity`` when no mask is configured.
        shortcut (nn.Parameter): Learnable per-channel skip-connection scale
            of shape ``(hidden_dim,)``.  Added as ``shortcut ⊙ x`` after each
            convolution.  Initialised with Kaiming-uniform scale
            ``uniform(-1/√hidden_dim, 1/√hidden_dim)``.
        fftconv_fn (callable): Selected FFT convolution function, one of the
            four functions from :mod:`nvsubquadratic.ops.fftconv_multihead`.
            Signature varies by full-rank vs. low-rank path:
            full-rank: ``(x, kernel, shortcut) → output``;
            low-rank: ``(x, kernel_u, kernel_v, shortcut) → output``.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        num_heads: int,
        kernel_cfg: LazyConfig,
        mask_cfg: LazyConfig,
        grid_type: Literal["double", "single"],
        fft_padding: Literal["zero", "circular"],
        kernel_rank: int | None = None,
    ):
        """Construct a CKConvMultiheadND operator.

        Validates parameter combinations, derives ``head_dim``, adjusts the
        kernel output-scale for variance control, initialises the ``shortcut``
        parameter, and selects the appropriate FFT convolution function from
        :mod:`nvsubquadratic.ops.fftconv_multihead`.

        Args:
            data_dim: Spatial rank of the input signal.  Must be ``2``; a
                value other than 2 raises ``AssertionError``.
            hidden_dim: Total number of channels ``C``.  Must be divisible by
                ``num_heads``; a violation raises ``AssertionError``.
            num_heads: Number of independent convolution heads ``H``.  The
                channels are split evenly: ``head_dim = hidden_dim // num_heads``.
            kernel_cfg: ``LazyConfig`` that instantiates the implicit kernel
                generator (e.g. ``SIRENKernelND``).  The generator's output
                dimension must be set externally to match the expected flat
                kernel size:

                * Full-rank: ``out_dim = num_heads * head_dim * head_dim``
                * Low-rank: ``out_dim = num_heads * 2 * kernel_rank * head_dim``

                The output-scale weight ``kernel.out_linear.weight`` is
                multiplied in-place at construction time for variance control
                (see Notes).
            mask_cfg: ``LazyConfig`` for an optional attenuation mask applied
                to the generated kernel values.  Use ``torch.nn.Identity``
                for no masking.
            grid_type: Relationship between the SIREN coordinate grid and the
                input spatial size:

                * ``"single"``: grid spans ``(N+1)//2`` points per axis,
                  producing a kernel of size ``≈ N`` (for circular conv).
                * ``"double"``: grid spans ``N`` points per axis, producing a
                  kernel of size ``2N - 1 ≈ 2N`` (for zero-padded conv).

            fft_padding: Boundary condition for the convolution:

                * ``"zero"``: zero-padded linear convolution, "same" output
                  size.
                * ``"circular"``: periodic (wrap-around) convolution.  Requires
                  ``grid_type="single"`` and ``K_x == H``, ``K_y == W`` at
                  runtime (enforced by an ``AssertionError``).

            kernel_rank: Rank ``r`` for the low-rank kernel factorisation.
                When ``None`` (default), full-rank ``[d, d, K_x, K_y]``
                kernels are used and the SIREN outputs
                ``num_heads * d^2`` values per spatial position.  When set to
                an integer ``r < d``, the kernel is factored as
                :math:`K = U V` with ``U`` of shape ``[d, r, K_x, K_y]`` and
                ``V`` of shape ``[r, d, K_x, K_y]``, and the SIREN outputs
                ``num_heads * 2 * r * d`` values instead.

        Raises:
            AssertionError: If ``data_dim != 2``, ``hidden_dim % num_heads != 0``,
                ``grid_type`` is not ``"double"`` or ``"single"``,
                ``fft_padding`` is not ``"zero"`` or ``"circular"``, or
                ``fft_padding="circular"`` is combined with
                ``grid_type != "single"``.

        Notes:
            **Output-scale initialisation for variance control.**  The SIREN's
            final linear layer weight is rescaled in-place (via
            ``torch.no_grad()``) so that the convolution output has unit
            variance at initialisation:

            * Full-rank: each output channel sums over ``head_dim`` input
              channels weighted by the kernel, so the weight is multiplied by
              ``1 / √head_dim``.
            * Low-rank: the two-step ``U @ V`` contraction has variance that
              depends on both ``head_dim`` and ``kernel_rank``, so the weight
              is multiplied by ``(1 / (head_dim * kernel_rank))^{1/4}`` — the
              geometric mean of the per-factor scales.
        """
        assert data_dim == 2, f"CKConvMultiheadND currently only supports 2D. Got {data_dim}D."
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        assert grid_type in ["double", "single"], f"Invalid grid_type: {grid_type}"
        assert fft_padding in ["zero", "circular"], f"Invalid fft_padding: {fft_padding}"

        if fft_padding == "circular":
            assert grid_type == "single", (
                "fft_padding='circular' requires grid_type='single' (kernel size equals input size)."
            )

        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.fft_padding = fft_padding
        self.grid_type = grid_type
        self.kernel_rank = kernel_rank

        # Construct kernel (SIREN) and mask
        self.kernel = instantiate(kernel_cfg)
        self.mask = instantiate(mask_cfg)

        # Apply output scaling for variance control.
        # Full-rank: conv sums over head_dim terms → scale by 1/sqrt(head_dim).
        # Low-rank: conv sums over head_dim then rank → scale by 1/(head_dim * rank)^{1/4}
        #   so that each factor (U, V) has the right variance independently.
        with torch.no_grad():
            if kernel_rank is not None:
                self.kernel.out_linear.weight.data *= (1.0 / (self.head_dim * kernel_rank)) ** 0.25
            else:
                self.kernel.out_linear.weight.data *= math.sqrt(1.0 / self.head_dim)

        # Shortcut parameter for residual connection
        self.shortcut = torch.nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        bounds = math.sqrt(1.0 / hidden_dim)
        self.shortcut.data.uniform_(-bounds, bounds)

        # Select FFT convolution function
        if kernel_rank is not None:
            if fft_padding == "circular":
                self.fftconv_fn = fftconv2d_multihead_lowrank_circular_bhl
            else:
                self.fftconv_fn = fftconv2d_multihead_lowrank_bhl
        else:
            if fft_padding == "circular":
                self.fftconv_fn = fftconv2d_multihead_circular_bhl
            else:
                self.fftconv_fn = fftconv2d_multihead_bhl

    def extra_repr(self) -> str:
        """Return a concise summary string for ``print(module)`` and ``repr(module)``.

        Returns:
            A human-readable string listing ``data_dim``, ``hidden_dim``,
            ``num_heads``, ``head_dim``, ``fft_padding``, ``grid_type``, and
            (if set) ``kernel_rank``.
        """
        parts = (
            f"data_dim={self.data_dim}, hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"fft_padding={self.fft_padding!r}, grid_type={self.grid_type!r}"
        )
        if self.kernel_rank is not None:
            parts += f", kernel_rank={self.kernel_rank}"
        return parts

    def _reshape_lowrank_kernel(self, conv_kernel_flat: torch.Tensor, K_x: int, K_y: int, B: int | None = None):
        """Reshape the flat SIREN output into the low-rank ``U`` and ``V`` factors.

        The SIREN network outputs a flat tensor whose last dimension encodes
        all heads and both low-rank factors interleaved.  This method splits
        and permutes the tensor into the shapes expected by the low-rank FFT
        convolution ops in :mod:`nvsubquadratic.ops.fftconv_multihead`.

        The flat layout (last dim of ``conv_kernel_flat``) is:

        .. code-block:: none

            [head_0_factor_U | head_0_factor_V | head_1_factor_U | ... ]

        More precisely, for each head the SIREN outputs
        ``2 * rank * head_dim`` values arranged as
        ``[rank, head_dim]`` for U followed by ``[rank, head_dim]`` for V,
        with a factor index (``0`` = U, ``1`` = V) in the second-to-last
        logical dimension after reshaping to
        ``[..., num_heads, 2, rank, head_dim]``.

        Args:
            conv_kernel_flat: Flat SIREN output tensor.

                * Unbatched (``B=None``): shape
                  ``[K_x, K_y, num_heads * 2 * rank * head_dim]``.
                * FiLM-batched (``B`` is not ``None``): shape
                  ``[B, K_x, K_y, num_heads * 2 * rank * head_dim]``.

            K_x: Kernel spatial height (first spatial axis of the SIREN grid).
            K_y: Kernel spatial width (second spatial axis of the SIREN grid).
            B: Batch size when each sample has its own kernel (FiLM
                conditioning).  ``None`` for the standard unbatched path where
                one kernel is shared across all samples in the batch.

        Returns:
            A ``(kernel_u, kernel_v)`` tuple of contiguous tensors:

            * Unbatched (``B=None``):

              * ``kernel_u``: shape ``[num_heads, head_dim, rank, K_x, K_y]``
                — the "output projection" factor.
              * ``kernel_v``: shape ``[num_heads, rank, head_dim, K_x, K_y]``
                — the "input projection" factor.

            * FiLM-batched (``B`` is not ``None``):

              * ``kernel_u``: shape ``[B, num_heads, head_dim, rank, K_x, K_y]``.
              * ``kernel_v``: shape ``[B, num_heads, rank, head_dim, K_x, K_y]``.

        Notes:
            The contraction order in the FFT convolution is
            ``z = V x`` (input projection) followed by ``y = U z``
            (output projection), i.e. the einsum chain is
            ``(n, r, d_in) × (n, d_in) → (n, r)`` then
            ``(n, d_out, r) × (n, r) → (n, d_out)``.  The shape convention
            for ``kernel_u`` and ``kernel_v`` matches the einsum indices used
            in :func:`~CKConvMultiheadND.apply_convolution_batched_lowrank`.
        """
        rank = self.kernel_rank
        head_dim = self.head_dim
        num_heads = self.num_heads

        if B is not None:
            # FiLM-batched: [B, K_x, K_y, num_heads * 2 * rank * head_dim]
            # -> [B, K_x, K_y, num_heads, 2, rank, head_dim]
            reshaped = conv_kernel_flat.view(B, K_x, K_y, num_heads, 2, rank, head_dim)
            # After [:, :, :, :, idx, :, :] -> [B, K_x, K_y, num_heads, rank, head_dim]
            # U target: [B, num_heads, head_dim, rank, K_x, K_y]
            kernel_u = reshaped[:, :, :, :, 0, :, :].permute(0, 3, 5, 4, 1, 2).contiguous()
            # V target: [B, num_heads, rank, head_dim, K_x, K_y]
            kernel_v = reshaped[:, :, :, :, 1, :, :].permute(0, 3, 4, 5, 1, 2).contiguous()
        else:
            # Unbatched: [1, K_x, K_y, num_heads * 2 * rank * head_dim]
            # -> [K_x, K_y, num_heads, 2, rank, head_dim]
            reshaped = conv_kernel_flat.view(K_x, K_y, num_heads, 2, rank, head_dim)
            # After [:, :, :, idx, :, :] -> [K_x, K_y, num_heads, rank, head_dim]
            # U target: [num_heads, head_dim, rank, K_x, K_y]
            kernel_u = reshaped[:, :, :, 0, :, :].permute(2, 4, 3, 0, 1).contiguous()
            # V target: [num_heads, rank, head_dim, K_x, K_y]
            kernel_v = reshaped[:, :, :, 1, :, :].permute(2, 3, 4, 0, 1).contiguous()

        return kernel_u, kernel_v

    def apply_convolution(self, x: torch.Tensor, conv_kernel: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        """Apply the full-rank multi-head FFT convolution (shared kernel across batch).

        Calls the pre-selected ``fftconv_fn`` from
        :mod:`nvsubquadratic.ops.fftconv_multihead`.  The convolution is
        performed in float32 regardless of the input dtype to avoid numerical
        instability; the output is cast back to the input dtype before
        returning.

        The operation per head ``h`` is:

        .. code-block:: none

            y^h = K^h * x^h + shortcut^h ⊙ x^h

        implemented via rFFT in the frequency domain as a dense
        ``[head_dim × head_dim]`` matrix multiply at each spatial frequency.

        Args:
            x: Input tensor of shape ``[B, num_heads, head_dim, H, W]``,
                any floating-point dtype.
            conv_kernel: Full-rank kernel tensor of shape
                ``[num_heads, head_dim, head_dim, K_x, K_y]``, float32.
                ``head_dim`` appears twice — first as ``d_out``, second as
                ``d_in``.
            shortcut: Learnable skip-connection scale of shape
                ``[hidden_dim]``, float32.  Fused into the FFT op.

        Returns:
            Output tensor of shape ``[B, num_heads, head_dim, H, W]`` in the
            same dtype as the input ``x``.
        """
        x_dtype = x.dtype
        out = self.fftconv_fn(
            x.to(torch.float32),
            conv_kernel.to(torch.float32),
            shortcut.to(torch.float32),
        )
        return out.to(x_dtype)

    def apply_convolution_lowrank(
        self, x: torch.Tensor, kernel_u: torch.Tensor, kernel_v: torch.Tensor, shortcut: torch.Tensor
    ) -> torch.Tensor:
        """Apply the low-rank multi-head FFT convolution (shared kernel across batch).

        Calls the pre-selected low-rank ``fftconv_fn`` from
        :mod:`nvsubquadratic.ops.fftconv_multihead`.  The two-step contraction
        avoids materialising the full ``[head_dim × head_dim]`` kernel spectrum
        per spatial frequency:

        .. code-block:: none

            z = V x          (shape: [B, num_heads, rank, H, W])
            y = U z          (shape: [B, num_heads, head_dim, H, W])
            y += shortcut ⊙ x

        The convolution is performed in float32; the output is cast back to
        the input dtype before returning.

        Args:
            x: Input tensor of shape ``[B, num_heads, head_dim, H, W]``,
                any floating-point dtype.
            kernel_u: Output-projection factor of shape
                ``[num_heads, head_dim, rank, K_x, K_y]``, float32.
            kernel_v: Input-projection factor of shape
                ``[num_heads, rank, head_dim, K_x, K_y]``, float32.
            shortcut: Learnable skip-connection scale of shape
                ``[hidden_dim]``, float32.  Fused into the FFT op.

        Returns:
            Output tensor of shape ``[B, num_heads, head_dim, H, W]`` in the
            same dtype as the input ``x``.
        """
        x_dtype = x.dtype
        out = self.fftconv_fn(
            x.to(torch.float32),
            kernel_u.to(torch.float32),
            kernel_v.to(torch.float32),
            shortcut.to(torch.float32),
        )
        return out.to(x_dtype)

    def apply_convolution_batched(
        self, x: torch.Tensor, conv_kernel: torch.Tensor, shortcut: torch.Tensor
    ) -> torch.Tensor:
        """Apply the full-rank multi-head FFT convolution with per-sample kernels.

        Used when FiLM conditioning is active and each sample in the batch has
        its own kernel (``conv_kernel.shape[0] == B``).  The FFT convolution is
        implemented directly via ``torch.fft.rfft2`` / ``irfft2`` and
        ``torch.einsum``, bypassing the ``fftconv_fn`` (which expects a
        shared-kernel layout).

        The zero-padded FFT size is computed per-axis as:

        .. code-block:: none

            fft_h = min(H + (K_x + 1) // 2, 2 * H)
            fft_w = min(W + (K_y + 1) // 2, 2 * W)

        which matches the padding convention of the standard (non-batched)
        FFT convolution ops in :mod:`nvsubquadratic.ops.fftconv_multihead`.

        The frequency-domain operation per sample is:

        .. code-block:: none

            ŷ_{b,n,o,fx,fy} = Σ_i  K̂_{b,n,o,i,fx,fy} · x̂_{b,n,i,fx,fy}

        implemented as a single einsum ``"bnihw,bnoihw->bnohw"``.

        After the inverse FFT, the output is cropped to ``[H, W]`` using a
        centered crop starting at ``(K_x // 2, K_y // 2)``, and the shortcut
        term is added.

        Args:
            x: Input tensor of shape ``[B, num_heads, head_dim, H, W]``,
                any floating-point dtype.  Cast to float32 internally.
            conv_kernel: Per-sample full-rank kernel of shape
                ``[B, num_heads, head_dim, head_dim, K_x, K_y]``, float32.
                ``head_dim`` appears twice (``d_out``, ``d_in``).
            shortcut: Learnable skip-connection scale of shape
                ``[hidden_dim]``, float32.  Reshaped to
                ``[1, num_heads, head_dim, 1, 1]`` and added as
                ``shortcut ⊙ x`` after the inverse FFT.

        Returns:
            Output tensor of shape ``[B, num_heads, head_dim, H, W]``, float32.
        """
        x = x.to(torch.float32)
        conv_kernel = conv_kernel.to(torch.float32)
        shortcut = shortcut.to(torch.float32)

        _B, num_heads, head_dim, H, W = x.shape
        K_x, K_y = conv_kernel.shape[-2], conv_kernel.shape[-1]

        fft_h = min(H + (K_x + 1) // 2, 2 * H)
        fft_w = min(W + (K_y + 1) // 2, 2 * W)

        x_fft = torch.fft.rfft2(x, s=(fft_h, fft_w))
        k_fft = torch.fft.rfft2(conv_kernel, s=(fft_h, fft_w))

        # Batched dense conv: both x and kernel have batch dim
        out_fft = torch.einsum("bnihw,bnoihw->bnohw", x_fft, k_fft)

        crop_h = K_x // 2
        crop_w = K_y // 2
        out_full = torch.fft.irfft2(out_fft, s=(fft_h, fft_w))
        out = out_full[..., crop_h : crop_h + H, crop_w : crop_w + W]

        shortcut_reshaped = shortcut.view(1, num_heads, head_dim, 1, 1)
        out = out + x * shortcut_reshaped

        return out

    def apply_convolution_batched_lowrank(
        self,
        x: torch.Tensor,
        kernel_u: torch.Tensor,
        kernel_v: torch.Tensor,
        shortcut: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the low-rank multi-head FFT convolution with per-sample kernels.

        Used when FiLM conditioning is active and each sample in the batch has
        its own low-rank kernel pair ``(U, V)``.  The two-step frequency-domain
        contraction avoids materialising the full ``[head_dim × head_dim]``
        kernel spectrum per frequency bin:

        .. code-block:: none

            ẑ_{b,n,r,fx,fy} = Σ_i  V̂_{b,n,r,i,fx,fy} · x̂_{b,n,i,fx,fy}   ("bnihw,bnrihw->bnrhw")
            ŷ_{b,n,o,fx,fy} = Σ_r  Û_{b,n,o,r,fx,fy} · ẑ_{b,n,r,fx,fy}   ("bnrhw,bnorhw->bnohw")

        The same zero-padded FFT size and centered-crop convention as
        :meth:`apply_convolution_batched` are used.  After the inverse FFT the
        shortcut term is added element-wise.

        Args:
            x: Input tensor of shape ``[B, num_heads, head_dim, H, W]``,
                any floating-point dtype.  Cast to float32 internally.
            kernel_u: Per-sample output-projection factor of shape
                ``[B, num_heads, head_dim, rank, K_x, K_y]``, float32.
            kernel_v: Per-sample input-projection factor of shape
                ``[B, num_heads, rank, head_dim, K_x, K_y]``, float32.
            shortcut: Learnable skip-connection scale of shape
                ``[hidden_dim]``, float32.  Reshaped to
                ``[1, num_heads, head_dim, 1, 1]`` and added after the inverse
                FFT.

        Returns:
            Output tensor of shape ``[B, num_heads, head_dim, H, W]``, float32.
        """
        x = x.to(torch.float32)
        kernel_u = kernel_u.to(torch.float32)
        kernel_v = kernel_v.to(torch.float32)
        shortcut = shortcut.to(torch.float32)

        _B, num_heads, head_dim, H, W = x.shape
        K_x, K_y = kernel_u.shape[-2], kernel_u.shape[-1]

        fft_h = min(H + (K_x + 1) // 2, 2 * H)
        fft_w = min(W + (K_y + 1) // 2, 2 * W)

        x_fft = torch.fft.rfft2(x, s=(fft_h, fft_w))
        u_fft = torch.fft.rfft2(kernel_u, s=(fft_h, fft_w))
        v_fft = torch.fft.rfft2(kernel_v, s=(fft_h, fft_w))

        # Two-step low-rank conv (avoids materializing full [head_dim x head_dim] K_fft)
        # Step 1: z = V @ x — contract over head_dim_in
        z_fft = torch.einsum("bnihw,bnrihw->bnrhw", x_fft, v_fft)
        # Step 2: y = U @ z — contract over rank
        out_fft = torch.einsum("bnrhw,bnorhw->bnohw", z_fft, u_fft)

        crop_h = K_x // 2
        crop_w = K_y // 2
        out_full = torch.fft.irfft2(out_fft, s=(fft_h, fft_w))
        out = out_full[..., crop_h : crop_h + H, crop_w : crop_w + W]

        shortcut_reshaped = shortcut.view(1, num_heads, head_dim, 1, 1)
        out = out + x * shortcut_reshaped

        return out

    def forward(
        self,
        x: torch.Tensor,
        is_bhl_input: bool = False,
        cp_group: torch.distributed.ProcessGroup = None,
        **mixer_kwargs,
    ) -> torch.Tensor:
        """Run the CKConvMultiheadND forward pass.

        Generates the per-head implicit kernel from the SIREN network,
        optionally applies the attenuation mask, reshapes the flat SIREN output
        into the per-head kernel layout, and applies the FFT convolution with
        the shortcut term.

        The computation (non-FiLM path) is:

        .. code-block:: none

            # 1. Determine grid size
            grid_lens = [(s + 1) // 2 for s in (H, W)]   # if grid_type == "single"
                      = [H, W]                             # if grid_type == "double"

            # 2. Generate kernel from SIREN
            k_flat, grid = self.kernel(grid_lens)          # [1, K_x, K_y, C_flat]

            # 3. Apply mask (if not Identity)
            k_flat = self.mask(grid=grid, x=k_flat)

            # 4. Reshape flat output into per-head kernel
            # Full-rank:  conv_kernel [num_heads, head_dim, head_dim, K_x, K_y]
            # Low-rank:   kernel_u [num_heads, head_dim, rank, K_x, K_y],
            #             kernel_v [num_heads, rank, head_dim, K_x, K_y]

            # 5. Apply FFT convolution + shortcut
            out_heads = apply_convolution*(x_heads, kernel*, shortcut)

        When the kernel is FiLM-conditioned (``batched_kernel=True``), each
        sample receives its own kernel and the batched convolution methods
        (:meth:`apply_convolution_batched` or
        :meth:`apply_convolution_batched_lowrank`) are used instead.

        Args:
            x: Input signal tensor.  Two supported layouts:

                * **Channels-last** (``is_bhl_input=False``, default): shape
                  ``[B, H, W, hidden_dim]``.  Rearranged internally to
                  ``[B, num_heads, head_dim, H, W]`` via einops.
                * **Channels-first** (``is_bhl_input=True``): shape
                  ``[B, hidden_dim, H, W]``.  Reshaped internally to
                  ``[B, num_heads, head_dim, H, W]`` via ``view``.

            is_bhl_input: If ``True``, treat ``x`` as channels-first
                (BHL) layout.  Default: ``False`` (channels-last / BLH).
            cp_group: Context-parallel process group.  **Not supported** — if
                provided and ``cp_group.size() > 1`` a ``NotImplementedError``
                is raised immediately.  Accepted as a keyword argument for
                interface compatibility with :class:`CKConvND`.
            **mixer_kwargs: Additional keyword arguments forwarded to the
                kernel generator.  Recognised key:

                * ``conditioning`` (``torch.Tensor``, shape ``[B, cond_dim]``):
                  conditioning vector for FiLM-enabled kernels such as
                  ``SIRENKernelND`` with a ``film_cfg``.  When supplied, the
                  SIREN returns a per-sample kernel (batch dimension == B);
                  otherwise the kernel is shared across the batch (batch
                  dimension == 1).

        Returns:
            Output tensor in the same memory layout as the input ``x``:

            * Channels-last: shape ``[B, H, W, hidden_dim]``
              (when ``is_bhl_input=False``).
            * Channels-first: shape ``[B, hidden_dim, H, W]``
              (when ``is_bhl_input=True``).

        Raises:
            NotImplementedError: If ``cp_group`` is provided with
                ``cp_group.size() > 1``.  Context parallelism is not yet
                implemented for ``CKConvMultiheadND``.
        """
        if cp_group is not None and cp_group.size() > 1:
            raise NotImplementedError("Context parallelism not yet supported for CKConvMultiheadND")

        # Get spatial dimensions
        if is_bhl_input:
            B, _C, H, W = x.shape
            # Reshape to [B, num_heads, head_dim, H, W]
            x_heads = x.view(B, self.num_heads, self.head_dim, H, W)
        else:
            B, H, W, _C = x.shape
            # Reshape to [B, num_heads, head_dim, H, W]
            x_heads = rearrange(x, "b h w (n d) -> b n d h w", n=self.num_heads, d=self.head_dim)

        spatial_dims = (H, W)

        # Determine grid lengths for kernel generation
        if self.grid_type == "single":
            grid_lens = [(s + 1) // 2 for s in spatial_dims]
        else:  # "double"
            grid_lens = spatial_dims

        # Generate kernel from SIREN (pass conditioning if available for FiLM-enabled kernels)
        # Full-rank output: [1, *spatial, num_heads * head_dim * head_dim]
        # Low-rank output: [1, *spatial, num_heads * 2 * rank * head_dim]
        conditioning = mixer_kwargs.get("conditioning", None)
        conv_kernel_flat, grid = self.kernel(grid_lens, conditioning=conditioning)

        # Apply mask if not identity
        if not isinstance(self.mask, torch.nn.Identity):
            conv_kernel_flat = self.mask(grid=grid, x=conv_kernel_flat)

        K_x, K_y = conv_kernel_flat.shape[-3], conv_kernel_flat.shape[-2]
        batched_kernel = conv_kernel_flat.shape[0] > 1

        if self.kernel_rank is not None:
            # Low-rank path: reshape SIREN output into U and V factors
            kernel_u, kernel_v = self._reshape_lowrank_kernel(
                conv_kernel_flat, K_x, K_y, B=B if batched_kernel else None
            )

            # Cache kernel stats for debugging
            if getattr(self, "_cache_debug_stats", False):
                with torch.no_grad():
                    self._debug_stats = {
                        "norm_u": kernel_u.norm().item(),
                        "norm_v": kernel_v.norm().item(),
                        "max_abs_u": kernel_u.abs().max().item(),
                        "max_abs_v": kernel_v.abs().max().item(),
                    }

            if batched_kernel:
                out_heads = self.apply_convolution_batched_lowrank(x_heads, kernel_u, kernel_v, self.shortcut)
            else:
                out_heads = self.apply_convolution_lowrank(x_heads, kernel_u, kernel_v, self.shortcut)
        else:
            # Full-rank path: reshape SIREN output into [num_heads, head_dim, head_dim, K_x, K_y]
            if batched_kernel:
                conv_kernel = conv_kernel_flat.view(B, K_x, K_y, self.num_heads, self.head_dim, self.head_dim)
                conv_kernel = conv_kernel.permute(0, 3, 4, 5, 1, 2).contiguous()
            else:
                conv_kernel = conv_kernel_flat.view(K_x, K_y, self.num_heads, self.head_dim, self.head_dim)
                conv_kernel = conv_kernel.permute(2, 3, 4, 0, 1).contiguous()

            # Cache kernel stats for debugging
            if getattr(self, "_cache_debug_stats", False):
                with torch.no_grad():
                    self._debug_stats = {
                        "norm": conv_kernel.norm().item(),
                        "max_abs": conv_kernel.abs().max().item(),
                    }

            if batched_kernel:
                out_heads = self.apply_convolution_batched(x_heads, conv_kernel, self.shortcut)
            else:
                out_heads = self.apply_convolution(x_heads, conv_kernel, self.shortcut)

        # Reshape back to original format
        if is_bhl_input:
            # [B, num_heads, head_dim, H, W] -> [B, hidden_dim, H, W]
            out = out_heads.reshape(B, self.hidden_dim, H, W)
        else:
            # [B, num_heads, head_dim, H, W] -> [B, H, W, hidden_dim]
            out = rearrange(out_heads, "b n d h w -> b h w (n d)")

        return out
