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


"""Implicit / learned kernel parametrisations for N-dimensional convolutional filters.

Overview
--------
Standard convolutional sequence models fix the filter bank at construction time
(e.g. a Gabor filter or a learnable lookup table).  The classes in this module
instead **parametrise the kernel implicitly**: a small MLP maps spatial
coordinates to kernel values, so the filter shape is determined by the MLP's
learned weights rather than by an explicit table of size proportional to the
signal length.  This approach — sometimes called an *implicit neural
representation* (INR) of the kernel — has two key advantages:

1.  **Resolution-independence**: the same parameter count describes kernels of
    any length.  A model trained at resolution N can be evaluated at a
    different resolution without any weight surgery.

2.  **Continuous inductive bias**: the MLP is smooth almost everywhere, which
    acts as a spectral regulariser and avoids the aliasing artefacts that arise
    when a fixed filter is up-sampled or sub-sampled.

The main consumer of this module is ``nvsubquadratic.modules.hyena_nd.Hyena``
(via ``nvsubquadratic.modules.ckconv_nd.CKConvND``), where the generated kernel
is passed directly to the FFT convolution primitives in ``nvsubquadratic.ops``.

Two positional-encoding families are provided, each yielding a matching kernel
class:

* **Random Fourier Features (RFF)**: the first layer is a random (fixed)
  frequency matrix; activations are cosine+sine concatenated.  The result is
  that the MLP's effective prior is a stationary (shift-invariant) kernel
  corresponding to the RBF kernel.  Controlled by ``omega_0`` (bandwidth).

* **SIREN** (sinusoidal representation network, Sitzmann et al. 2020): every
  layer uses ``sin`` activation.  The frequency of the first layer is set by
  ``omega_0``; subsequent layers use ``hidden_omega_0``.  Produces smoother
  high-frequency content than RFF and is amenable to multi-frequency
  initialisations (see ``MultiOmegaSIRENKernelND`` and the block-diagonal
  variants below).

ND here refers to the spatial dimensionality of the signal: 1D (sequences),
2D (images), 3D (video / volumetric data), or higher.  The coordinate grid
covers ``[-1, 1]^D`` normalised across all axes; the ``L_cache`` parameter
controls the number of discrete grid positions cached per axis, and the cache
grows automatically at runtime whenever a larger input is encountered.

Input-dependent / conditional kernels
--------------------------------------
``SIRENKernelND`` and all its subclasses accept an optional
``conditioning`` argument of shape ``[B, C]`` in their ``forward`` method.
When a ``KernelFiLMGenerator`` is wired in via the ``film_cfg`` constructor
argument, the generator maps this conditioning vector to a list of per-layer
``(gamma, beta)`` pairs that modulate the SIREN's hidden activations via
Feature-wise Linear Modulation (FiLM):

    h_i <- gamma_i * h_i + beta_i

This makes the produced kernel batch-dependent: the output has shape
``[B, *spatial, out_dim]`` instead of the usual ``[1, *spatial, out_dim]``.
This feature is used in diffusion models and other conditional generation tasks
where each sample needs a different long-range filter.  The ``conditioning``
argument is ignored (no-op) when no ``film_cfg`` is provided.

Kernel classes in this module
------------------------------
``RandomFourierKernelND``
    RFF-based kernel.  Recommended when a well-understood stationary prior is
    desired.  The ``omega_0`` parameter directly controls the bandwidth of the
    implied RBF kernel.

``SIRENKernelND``
    SIREN-based kernel.  Supports optional FiLM conditioning (input-dependent
    kernels).  The recommended default for Hyena-ND models.

``MultiOmegaSIRENKernelND``
    SIREN kernel with a per-row ``omega_0`` in the first layer, allowing the
    model to represent multiple frequency bands simultaneously.

``BlockDiagonalMultiOmegaSIRENKernelND``
    Multi-omega SIREN with block-diagonal weight masking at init, so each
    frequency block starts as an independent narrow-band SIREN.

``LearnableOmegaSIRENKernelND``
    SIREN kernel whose per-row ``omega_0`` multiplier is a learnable parameter
    (clamped to a configurable range), enabling frequency adaptation during
    training.

``BlockDiagonalLearnableOmegaSIRENKernelND``
    Combines block-diagonal MLP init with learnable per-row omega scaling —
    the most expressive variant in the family.

For test, please run:
    PYTHONPATH=. python nvsubquadratic/modules/kernels_nd.py

"""

import math
from collections.abc import Sequence
from typing import Callable

import torch
import torch.nn.functional as torch_F
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


def _normalize_l_cache(L_cache: int | Sequence[int], data_dim: int) -> tuple[int, ...]:
    """Broadcast ``L_cache`` to a per-axis tuple of positive ints (>= 2).

    Accepts a single int (broadcast to all axes) or a sequence of ints of
    length ``data_dim``.  Used by the kernel and positional-embedding
    classes so each spatial axis can have its own grid resolution
    (anisotropic kernel grid).  Booleans are explicitly rejected because
    ``True``/``False`` would silently broadcast to ``(1,)*data_dim``,
    which would then fail the lower-bound check below.

    Args:
        L_cache: Scalar cache extent or per-axis sequence.
        data_dim: Number of spatial dimensions.

    Returns:
        Tuple of length ``data_dim`` with one positive int per axis.

    Raises:
        TypeError: If ``L_cache`` is a ``bool`` (which would silently cast to
            ``0`` or ``1`` and fail the minimum-value check), or if it is
            neither an ``int`` nor a sequence of ints.
        ValueError: If ``L_cache`` is a sequence whose length differs from
            ``data_dim``, or if any value in the resulting tuple is less
            than 2 (required because the grid uses ``linspace(-1, 1, 2*L-1)``
            which needs at least 3 points, and the step ``1/(L-1)`` is
            undefined at ``L=1``).
    """
    if isinstance(L_cache, bool):
        raise TypeError("L_cache must be an int or a sequence of ints, got bool")
    if isinstance(L_cache, int):
        per_axis: tuple[int, ...] = (int(L_cache),) * data_dim
    elif isinstance(L_cache, Sequence) and not isinstance(L_cache, (str, bytes)):
        per_axis = tuple(int(v) for v in L_cache)
        if len(per_axis) != data_dim:
            raise ValueError(f"L_cache sequence must have length data_dim={data_dim}, got length {len(per_axis)}")
    else:
        raise TypeError(f"L_cache must be an int or a sequence of ints, got {type(L_cache).__name__}")
    for L in per_axis:
        # L must be >= 2 because step_size = 1/(L-1) and the cache uses
        # ``linspace(-1, 1, 2*L - 1)`` (which needs >= 3 points).
        if L < 2:
            raise ValueError(f"L_cache values must be >= 2, got {L}")
    return per_axis


class RandomFourierPositionalEmbeddingND(torch.nn.Module):
    """N-dimensional positional embedding using Random Fourier Features (RFF).

    Mathematical form
    -----------------
    Given a coordinate grid ``x`` of shape ``[1, *spatial_dims, data_dim]`` with
    values normalised to ``[-1, 1]`` per axis, the embedding is:

        phi(x) = [ cos(W x + b), sin(W x + b) ]   shape [..., embedding_dim]

    where:

    * ``W`` is the first-layer weight matrix of shape
      ``[embedding_dim//2, data_dim]``, drawn once at construction from
      ``N(0, (2*pi*omega_0)^2)`` and then **frozen** (not trained).
    * ``b`` is a bias vector of shape ``[embedding_dim//2]``, initialised to
      zero.  It is also frozen.
    * The concatenation of cosine and sine doubles the embedding dimension.

    The resulting features approximate the feature map of a stationary RBF
    (Gaussian) kernel with bandwidth ``omega_0`` — the larger ``omega_0``, the
    higher the dominant spatial frequency encoded in the embedding.

    Grid caching
    ------------
    To avoid rebuilding the meshgrid on every forward pass, the module
    maintains a ``grid_cache`` buffer (a pre-computed coordinate tensor of
    shape ``[1, 2*L_0-1, ..., 2*L_{d-1}-1, data_dim]`` in float32).  On each
    forward call the central ``[2*seq_len_i - 1]`` points are sliced per axis.
    When a larger ``seq_len`` is seen at runtime the cache grows automatically
    via ``_maybe_extend_grid_cache``, preserving the original step size on
    each axis.

    Note: the ``W`` and ``b`` parameters have ``_no_weight_decay = True`` set
    so that any weight-decay optimizer does not shrink the random projection.

    Attributes:
        data_dim (int): Number of spatial / temporal input dimensions.
        embedding_dim (int): Output embedding size (must be even; split equally
            between cos and sin features).
        L_cache_per_axis (tuple[int, ...]): Current per-axis cache extents.
            May grow at runtime; the original value at construction is stored in
            ``self.L_cache``.
        L_cache (int | Sequence[int]): Original ``L_cache`` argument (for
            diagnostics and external read-back).
        omega_0 (float): Bandwidth / frequency scaling factor used for weight
            init and for diagnostics.
        use_bias (bool): Whether a bias is present in the linear projection.
        linear (torch.nn.Linear): The frozen random frequency projection
            ``W`` (and optionally ``b``), shape ``[embedding_dim//2, data_dim]``.
        grid_cache (torch.Tensor): Non-persistent float32 buffer of shape
            ``[1, 2*L_0-1, ..., 2*L_{d-1}-1, data_dim]``.
        step_sizes (tuple[float, ...]): Per-axis grid step
            ``1/(L_i - 1)`` at construction; used by cache extensions to
            keep spacing constant.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        L_cache: int | Sequence[int],
        omega_0: float,
        use_bias: bool = True,
    ):
        """Initialize the RandomFourierPositionalEmbeddingND.

        Args:
            data_dim: Dimension of input data.
            embedding_dim: Dimensionality of the positional embedding. Must be even.
            L_cache: Number of cached time steps per axis. Either a scalar int
                (broadcast to every axis, isotropic grid) or a sequence of
                length ``data_dim`` (one extent per spatial axis, anisotropic
                grid).  The cached grid then has shape
                ``(1, 2*L_0 - 1, ..., 2*L_{d-1} - 1, data_dim)``.
            omega_0: Frequency scaling factor for the Fourier features.
            use_bias: Whether to use a bias term in the linear layer.

        Raises:
            ValueError: If `embedding_dim` is not an even number.
        """
        if embedding_dim % 2 != 0:
            raise ValueError(f"emb_dim must be even. Current {embedding_dim}")

        super().__init__()
        self.data_dim = data_dim
        self.embedding_dim = embedding_dim
        # Canonical per-axis form; ``L_cache`` is preserved as the input
        # value for diagnostics / external read-back.
        self.L_cache_per_axis = _normalize_l_cache(L_cache, data_dim)
        self.L_cache = L_cache
        self.omega_0 = omega_0
        self.use_bias = use_bias

        # Construct linear projection
        linear_out_channels = embedding_dim // 2
        self.linear = torch.nn.Linear(in_features=data_dim, out_features=linear_out_channels, bias=use_bias)

        # Initialize linear projection to be normal with mean 0 and std 2 * pi * omega_0.
        self.linear.weight.data.normal_(mean=0.0, std=2 * torch.pi * self.omega_0)
        if self.linear.bias is not None:
            torch.nn.init.constant_(self.linear.bias, 0.0)

        # Construct grid cache: per-axis ``linspace(-1, 1, 2*L - 1)`` so each
        # axis spans its own resolution at full [-1, 1] extent.
        # TODO(@dwromero): We must make sure that the grid_cache is kept in float32.
        with torch.inference_mode(False):
            with torch.no_grad():
                grid_cache = self._build_grid_cache(self.L_cache_per_axis)
        self.register_buffer("grid_cache", grid_cache, persistent=False)

        # Per-axis step size: kept frozen at the *original* L_cache so that
        # subsequent runtime cache extensions preserve the spacing they were
        # built with (a longer sequence still uses the same step).
        self.step_sizes = tuple(1.0 / (L - 1) for L in self.L_cache_per_axis)

        # Add ._no_weight_decay flag to all parameters to avoid weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    @staticmethod
    def _build_grid_cache(
        L_per_axis: Sequence[int],
        max_limits: Sequence[float] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build the cached coordinate grid for the given per-axis lengths.

        Each axis spans ``[-max_limit_i, +max_limit_i]`` sampled at
        ``2 * L_i - 1`` points (default ``max_limit_i = 1.0`` at construction,
        possibly larger after a runtime extension to keep step size constant).

        Args:
            L_per_axis: Per-axis cache extents.  The number of grid points
                along axis ``i`` is ``2 * L_per_axis[i] - 1``.
            max_limits: Per-axis coordinate limits; axis ``i`` spans
                ``[-max_limits[i], +max_limits[i]]``.  Defaults to ``1.0``
                on all axes (the standard ``[-1, 1]`` normalised range).
            device: Target device for the returned tensor.  Defaults to CPU.

        Returns:
            Float32 tensor of shape
            ``[1, 2*L_0-1, ..., 2*L_{d-1}-1, data_dim]`` representing the
            coordinate meshgrid, with a leading batch dimension of 1.
        """
        if max_limits is None:
            max_limits = (1.0,) * len(L_per_axis)
        ts = [
            torch.linspace(-float(m), float(m), 2 * L - 1, device=device, dtype=torch.float32)
            for L, m in zip(L_per_axis, max_limits)
        ]
        return rearrange(torch.stack(torch.meshgrid(*ts, indexing="ij"), dim=-1), "... -> 1 ...")

    def _maybe_extend_grid_cache(self, seq_lens: tuple[int, ...]) -> None:
        """Grow ``grid_cache`` per axis whenever any axis exceeds its cache.

        Each axis is extended independently while preserving its original
        step size: along an extended axis the new range becomes
        ``[-max_limit, +max_limit]`` with
        ``max_limit = 1.0 + step_size * (seq_len - L_cache_orig)``.  Axes
        that are not extended are rebuilt at their existing extent so the
        cache remains a single rectangular tensor.

        Args:
            seq_lens: Requested per-axis output sequence lengths.  Any axis
                where ``seq_lens[i] > self.L_cache_per_axis[i]`` triggers a
                cache extension for that axis.

        Returns:
            None.  Modifies ``self.grid_cache`` and
            ``self.L_cache_per_axis`` in-place when an extension is needed.
        """
        if all(L >= sl for L, sl in zip(self.L_cache_per_axis, seq_lens)):
            return
        new_L_per_axis = tuple(max(L, sl) for L, sl in zip(self.L_cache_per_axis, seq_lens))
        max_limits = tuple(
            1.0 + step * (new_L - L) for L, new_L, step in zip(self.L_cache_per_axis, new_L_per_axis, self.step_sizes)
        )
        with torch.inference_mode(False):
            with torch.no_grad():
                self.grid_cache = self._build_grid_cache(
                    new_L_per_axis,
                    max_limits=max_limits,
                    device=self.grid_cache.device,
                )
        self.L_cache_per_axis = new_L_per_axis

    def forward(self, seq_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the RFF positional embeddings for a given spatial grid.

        Args:
            seq_lens: Per-axis output sequence lengths.  Length must equal
                ``self.data_dim``.  For example, for a 2D signal of height H
                and width W, pass ``(H, W)``.

        Returns:
            tuple:
                - torch.Tensor: The positional embeddings,
                  ``[cos(Wx+b), sin(Wx+b)]`` concatenated along the last axis.
                  Shape ``[1, *spatial_dims, embedding_dim]``.
                - torch.Tensor: The coordinate grid of positions normalised to
                  ``[-1, 1]`` per axis.  Shape ``[1, *spatial_dims, data_dim]``.

        Raises:
            AssertionError: If ``len(seq_lens) != self.data_dim``.
            AssertionError: If ``self.grid_cache`` is not ``float32``.
        """
        # Check that the sequence lengths are of the correct length.
        assert len(seq_lens) == self.data_dim, (
            f"seq_lens must be of length {self.data_dim}. Current length: {len(seq_lens)}"
        )

        # Per-axis cache extension: any axis whose runtime seq_len exceeds
        # its current cache triggers a rebuild that grows that axis only.
        self._maybe_extend_grid_cache(tuple(seq_lens))

        # Ensure that the cached positions tensor has the correct data type.
        assert self.grid_cache.dtype == torch.float32, (
            f"grid_cache must be float32. At lower precision, indexes will be merged together. Current dtype: {self.grid_cache.dtype}"
        )

        # Per-axis offsets: each axis is centered, so the slice picks the
        # central ``2 * seq_len - 1`` points of that axis's cache.
        offsets = [L - sl for L, sl in zip(self.L_cache_per_axis, seq_lens)]

        # Construct slice objects to index the grid cache.
        slices = [slice(offset, offset + (seq_len * 2) - 1) for offset, seq_len in zip(offsets, seq_lens)]
        grid = self.grid_cache[:, *slices]

        # Compute the linear projection. We need to ensure that the linear projection is done in float32.
        linear_dtype = self.linear.weight.dtype
        if linear_dtype != torch.float32:
            out = torch_F.linear(
                grid,
                self.linear.weight.to(torch.float32),
                self.linear.bias.to(torch.float32) if self.linear.bias is not None else None,
            ).to(linear_dtype)
        else:
            out = self.linear(grid)

        # Concatenate the sine and cosine values.
        return torch.cat([torch.cos(out), torch.sin(out)], dim=-1), grid


class RandomFourierKernelND(torch.nn.Module):
    """Learned convolutional kernel parametrised via Random Fourier Features and an MLP.

    Mathematical form
    -----------------
    The kernel at grid coordinate ``x`` is:

        k(x) = Linear_out( MLP( phi(x) ) )

    where:

    * ``phi(x) = [cos(W x + b), sin(W x + b)]`` is the RFF positional
      embedding (see ``RandomFourierPositionalEmbeddingND``).
    * ``MLP`` is a stack of ``num_layers - 1`` fully-connected layers each
      followed by the ``nonlinear_cfg`` activation.
    * ``Linear_out`` is a final linear layer that maps to ``out_dim`` channels.

    The output is a kernel tensor of shape ``[1, *spatial_dims, out_dim]``
    suitable for passing directly to the FFT convolution primitives in
    ``nvsubquadratic.ops`` (after rearranging to channels-first layout via the
    consuming ``CKConvND`` module).

    Hyperparameters controlling bandwidth / smoothness
    ---------------------------------------------------
    * ``omega_0``: Controls the frequency content of the RFF features.  Higher
      values concentrate the kernel's spectral energy at higher frequencies,
      producing a narrower, higher-bandwidth filter.  Typical range: 1.0–100.0.
    * ``mlp_hidden_dim``: Width of the hidden MLP layers.  Larger values allow
      more expressive kernel shapes.
    * ``num_layers``: Depth of the MLP.  Must be >= 2 (one hidden layer +
      output layer minimum).

    Initialisation
    --------------
    * Hidden MLP layers are initialised with the user-supplied ``init_method``
      if provided; otherwise use PyTorch defaults.
    * The output layer applies **Wang initialisation**: weights are scaled by
      ``sqrt(1 / kernel_volume)`` where ``kernel_volume = prod(L_cache_per_axis)``
      (collapses to ``L_cache**data_dim`` for an isotropic grid).  This
      normalises the kernel's initial energy to be independent of grid size.

    Attributes:
        out_dim (int): Number of output channels (kernel depth).
        data_dim (int): Number of spatial / temporal input dimensions.
        mlp_hidden_dim (int): Hidden width of the MLP.
        num_layers (int): Total number of MLP layers (>= 2).
        embedding_dim (int): RFF embedding dimensionality (must be even).
        omega_0 (float): Bandwidth scaling factor for the positional embedding.
        L_cache_per_axis (tuple[int, ...]): Per-axis cache extents (canonical form).
        L_cache (int | Sequence[int]): Original ``L_cache`` argument (diagnostics).
        positional_embedding (RandomFourierPositionalEmbeddingND): RFF encoder.
        kernel_network (torch.nn.Sequential): Hidden MLP layers.
        out_linear (torch.nn.Linear): Final projection to ``out_dim`` channels.
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        omega_0: float,
        L_cache: int | Sequence[int],
        use_bias: bool,
        nonlinear_cfg: LazyConfig,
        init_method: (Callable[[int], Callable[[torch.Tensor], torch.Tensor]] | None) = None,
    ):
        """Initialize the RandomFourierKernelND class.

        Args:
            out_dim: Number of output channels for the generated kernel.
            data_dim: Number of spatial/temporal input dimensions (size of coordinate vector).
            mlp_hidden_dim: Hidden width of the network.
            num_layers: Total number of layers including the first and hidden layers (>= 2).
            embedding_dim: Dimensionality of the positional embeddings.
            omega_0: Frequency scaling factor for the positional embeddings.
            L_cache: Per-axis cache extents.  Either a scalar int (broadcast
                to every axis, isotropic grid of size ``L_cache**data_dim``)
                or a sequence of length ``data_dim`` (anisotropic grid of
                size ``prod(L_cache)``).  The Wang init below uses the
                product of per-axis extents so it stays correct in both
                cases.
            use_bias: Whether to use bias in the network and embedding layers.
            nonlinear_cfg: Configuration for the nonlinear activation function.
            init_method: Optional initialization method for the kernel network.
        """
        super().__init__()

        self.out_dim = out_dim
        self.data_dim = data_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.omega_0 = omega_0
        # Canonical per-axis form drives the Wang init below; keep the
        # original input on ``self.L_cache`` for diagnostics.
        self.L_cache_per_axis = _normalize_l_cache(L_cache, data_dim)
        self.L_cache = L_cache

        # Construct positional embedding
        self.positional_embedding = RandomFourierPositionalEmbeddingND(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            omega_0=omega_0,
            L_cache=L_cache,
            use_bias=use_bias,
        )

        # Construct kernel network
        self.kernel_network = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, mlp_hidden_dim, bias=use_bias),
            instantiate(nonlinear_cfg),
        )
        for _ in range(num_layers - 2):
            self.kernel_network.append(torch.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=use_bias))
            self.kernel_network.append(instantiate(nonlinear_cfg))

        # Construct output linear layer of the kernel network
        self.out_linear = torch.nn.Linear(mlp_hidden_dim, out_dim, bias=use_bias)

        # Initialize layers and output layer
        if init_method is not None:
            for layer in self.kernel_network:
                if isinstance(layer, torch.nn.Linear):
                    init_method(mlp_hidden_dim)(layer.weight.data)
        if init_method is not None:
            init_method(out_dim)(self.out_linear.weight)
        # Add Wang initialization to the output layer (to account for the fact that the output is used as a convolutional kernel)
        # This boils down to modulating the weight of the output layer by the expected kernel size, which is the
        # product of the per-axis cache extents (degenerates to ``L_cache**data_dim`` for an isotropic grid).
        with torch.no_grad():
            kernel_volume = math.prod(self.L_cache_per_axis)
            self.out_linear.weight.data *= math.sqrt(1.0 / kernel_volume)

    def forward(self, seq_lens: tuple[int, ...], conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the RFF kernel for a given grid of spatial dimensions.

        Args:
            seq_lens: Per-axis output sequence lengths.  Length must equal
                ``self.data_dim``.
            conditioning: Unused.  Accepted for API compatibility with
                FiLM-enabled kernels (e.g. ``SIRENKernelND`` with
                ``film_cfg``).

        Returns:
            tuple:
                - torch.Tensor: Kernel values of shape
                  ``[1, *spatial_dims, out_dim]``.
                - torch.Tensor: Coordinate grid of shape
                  ``[1, *spatial_dims, data_dim]``.
        """
        # Generate positional embeddings and corresponding grid values
        pos_emb, grid = self.positional_embedding(seq_lens)
        # Pass embeddings through the kernel network and output layer
        kernel = self.out_linear(self.kernel_network(pos_emb))
        return kernel, grid


def _init_siren_weights(layer: torch.nn.Linear, is_first_layer: bool, w0: float) -> None:
    """Initialise a ``nn.Linear`` layer with the SIREN uniform distribution.

    From Sitzmann et al. 2020 ("Implicit Neural Representations with Periodic
    Activation Functions"), the weight init that keeps the distribution of
    pre-activations stationary across layers of ``sin`` activations is:

        first layer:  W ~ U(-1/d, +1/d)  (note: scaled by 2pi*w0 below)
        hidden layers: W ~ U(-sqrt(6/d)/(2pi*w0), +sqrt(6/d)/(2pi*w0))

    where ``d = layer.in_features``.  The factor ``2*pi*w0`` is absorbed into
    the init bound so that the frequency content of each layer matches ``w0``
    at initialisation without applying the factor at every forward pass.

    Bias is always zero-initialised, matching the SIREN paper's convention.

    Args:
        layer: The ``nn.Linear`` layer to initialise (modified in-place).
        is_first_layer: If True, use the first-layer bound ``1/d``; otherwise
            use the hidden-layer bound ``sqrt(6/d)/(2pi*w0)``.
        w0: Frequency scaling factor.  Larger values produce higher-frequency
            features at initialisation.

    Returns:
        None.  Modifies ``layer.weight`` (and ``layer.bias`` if present) in-place.
    """
    with torch.no_grad():
        # Compute the bound for the weights based on the SIREN paper.
        in_features = layer.in_features
        if is_first_layer:
            bound = 1.0 / in_features
        else:
            bound = math.sqrt(6.0 / in_features) / float(2.0 * math.pi * w0)

        # Scale the bound by the frequency scaling factor.
        # Instead of having w_0 being applied during the nonlinearity, we initialize the weights
        # to the expected values, and let the network decide the frequency scaling.
        bound = 2.0 * math.pi * w0 * bound

        # Apply the bound to the weights.
        layer.weight.uniform_(-bound, bound)

        # Initialize the bias to 0.
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


class Sine(torch.nn.Module):
    """Sine activation function used in SIREN networks.

    Computes ``sin(x)`` element-wise.  No frequency scaling is applied here;
    the ``omega_0`` / ``hidden_omega_0`` factors are absorbed into the weight
    initialisation (see ``_init_siren_weights``) so that the effective
    frequency at each layer is determined at init without altering the forward
    pass arithmetic.

    This design choice follows the SIREN paper and keeps the forward pass
    free of any scalar multiplications that might interact poorly with
    mixed-precision training.

    Attributes:
        (none beyond the base nn.Module bookkeeping)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sine element-wise.

        Args:
            x: Input tensor of any shape and dtype.

        Returns:
            Tensor of the same shape and dtype as ``x`` with values
            ``sin(x_i)`` for each element ``x_i``.
        """
        return torch.sin(x)


class SIRENPositionalEmbeddingND(torch.nn.Module):
    """N-dimensional positional embedding using a SIREN first layer.

    Mathematical form
    -----------------
    Given a coordinate grid ``x`` of shape ``[1, *spatial_dims, data_dim]``
    with values normalised to ``[-1, 1]`` per axis, the embedding is:

        phi(x) = sin( W x + b )   shape [..., embedding_dim]

    where:

    * ``W`` is a learned weight matrix of shape ``[embedding_dim, data_dim]``,
      initialised from ``U(-2*pi*omega_0/d, +2*pi*omega_0/d)`` (first-layer
      SIREN bound, see ``_init_siren_weights``).  Unlike the RFF counterpart,
      this weight **is** trainable.
    * ``b`` is an optional bias vector, zero-initialised.

    The ``omega_0`` parameter controls the frequency content at init: higher
    values bias the embedding toward higher spatial frequencies, giving the
    downstream MLP a head-start in representing rapid kernel variations.
    During training the weight can drift away from the init distribution.

    Grid caching
    ------------
    Identical to ``RandomFourierPositionalEmbeddingND``: a coordinate tensor of
    shape ``[1, 2*L_0-1, ..., 2*L_{d-1}-1, data_dim]`` is pre-computed in
    float32 and cached as a non-persistent buffer.  The forward pass slices the
    central ``[2*seq_len_i - 1]`` entries per axis and calls
    ``_maybe_extend_grid_cache`` if any axis is larger than the current cache.

    Note: the linear projection is forced to float32 internally (even under
    autocast) to avoid quantisation errors in the SIREN's high-frequency sine.
    The output is cast back to the weight's dtype before return.

    Attributes:
        data_dim (int): Number of spatial / temporal input dimensions.
        embedding_dim (int): Output embedding size.
        L_cache_per_axis (tuple[int, ...]): Current per-axis cache extents.
        L_cache (int | Sequence[int]): Original ``L_cache`` argument (diagnostics).
        omega_0 (float): Frequency scaling factor used for SIREN init.
        use_bias (bool): Whether a bias is present in the linear projection.
        linear (torch.nn.Linear): Trainable SIREN first-layer projection,
            shape ``[embedding_dim, data_dim]``.
        grid_cache (torch.Tensor): Non-persistent float32 buffer of shape
            ``[1, 2*L_0-1, ..., 2*L_{d-1}-1, data_dim]``.
        step_sizes (tuple[float, ...]): Per-axis grid step ``1/(L_i - 1)``
            at construction; kept frozen for consistent cache extension.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        L_cache: int | Sequence[int],
        omega_0: float,
        use_bias: bool = True,
    ):
        """Initialize the SIRENPositionalEmbeddingND class.

        Args:
            data_dim: Dimension of input data.
            embedding_dim: Dimensionality of the positional embedding.
            L_cache: Per-axis cache extents.  Either a scalar int (broadcast
                to every axis, isotropic grid) or a sequence of length
                ``data_dim`` (one extent per spatial axis, anisotropic grid).
                The cached grid then has shape
                ``(1, 2*L_0 - 1, ..., 2*L_{d-1} - 1, data_dim)`` and each axis
                spans ``[-1, 1]`` at its own resolution.
            omega_0: Frequency scaling factor for the Fourier features.
            use_bias: Whether to use a bias term in the linear layer.
        """
        super().__init__()
        self.data_dim = data_dim
        self.embedding_dim = embedding_dim
        # Canonical per-axis form; ``self.L_cache`` retains the input value
        # so external callers (e.g. CKConv) can read back what was passed.
        self.L_cache_per_axis = _normalize_l_cache(L_cache, data_dim)
        self.L_cache = L_cache
        self.omega_0 = omega_0
        self.use_bias = use_bias

        # Construct linear projection
        self.linear = torch.nn.Linear(in_features=data_dim, out_features=embedding_dim, bias=use_bias)

        # Initialize linear projection following SIREN initialization.
        _init_siren_weights(self.linear, is_first_layer=True, w0=self.omega_0)

        # Construct grid cache: per-axis ``linspace(-1, 1, 2*L_i - 1)`` so that
        # every axis spans the full [-1, 1] range at its own resolution.
        with torch.inference_mode(False):
            with torch.no_grad():
                grid_cache = self._build_grid_cache(self.L_cache_per_axis)
        self.register_buffer("grid_cache", grid_cache, persistent=False)

        # Per-axis step size: kept frozen at the *original* L_cache so a
        # later runtime extension preserves the spacing it was built with.
        self.step_sizes = tuple(1.0 / (L - 1) for L in self.L_cache_per_axis)

        # Add ._no_weight_decay flag to all parameters to avoid weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    @staticmethod
    def _build_grid_cache(
        L_per_axis: Sequence[int],
        max_limits: Sequence[float] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build the cached coordinate grid for the given per-axis lengths.

        Each axis spans ``[-max_limit_i, +max_limit_i]`` sampled at
        ``2 * L_i - 1`` points.  At construction every ``max_limit_i`` is
        ``1.0``; runtime extensions can pass per-axis values larger than 1
        to preserve the original step size on extended axes.

        Args:
            L_per_axis: Per-axis cache extents.  The number of grid points
                along axis ``i`` is ``2 * L_per_axis[i] - 1``.
            max_limits: Per-axis coordinate limits; axis ``i`` spans
                ``[-max_limits[i], +max_limits[i]]``.  Defaults to ``1.0``
                on all axes.
            device: Target device for the returned tensor.  Defaults to CPU.

        Returns:
            Float32 tensor of shape
            ``[1, 2*L_0-1, ..., 2*L_{d-1}-1, data_dim]`` representing the
            coordinate meshgrid, with a leading batch dimension of 1.
        """
        if max_limits is None:
            max_limits = (1.0,) * len(L_per_axis)
        ts = [
            torch.linspace(-float(m), float(m), 2 * L - 1, device=device, dtype=torch.float32)
            for L, m in zip(L_per_axis, max_limits)
        ]
        return rearrange(torch.stack(torch.meshgrid(*ts, indexing="ij"), dim=-1), "... -> 1 ...")

    def _maybe_extend_grid_cache(self, seq_lens: tuple[int, ...]) -> None:
        """Grow ``grid_cache`` per axis whenever any axis exceeds its cache.

        Each axis is extended independently while preserving its original
        step size: along an extended axis the new range becomes
        ``[-max_limit, +max_limit]`` with
        ``max_limit = 1.0 + step_size * (seq_len - L_cache_orig)``.  Axes
        that already cover their requested ``seq_len`` are rebuilt at their
        existing extent so the cache stays a single rectangular tensor.

        Args:
            seq_lens: Requested per-axis output sequence lengths.  Any axis
                where ``seq_lens[i] > self.L_cache_per_axis[i]`` triggers a
                cache extension for that axis.

        Returns:
            None.  Modifies ``self.grid_cache`` and
            ``self.L_cache_per_axis`` in-place when an extension is needed.
        """
        if all(L >= sl for L, sl in zip(self.L_cache_per_axis, seq_lens)):
            return
        new_L_per_axis = tuple(max(L, sl) for L, sl in zip(self.L_cache_per_axis, seq_lens))
        max_limits = tuple(
            1.0 + step * (new_L - L) for L, new_L, step in zip(self.L_cache_per_axis, new_L_per_axis, self.step_sizes)
        )
        with torch.inference_mode(False):
            with torch.no_grad():
                self.grid_cache = self._build_grid_cache(
                    new_L_per_axis,
                    max_limits=max_limits,
                    device=self.grid_cache.device,
                )
        self.L_cache_per_axis = new_L_per_axis

    def forward(self, seq_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the SIREN positional embeddings for a given spatial grid.

        Args:
            seq_lens: Per-axis output sequence lengths.  Length must equal
                ``self.data_dim``.  For example, for a 2D signal of height H
                and width W, pass ``(H, W)``.

        Returns:
            tuple:
                - torch.Tensor: The positional embeddings ``sin(W x + b)``,
                  where the linear projection is computed in float32 and the
                  result is cast back to the weight dtype.
                  Shape ``[1, *spatial_dims, embedding_dim]``.
                - torch.Tensor: The coordinate grid of positions normalised to
                  ``[-1, 1]`` per axis.  Shape ``[1, *spatial_dims, data_dim]``.

        Raises:
            AssertionError: If ``len(seq_lens) != self.data_dim``.
            AssertionError: If ``self.grid_cache`` is not ``float32``.
        """
        # Check that the sequence lengths are of the correct length.
        assert len(seq_lens) == self.data_dim, (
            f"seq_lens must be of length {self.data_dim}. Current length: {len(seq_lens)}"
        )

        # Per-axis cache extension: any axis whose runtime seq_len exceeds
        # its current cache triggers a rebuild that grows that axis only.
        self._maybe_extend_grid_cache(tuple(seq_lens))

        # Ensure that the cached positions tensor has the correct data type.
        assert self.grid_cache.dtype == torch.float32, (
            f"grid_cache must be float32. At lower precision, indexes will be merged together. Current dtype: {self.grid_cache.dtype}"
        )

        # Per-axis offsets: each axis is centered, so the slice picks the
        # central ``2 * seq_len - 1`` points of that axis's cache.
        offsets = [L - sl for L, sl in zip(self.L_cache_per_axis, seq_lens)]

        # Construct slice objects to index the grid cache.
        slices = [slice(offset, offset + (seq_len * 2) - 1) for offset, seq_len in zip(offsets, seq_lens)]
        grid = self.grid_cache[:, *slices]  # type: ignore

        # Compute the linear projection.
        linear_dtype = self.linear.weight.dtype
        if linear_dtype != torch.float32:
            out = torch_F.linear(
                grid,
                self.linear.weight.to(torch.float32),
                self.linear.bias.to(torch.float32) if self.linear.bias is not None else None,
            ).to(linear_dtype)
        else:
            out = self.linear(grid)

        # Apply sine activation in place.
        return out.sin_(), grid


class SIRENKernelND(torch.nn.Module):
    """Convolutional kernel parametrised by a SIREN (sinusoidal representation network) MLP.

    Mathematical form
    -----------------
    The kernel at coordinate ``x`` is:

        k(x) = Linear_out( SIREN_MLP( phi(x) ) )

    where:

    * ``phi(x) = sin(W_0 x + b_0)`` is the SIREN positional embedding
      (``SIRENPositionalEmbeddingND``) with first-layer frequency ``omega_0``.
    * ``SIREN_MLP`` is a stack of ``num_layers - 1`` layers, each computing
      ``sin(W_i h + b_i)`` with weights initialised at frequency
      ``hidden_omega_0``.
    * ``Linear_out`` is a linear readout to ``out_dim`` channels, scaled by
      Wang init (``sqrt(1 / kernel_volume)``) to normalise initial kernel energy.

    The full pipeline (without FiLM conditioning) is therefore:

        h_0 = sin(W_0 x + b_0)                     -- pos embedding
        h_i = sin(W_i h_{i-1} + b_i)  for i=1..N-1 -- hidden layers
        k   = W_out h_{N-1} + b_out                 -- output layer

    Hyperparameters controlling bandwidth / smoothness
    ---------------------------------------------------
    * ``omega_0``: Frequency of the first SIREN layer.  Higher values produce
      higher-frequency positional features at init.  Typical range: 1.0–30.0.
    * ``hidden_omega_0``: Frequency of the hidden SIREN layers.  Usually set to
      1.0 (default) following the recommendation in the SIREN paper.
    * ``mlp_hidden_dim``: Width of all hidden layers; wider networks can
      express more complex kernel shapes.

    FiLM conditioning
    -----------------
    When ``film_cfg`` is provided, a ``KernelFiLMGenerator`` is instantiated
    and called on the ``conditioning`` tensor (shape ``[B, C]``) to produce
    per-layer ``(gamma, beta)`` pairs (each of shape ``[B, mlp_hidden_dim]``).
    The hidden activations are then modulated as:

        h_i <- gamma_i * h_i + beta_i

    When ``film_after_pos_embed=True``, an *extra* FiLM layer is applied to
    the output of the positional embedding (before the first hidden layer),
    making the positional features themselves input-dependent.  This requires
    ``embedding_dim == mlp_hidden_dim`` and one additional film layer in the
    generator (``num_film_layers = num_layers``).

    When conditioning is present, the output kernel has shape
    ``[B, *spatial, out_dim]``; otherwise it is ``[1, *spatial, out_dim]``.

    Initialisation
    --------------
    * All ``hidden_linears`` are SIREN-initialised with ``hidden_omega_0``.
    * ``out_linear`` is SIREN-initialised with ``hidden_omega_0``, then
      additionally Wang-scaled by ``sqrt(1 / prod(L_cache_per_axis))``.
      This "Wang init" (from the CKConv paper, Romero et al. 2021) divides
      the output layer's weights by the square root of the total grid volume
      (``L_cache**data_dim`` for isotropic grids), so the initial filter's
      L2 energy is independent of the grid resolution.
    * Hidden linear weights and output bias get ``_no_weight_decay = True``
      so that weight-decay optimizers do not destroy the SIREN spectrum.

    Attributes:
        out_dim (int): Number of output channels (kernel depth).
        data_dim (int): Number of spatial / temporal input dimensions.
        mlp_hidden_dim (int): Hidden width of the SIREN MLP.
        num_layers (int): Total number of SIREN layers (>= 2).
        embedding_dim (int): SIREN positional-embedding dimensionality.
        omega_0 (float): First-layer frequency scaling.
        hidden_omega_0 (float): Hidden-layer frequency scaling.
        L_cache_per_axis (tuple[int, ...]): Per-axis cache extents (canonical form).
        L_cache (int | Sequence[int]): Original ``L_cache`` argument (diagnostics).
        positional_embedding (SIRENPositionalEmbeddingND): First SIREN layer.
        hidden_linears (torch.nn.ModuleList): Hidden linear layers (length
            ``num_layers - 1``).  Interleaved with ``self.sine`` in the forward
            pass; stored separately so FiLM can be inserted between them.
        sine (Sine): Shared sine activation applied after every hidden linear.
        out_linear (torch.nn.Linear): Final readout to ``out_dim`` channels.
        num_film_layers (int): Number of hidden layers eligible for FiLM
            modulation (equal to ``len(hidden_linears)``).
        film_generator: ``KernelFiLMGenerator`` instance or ``None``.
        film_after_pos_embed (bool): Whether the first FiLM pair modulates the
            positional embedding output.

    Args:
        out_dim: Number of output channels for the generated kernel.
        data_dim: Number of spatial/temporal input dimensions (size of coordinate vector).
        mlp_hidden_dim: Hidden width of the SIREN network.
        num_layers: Total number of layers including the first and hidden layers (>= 2).
        embedding_dim: Dimensionality of the SIREN positional embedding.
        omega_0: Frequency scaling for the first SIREN layer.
        L_cache: Cache extent controlling the maximum supported grid size before
            cache growth.  Either a scalar int (isotropic, same extent on all axes)
            or a sequence of length ``data_dim`` (anisotropic, per-axis extents).
        use_bias: Whether to include biases in linear layers.
        hidden_omega_0: Frequency scaling for subsequent SIREN layers (default 1.0).
        film_cfg: Optional LazyConfig for KernelFiLMGenerator. When provided, enables
            input-dependent FiLM conditioning of all hidden SIREN layers.
        film_after_pos_embed: If True, the first FiLM (gamma, beta) pair modulates
            the positional embedding *after* the sine activation.  Requires
            ``embedding_dim == mlp_hidden_dim`` and one extra FiLM layer in
            ``film_cfg`` (i.e. ``num_film_layers = num_layers - 1 + 1 = num_layers``).
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        omega_0: float,
        L_cache: int | Sequence[int],
        use_bias: bool,
        hidden_omega_0: float = 1.0,
        film_cfg: LazyConfig | None = None,
        film_after_pos_embed: bool = False,
    ):
        """Build SIREN MLP and optional FiLM conditioner."""
        super().__init__()
        self.film_after_pos_embed = film_after_pos_embed

        if film_after_pos_embed:
            assert embedding_dim == mlp_hidden_dim, (
                f"film_after_pos_embed requires embedding_dim == mlp_hidden_dim, "
                f"got {embedding_dim} != {mlp_hidden_dim}"
            )

        self.out_dim = out_dim
        self.data_dim = data_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.omega_0 = float(omega_0)
        self.hidden_omega_0 = float(hidden_omega_0)
        # Canonical per-axis form drives the Wang init below; ``self.L_cache``
        # retains the input value for diagnostics / external callers.
        self.L_cache_per_axis = _normalize_l_cache(L_cache, data_dim)
        self.L_cache = L_cache

        # Construct positional embedding
        self.positional_embedding = SIRENPositionalEmbeddingND(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            omega_0=omega_0,
            L_cache=L_cache,
            use_bias=use_bias,
        )

        # Construct kernel network as ModuleList of (Linear, Sine) pairs
        # so FiLM can be interleaved between layers.
        self.hidden_linears = torch.nn.ModuleList()
        self.hidden_linears.append(torch.nn.Linear(embedding_dim, mlp_hidden_dim, bias=use_bias))
        for _ in range(num_layers - 2):
            self.hidden_linears.append(torch.nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=use_bias))
        self.sine = Sine()

        # Number of hidden layers that can be FiLM-conditioned (all of them)
        self.num_film_layers = len(self.hidden_linears)

        # Construct output linear layer of the kernel network
        self.out_linear = torch.nn.Linear(mlp_hidden_dim, out_dim, bias=use_bias)

        # SIREN-initialize weights of the kernel network
        for linear in self.hidden_linears:
            _init_siren_weights(linear, is_first_layer=False, w0=self.hidden_omega_0)
        _init_siren_weights(self.out_linear, is_first_layer=False, w0=self.hidden_omega_0)
        # Add Wang initialization to the output layer (to account for the fact that the output is used as a
        # convolutional kernel).  The expected kernel size is the *product* of per-axis cache extents, which
        # collapses to ``L_cache**data_dim`` for an isotropic grid and stays correct for anisotropic ones.
        with torch.no_grad():
            kernel_volume = math.prod(self.L_cache_per_axis)
            self.out_linear.weight.data *= math.sqrt(1.0 / kernel_volume)

        # Add ._no_weight_decay flag to all parameters to avoid weight decay (except for self.out_linear weights)
        # Note that the positional embedding is already excluded from weight decay by the _no_weight_decay flag.
        for linear in self.hidden_linears:
            for param in linear.parameters():
                param._no_weight_decay = True
        if self.out_linear.bias is not None:
            self.out_linear.bias._no_weight_decay = True

        # Optional FiLM conditioning
        if film_cfg is not None:
            self.film_generator = instantiate(film_cfg)
            expected_film_layers = len(self.hidden_linears) + int(self.film_after_pos_embed)
            if self.film_generator.num_film_layers != expected_film_layers:
                raise ValueError(
                    f"film_generator.num_film_layers={self.film_generator.num_film_layers} "
                    f"does not match expected {expected_film_layers} "
                    f"(len(hidden_linears)={len(self.hidden_linears)} + "
                    f"int(film_after_pos_embed)={int(self.film_after_pos_embed)})"
                )
        else:
            self.film_generator = None

    def flop_count(self, grid_lens: tuple[int, ...], inference: bool = False) -> int:
        """Return an integer FLOP estimate for one kernel generation forward pass.

        At ``inference=True`` with no FiLM generator, returns 0 because the
        kernel is input-independent and can be precomputed once and cached.
        When a ``film_generator`` exists the kernel is input-dependent (via
        register-conditioned FiLM modulation) and must be recomputed on every
        forward pass regardless of the inference flag.

        Let ``G = prod(2 * L_i - 1 for L_i in grid_lens)`` be the total grid
        points.  FLOPs breakdown:

        - **Positional embedding** (:class:`SIRENPositionalEmbeddingND`):
          ``2 * G * data_dim * embedding_dim`` for the linear, plus
          ``G * embedding_dim`` for the ``sin`` activation.
        - **Hidden SIREN layers** (``len(self.hidden_linears) = num_layers - 1``).
          First layer: ``2 * G * embedding_dim * mlp_hidden_dim`` plus
          ``G * mlp_hidden_dim`` for the sin.  Each subsequent layer:
          ``2 * G * mlp_hidden_dim * mlp_hidden_dim`` plus
          ``G * mlp_hidden_dim`` for the sin.
        - **Output linear**: ``2 * G * mlp_hidden_dim * out_dim``.
        - **FiLM conditioning** (only when ``self.film_generator`` is set):
          the FiLM generator MLP costs ``self.film_generator.flop_count()``;
          applying ``gamma * h + beta`` per modulated layer costs
          ``2 * G * mlp_hidden_dim`` and is applied to each hidden layer
          (plus the positional embedding when ``film_after_pos_embed`` is
          ``True`` — which requires ``embedding_dim == mlp_hidden_dim``).

        Args:
            grid_lens: Per-axis output sequence lengths — the same tuple you
                would pass to ``forward`` as ``seq_lens``.  The total number
                of coordinate points the MLP processes is
                ``G = prod(2*L - 1 for L in grid_lens)``.
            inference: If True and no FiLM generator, return 0 (cacheable kernel).

        Returns:
            Total FLOPs as an integer.
        """
        has_film = self.film_generator is not None
        if inference and not has_film:
            return 0

        # Grid size: product of (2*L - 1) for each dimension
        G = 1
        for L in grid_lens:
            G *= 2 * L - 1

        flops = 0

        # 1. Positional embedding: Linear(data_dim -> embedding_dim) + sin
        flops += 2 * G * self.data_dim * self.embedding_dim
        flops += G * self.embedding_dim

        # 2. Hidden SIREN layers (iterate to get exact in/out dimensions)
        in_dim = self.embedding_dim
        for linear in self.hidden_linears:
            out_dim = linear.out_features
            flops += 2 * G * in_dim * out_dim  # Linear
            flops += G * out_dim  # sin activation
            in_dim = out_dim

        # 3. Output linear
        flops += 2 * G * self.mlp_hidden_dim * self.out_dim

        # 4. FiLM conditioning
        if has_film:
            flops += self.film_generator.flop_count()
            num_modulated = len(self.hidden_linears)
            if self.film_after_pos_embed:
                num_modulated += 1
            # gamma * h + beta = 2 elementwise ops per grid point per hidden_dim
            flops += num_modulated * 2 * G * self.mlp_hidden_dim

        return flops

    def forward(
        self, seq_lens: tuple[int, ...], conditioning: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the SIREN kernel for a given grid of spatial dimensions.

        Args:
            seq_lens: Lengths of the input grid for which to compute the positional embeddings.
            conditioning: Optional ``[B, C]`` conditioning vector for FiLM modulation.
                When provided and a ``film_generator`` exists, SIREN hidden layers are
                modulated, making the output kernel batch-dependent: ``[B, *spatial, out_dim]``.
                When ``None``, behaves identically to the original SIREN: ``[1, *spatial, out_dim]``.

        Returns:
            ``(kernel, grid)`` where ``kernel`` has shape ``[1|B, *spatial, out_dim]``
            and ``grid`` has shape ``[1, *spatial, data_dim]``.
        """
        # Generate FiLM parameters if conditioning is available
        film_params = None
        film_offset = 0
        if conditioning is not None and self.film_generator is not None:
            film_params = self.film_generator(conditioning)  # list of (gamma, beta), each [B, hidden_dim]

        # Generate positional embeddings and corresponding grid values
        pos_emb, grid = self.positional_embedding(seq_lens)  # [1, *spatial, emb], [1, *spatial, data_dim]

        # Optionally apply FiLM *after* the positional embedding sine
        if self.film_after_pos_embed and film_params is not None:
            gamma, beta = film_params[0]
            shape = [gamma.shape[0]] + [1] * self.data_dim + [gamma.shape[-1]]
            pos_emb = gamma.view(*shape) * pos_emb + beta.view(*shape)
            film_offset = 1

        # Forward through hidden layers with optional FiLM
        h = pos_emb
        for i, linear in enumerate(self.hidden_linears):
            h = self.sine(linear(h))
            if film_params is not None:
                gamma, beta = film_params[i + film_offset]
                # Reshape [B, hidden_dim] -> [B, 1, ..., 1, hidden_dim] for broadcasting over spatial dims
                shape = [gamma.shape[0]] + [1] * self.data_dim + [gamma.shape[-1]]
                gamma = gamma.view(*shape)
                beta = beta.view(*shape)
                h = gamma * h + beta

        kernel = self.out_linear(h)
        return kernel, grid


# ---------------------------------------------------------------------------
# Multi-ω₀ SIREN kernels
# ---------------------------------------------------------------------------


def _as_float_tensor(values: Sequence[float] | torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalise an omega_0 schedule argument into a 1-D float64 tensor.

    Accepts any sequence of floats (``list``, ``tuple``) or a 1-D tensor.
    Used by the multi-omega SIREN classes so that LazyConfig-built
    instantiations (which naturally represent schedules as lists of floats)
    work out of the box without the caller having to convert manually.

    All entries must be strictly positive; this is validated here so that
    every consumer (``MultiOmegaSIRENPositionalEmbeddingND``,
    ``BlockDiagonalMultiOmegaSIRENKernelND``, etc.) gets a consistent error
    message and need not repeat the check.

    Args:
        values: A 1-D sequence of strictly-positive floats, or a 1-D
            ``torch.Tensor``.  Multi-dimensional tensors are rejected.
        name: Human-readable parameter name used in error messages
            (e.g. ``"omega_0_per_row"``).

    Returns:
        1-D ``torch.float64`` tensor with the same values.

    Raises:
        ValueError: If ``values`` is not 1-D, is empty, or contains a
            non-positive entry.
    """
    if isinstance(values, torch.Tensor):
        out = values.detach().to(dtype=torch.float64).flatten()
    else:
        out = torch.as_tensor(list(values), dtype=torch.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(out.shape)}")
    if out.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if (out <= 0).any():
        raise ValueError(f"{name} entries must all be strictly positive, got {out.tolist()}")
    return out


def _build_omega_0_per_block(
    *,
    num_blocks: int,
    omega_0_min: float,
    omega_0_max: float,
    schedule: str,
) -> torch.Tensor:
    """Build a per-block omega_0 frequency schedule of length ``num_blocks``.

    Two schedule types are supported:

    * ``"linear"``: equally spaced between ``omega_0_min`` and ``omega_0_max``.
      Block ``k`` gets ``omega_0_min + k * (omega_0_max - omega_0_min) / (num_blocks - 1)``.
    * ``"log"``: equally spaced in log-10 between the two endpoints.
      Block ``k`` gets ``10^(log10(min) + k * (log10(max) - log10(min)) / (num_blocks - 1))``.

    A ``"log"`` schedule is recommended when the frequency range spans more
    than a decade (e.g. ``omega_0_min=1, omega_0_max=30``) because it gives
    equal coverage to each octave rather than concentrating most blocks near
    the high end.

    Args:
        num_blocks: Number of frequency blocks.  Must be >= 1.
        omega_0_min: Lowest frequency in the schedule.  Must be positive.
        omega_0_max: Highest frequency in the schedule.  Must be >= ``omega_0_min``.
        schedule: Either ``"linear"`` or ``"log"``.

    Returns:
        1-D ``torch.float64`` tensor of shape ``[num_blocks]`` with the
        per-block omega_0 values.

    Raises:
        ValueError: If ``num_blocks < 1``, if the endpoints are non-positive,
            if ``omega_0_max < omega_0_min``, or if ``schedule`` is unknown.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if omega_0_min <= 0 or omega_0_max <= 0:
        raise ValueError(f"omega_0_min/omega_0_max must be positive, got ({omega_0_min}, {omega_0_max})")
    if omega_0_max < omega_0_min:
        raise ValueError(f"omega_0_max must be >= omega_0_min, got ({omega_0_min}, {omega_0_max})")
    if schedule == "linear":
        return torch.linspace(float(omega_0_min), float(omega_0_max), num_blocks, dtype=torch.float64)
    if schedule == "log":
        return torch.logspace(
            math.log10(float(omega_0_min)), math.log10(float(omega_0_max)), num_blocks, dtype=torch.float64
        )
    raise ValueError(f"schedule must be 'linear' or 'log', got {schedule!r}")


class MultiOmegaSIRENPositionalEmbeddingND(SIRENPositionalEmbeddingND):
    """SIREN positional embedding with a per-row ω₀ in the first layer.

    The standard ``SIRENPositionalEmbeddingND`` draws every row of the first
    linear's weight from ``Uniform(-2π·ω₀/d, +2π·ω₀/d)`` using a single scalar
    ω₀.  This variant takes a *vector* of ω₀ values (one per embedding-dim /
    row) and re-draws each row independently with its own bound
    ``2π·ω₀_k / d``.

    This is the "per-row dense" multi-ω₀ init.  Every row is independent but
    downstream MLP layers mix all rows as usual, so at init all output
    channels see a weighted combination of every ω₀ in the schedule.  See
    ``BlockDiagonalMultiOmegaSIRENKernelND`` for a variant that also
    block-masks the MLP to keep rows disjoint at init.

    Attributes:
        omega_0 (float): Mean of the ``omega_0_per_row`` schedule; stored for
            parity with the scalar-``omega_0`` parent's diagnostic attribute.
        omega_0_per_row (torch.Tensor): Non-persistent float32 buffer of shape
            ``[embedding_dim]`` holding the per-row omega_0 values.
        linear (torch.nn.Linear): First-layer weight with per-row SIREN init;
            shape ``[embedding_dim, data_dim]``.  Each row ``k`` is initialised
            from ``U(-2*pi*omega_0_per_row[k]/d, +2*pi*omega_0_per_row[k]/d)``.
        grid_cache, step_sizes, L_cache_per_axis, L_cache:
            Inherited from ``SIRENPositionalEmbeddingND``; see that class.

    Args:
        data_dim: Number of spatial/temporal input dimensions.
        embedding_dim: Dimensionality of the positional embedding.
        L_cache: Cache extent (controls the initial grid cache size).
        omega_0_per_row: Sequence of ``embedding_dim`` strictly-positive floats
            (or a 1-D tensor) giving the omega_0 used for row *k* of the first
            linear.
        use_bias: Whether to include a bias term.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        L_cache: int | Sequence[int],
        omega_0_per_row: Sequence[float] | torch.Tensor,
        use_bias: bool = True,
    ):
        """Initialize the per-row multi-omega SIREN positional embedding; see the class docstring."""
        omega = _as_float_tensor(omega_0_per_row, name="omega_0_per_row")
        if omega.numel() != embedding_dim:
            raise ValueError(f"omega_0_per_row length ({omega.numel()}) must equal embedding_dim ({embedding_dim})")

        # Parent draws the first-layer weight with placeholder ω₀=1.0; we
        # immediately overwrite every row with its own ω₀-bound uniform.
        super().__init__(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            L_cache=L_cache,
            omega_0=1.0,
            use_bias=use_bias,
        )
        with torch.no_grad():
            d = float(self.linear.in_features)
            for k in range(embedding_dim):
                bound = 2.0 * math.pi * float(omega[k]) / d
                self.linear.weight[k, :].uniform_(-bound, bound)
            # Parent already zero-initialized bias; nothing more to do.

        # Bookkeeping: record the schedule and set the scalar ``omega_0`` to
        # the schedule's mean (for diagnostics and parity with the parent).
        self.register_buffer(
            "omega_0_per_row",
            omega.to(dtype=torch.float32),
            persistent=False,
        )
        self.omega_0 = float(omega.mean().item())


class MultiOmegaSIRENKernelND(SIRENKernelND):
    """SIRENKernelND with a per-row ω₀ in the first (positional-embedding) layer.

    Identical to ``SIRENKernelND`` except that the positional embedding is a
    :class:`MultiOmegaSIRENPositionalEmbeddingND` built from the supplied
    ``omega_0_per_row`` schedule.  All hidden/output layers retain the usual
    SIREN init with ``hidden_omega_0``.

    The ``omega_0`` attribute reported on the module equals the mean of the
    schedule, purely for diagnostic purposes.

    Attributes:
        omega_0 (float): Mean of ``omega_0_per_row``; for diagnostics.
        omega_0_per_row (torch.Tensor): Non-persistent float32 buffer of shape
            ``[embedding_dim]`` holding the per-row omega_0 schedule.
        positional_embedding (MultiOmegaSIRENPositionalEmbeddingND): Per-row
            omega_0 positional encoder (replaces the scalar-omega parent's
            ``SIRENPositionalEmbeddingND``).
        hidden_linears, sine, out_linear, film_generator:
            Inherited from :class:`SIRENKernelND`; see that class.

    Args:
        out_dim: Number of output channels for the generated kernel.
        data_dim: Number of spatial/temporal input dimensions.
        mlp_hidden_dim: Hidden width of the SIREN MLP.
        num_layers: Total number of SIREN layers (>= 2).
        embedding_dim: Positional-embedding dimensionality.
        omega_0_per_row: Sequence of ``embedding_dim`` strictly-positive floats
            giving the per-row omega_0 in the first layer.
        L_cache: Cache extent (controls the initial grid cache size).
        use_bias: Whether to include biases in linear layers.
        hidden_omega_0: Frequency scaling for hidden SIREN layers (unchanged
            from the parent).
        film_cfg, film_after_pos_embed: Same semantics as in the parent.
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        omega_0_per_row: Sequence[float] | torch.Tensor,
        L_cache: int | Sequence[int],
        use_bias: bool,
        hidden_omega_0: float = 1.0,
        film_cfg: LazyConfig | None = None,
        film_after_pos_embed: bool = False,
    ):
        """Initialize the per-row multi-omega SIREN kernel; see the class docstring for argument semantics."""
        omega = _as_float_tensor(omega_0_per_row, name="omega_0_per_row")
        if omega.numel() != embedding_dim:
            raise ValueError(f"omega_0_per_row length ({omega.numel()}) must equal embedding_dim ({embedding_dim})")

        # Build the parent with a placeholder scalar ω₀ — we swap out its
        # positional embedding for the per-row variant immediately below.
        # Using the mean as placeholder keeps the parent's scalar-ω₀ diagnostic
        # close to the schedule's ``omega_0`` value even before we overwrite it.
        super().__init__(
            out_dim=out_dim,
            data_dim=data_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            omega_0=float(omega.mean().item()),
            L_cache=L_cache,
            use_bias=use_bias,
            hidden_omega_0=hidden_omega_0,
            film_cfg=film_cfg,
            film_after_pos_embed=film_after_pos_embed,
        )

        # Replace the positional embedding with the per-row variant.  The
        # original one was already constructed (consuming its RNG draws) and
        # is now discarded; the new embedding performs an additional uniform
        # draw per row, so the overall RNG trajectory differs from a scalar-ω₀
        # SIREN with the same seed.  Hidden/output weights remain byte-identical
        # to a scalar SIREN built with ``omega_0 = mean(omega_0_per_row)``.
        self.positional_embedding = MultiOmegaSIRENPositionalEmbeddingND(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            L_cache=L_cache,
            omega_0_per_row=omega,
            use_bias=use_bias,
        )
        self.register_buffer(
            "omega_0_per_row",
            omega.to(dtype=torch.float32),
            persistent=False,
        )
        self.omega_0 = float(omega.mean().item())


class BlockDiagonalMultiOmegaSIRENKernelND(MultiOmegaSIRENKernelND):
    """Per-block ω₀ + (near-)block-diagonal MLP init for a SIREN kernel.

    Extends :class:`MultiOmegaSIRENKernelND` in two ways:

    1.  The first-layer ω₀ is piecewise-constant across ``num_blocks`` equal
        groups of embedding rows, with ω₀ for block *k* drawn from a
        ``linear`` or ``log`` schedule over ``[omega_0_min, omega_0_max]``.

    2.  Every hidden linear and the output linear have their weights multiplied
        by a block mask: weights on the block diagonal are preserved at 1.0,
        off-diagonal entries are scaled by ``off_block_scale``.  At init this
        reduces cross-correlation between output channels of different blocks,
        so early in training each block behaves like a small independent SIREN
        tuned to its own frequency band.  Training is free to fill in the
        off-block weights via gradient flow (they start small but are
        unconstrained).

    With ``off_block_scale = 0.0`` the kernel is *mathematically equivalent at
    init* to K parallel SIRENs (sharing seeds) packed into a single dense
    SIREN.  With ``off_block_scale = 1.0`` the block structure is invisible
    at init and we recover the parent :class:`MultiOmegaSIRENKernelND`.

    ``embedding_dim``, ``mlp_hidden_dim``, and ``out_dim`` must all be
    divisible by ``num_blocks``.

    Production defaults (chosen from a spectral-coverage study on the N=29
    grid used by the ``vit5_hybrid`` config): ``num_blocks=8``,
    ``omega_0_min=1.0``, ``omega_0_max=12.0``, ``schedule="linear"``,
    ``off_block_scale=0.1``.

    When changing grid resolution by a factor ``m``, the schedule should be
    scaled uniformly by ``m`` (``omega_0_min *= m``, ``omega_0_max *= m``) to
    preserve the Nyquist-normalized spectral coverage.  This variant should
    be paired with :class:`BlockAlignedGaussianModulationND` so that the
    widest Gaussians land on the lowest-ω₀ block.

    Args:
        out_dim: Number of output channels for the generated kernel.
        data_dim: Number of spatial/temporal input dimensions.
        mlp_hidden_dim: Hidden width of the SIREN MLP.
        num_layers: Total number of SIREN layers (>= 2).
        embedding_dim: Positional-embedding dimensionality.
        L_cache: Cache extent.
        use_bias: Whether to include biases in linear layers.
        num_blocks: Number of ω₀ blocks.  Must divide all of ``embedding_dim``,
            ``mlp_hidden_dim``, and ``out_dim``.
        omega_0_min, omega_0_max: Endpoints of the schedule (ignored if
            ``omega_0_per_block`` is supplied).
        schedule: ``"linear"`` or ``"log"``.
        off_block_scale: Scale applied to off-block entries in hidden + output
            linears at init.  ``0.0`` → strict block-diagonal; ``1.0`` →
            equivalent to the parent.
        omega_0_per_block: Optional explicit ω₀ schedule of length
            ``num_blocks``.  When supplied, overrides
            ``omega_0_min``/``omega_0_max``/``schedule``.
        hidden_omega_0, film_cfg, film_after_pos_embed: Same as the parent.

    Attributes:
        num_blocks (int): Number of frequency blocks.
        off_block_scale (float): Off-diagonal weight scale applied at init.
        omega_0_per_block (torch.Tensor): Non-persistent float32 buffer of
            shape ``[num_blocks]`` holding the per-block omega_0 schedule.
        positional_embedding (MultiOmegaSIRENPositionalEmbeddingND): Per-row
            omega_0 positional encoder (constant within each block).
        hidden_linears, out_linear, omega_0_per_row:
            Inherited from :class:`MultiOmegaSIRENKernelND`; see that class.
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        L_cache: int | Sequence[int],
        use_bias: bool,
        num_blocks: int = 8,
        omega_0_min: float = 1.0,
        omega_0_max: float = 12.0,
        schedule: str = "linear",
        off_block_scale: float = 0.1,
        omega_0_per_block: Sequence[float] | torch.Tensor | None = None,
        hidden_omega_0: float = 1.0,
        film_cfg: LazyConfig | None = None,
        film_after_pos_embed: bool = False,
    ):
        """Initialize the block-diagonal SIREN kernel; see the class docstring for argument semantics."""
        for name, dim in [("embedding_dim", embedding_dim), ("mlp_hidden_dim", mlp_hidden_dim), ("out_dim", out_dim)]:
            if dim % num_blocks != 0:
                raise ValueError(f"{name}={dim} must be divisible by num_blocks={num_blocks}")

        # --- Build / validate the per-block ω₀ schedule.
        if omega_0_per_block is None:
            omega_per_block = _build_omega_0_per_block(
                num_blocks=num_blocks,
                omega_0_min=omega_0_min,
                omega_0_max=omega_0_max,
                schedule=schedule,
            )
        else:
            omega_per_block = _as_float_tensor(omega_0_per_block, name="omega_0_per_block")
            if omega_per_block.numel() != num_blocks:
                raise ValueError(
                    f"omega_0_per_block length ({omega_per_block.numel()}) must equal num_blocks ({num_blocks})"
                )

        # --- Expand per-block ω₀ into a per-row schedule (constant within each block).
        rows_per_block = embedding_dim // num_blocks
        omega_per_row = omega_per_block.repeat_interleave(rows_per_block)

        super().__init__(
            out_dim=out_dim,
            data_dim=data_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            omega_0_per_row=omega_per_row,
            L_cache=L_cache,
            use_bias=use_bias,
            hidden_omega_0=hidden_omega_0,
            film_cfg=film_cfg,
            film_after_pos_embed=film_after_pos_embed,
        )

        # --- Apply the block mask to every hidden linear + the output linear.
        with torch.no_grad():
            for linear in self.hidden_linears:
                mask = self._block_mask(
                    linear.out_features,
                    linear.in_features,
                    num_blocks=num_blocks,
                    off_block_scale=off_block_scale,
                    device=linear.weight.device,
                    dtype=linear.weight.dtype,
                )
                linear.weight.data.mul_(mask)
            mask = self._block_mask(
                self.out_linear.out_features,
                self.out_linear.in_features,
                num_blocks=num_blocks,
                off_block_scale=off_block_scale,
                device=self.out_linear.weight.device,
                dtype=self.out_linear.weight.dtype,
            )
            self.out_linear.weight.data.mul_(mask)

        self.num_blocks = int(num_blocks)
        self.off_block_scale = float(off_block_scale)
        self.register_buffer(
            "omega_0_per_block",
            omega_per_block.to(dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _block_mask(
        out_dim: int,
        in_dim: int,
        *,
        num_blocks: int,
        off_block_scale: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a block-mask: 1.0 on the diagonal blocks, ``off_block_scale`` elsewhere.

        The returned tensor has shape ``[out_dim, in_dim]``.  Block sizes are
        ``out_dim // num_blocks`` rows by ``in_dim // num_blocks`` columns.

        Note: if ``out_dim`` or ``in_dim`` is not divisible by ``num_blocks``,
        block sizes are silently truncated via integer division.  The calling
        ``__init__`` validates divisibility and raises ``ValueError`` before
        reaching this method.

        Args:
            out_dim: Number of output features (rows of the weight matrix).
            in_dim: Number of input features (columns of the weight matrix).
            num_blocks: Number of equal-sized blocks along both axes.
            off_block_scale: Scalar fill value for off-diagonal block entries.
                Use ``0.0`` for a strict block-diagonal and ``1.0`` to leave
                the weights unmodified.
            device: Target device for the mask tensor.
            dtype: Target dtype for the mask tensor (should match the weight).

        Returns:
            Tensor of shape ``[out_dim, in_dim]`` with ``1.0`` on the block
            diagonal and ``off_block_scale`` elsewhere.

        Raises:
            ZeroDivisionError: If ``num_blocks == 0``.
        """
        rows_per_block = out_dim // num_blocks
        cols_per_block = in_dim // num_blocks
        mask = torch.full((out_dim, in_dim), float(off_block_scale), device=device, dtype=dtype)
        for k in range(num_blocks):
            r0, r1 = k * rows_per_block, (k + 1) * rows_per_block
            c0, c1 = k * cols_per_block, (k + 1) * cols_per_block
            mask[r0:r1, c0:c1] = 1.0
        return mask


# ---------------------------------------------------------------------------
# Learnable-ω₀ SIREN kernels
#
# These classes take the ``2π·ω₀`` factor *out* of the first-layer weight init
# and apply it explicitly at every forward pass, with an extra learnable
# per-row ``ω₀_scale`` multiplier capped at a configurable maximum.  At init
# (``ω₀_scale = 1``) the produced kernel is identical to the matching
# fixed-ω₀ class; during training the model can re-tune each row's effective
# ω₀ within ``[ω₀ · scale_min, ω₀ · scale_max]``.
#
# Optimizer note: removing ``2π·ω₀`` from the first-layer init increases the
# gradient norm of that weight by ``2π·ω₀`` (the factor that was previously
# absorbed; see the SIREN paper).  When ``apply_lr_scale=True`` the
# first-layer weight gets ``_lr_scale = 1/(2π·ω₀)`` so that AdamW's effective
# per-step update size is the same as the standard SIREN init.
# ---------------------------------------------------------------------------


class LearnableOmegaSIRENPositionalEmbeddingND(SIRENPositionalEmbeddingND):
    """SIREN positional embedding with a learnable per-row ω₀ multiplier.

    Forward computes (in float32, regardless of input dtype):

        sin( 2π · ω₀ · ω₀_scale · (W·x + b) )

    where:

      * ``W`` is the first-layer weight, initialized to ``U(-1/d, +1/d)``
        (the standard SIREN-1 init *without* the usual ``2π·ω₀`` bound
        scaling).  ``2π·ω₀`` is instead applied at every iteration as a
        single scalar buffer.
      * ``ω₀_scale`` is a learnable per-row parameter of shape
        ``[embedding_dim]``, initialized to ``omega_0_scale_init`` and
        clamped in-place (forward pre-hook, ``"direct"`` parametrization)
        to ``[omega_0_scale_min, omega_0_scale_max]``.  The default lower
        bound is a small positive floor (``1e-2``) rather than ``0`` so
        a row's effective ω₀ never collapses to zero — at zero the row's
        sine becomes a constant ``sin(bias)`` and the gradient signal
        through that row's ``ω₀_scale`` largely vanishes, making
        recovery hard.  With the defaults (init=1, max=2) the per-row
        effective ω₀ ranges from roughly ``0.01·ω₀`` to ``2·ω₀`` and the
        total multiplier inside the sine reaches ``4π·ω₀``.

    The float32 path covers the linear projection, the multiplier, and the
    sine; the result is cast back to the original input dtype after the sine.
    This matches the precision discipline already used for the grid cache in
    the parent ``SIRENPositionalEmbeddingND``.

    Args:
        data_dim: Number of spatial/temporal input dimensions.
        embedding_dim: Dimensionality of the positional embedding.
        L_cache: Cache extent (controls the initial grid cache size).
        omega_0: Constant scalar absorbed into the runtime ``2π·ω₀`` factor.
        omega_0_scale_init: Initial value of the learnable per-row scale.
            Either a single float (broadcast to ``embedding_dim``) or a 1-D
            sequence/tensor of length ``embedding_dim``.  Defaults to 1.0,
            so the effective per-row ω₀ at init equals ``omega_0``.
        omega_0_scale_min: Lower clamp on ``ω₀_scale``. **Must be strictly
            positive** — at ``0`` the row's first-layer sine collapses to
            a constant and the gradient signal through its scale largely
            vanishes, making recovery hard.  Default ``1e-2``.
        omega_0_scale_max: Upper clamp on ``ω₀_scale``. Default 2.0, giving
            a total multiplier inside the sine of up to ``4π·ω₀``.
        use_bias: Whether to include a bias term.
        apply_lr_scale: When True, attach ``_lr_scale = 1/(2*pi*omega_0)``
            to ``self.linear.weight``.  The optimizer utility
            ``_build_param_groups`` (in ``experiments/``) reads this attribute
            and multiplies the layer's effective learning rate by
            ``_lr_scale``, compensating for the missing ``2*pi*omega_0``
            factor in the SIREN-1 init bound so that the per-step update size
            matches a standard SIREN.  Default False (opt-in, so existing
            runs are unaffected and the new classes can be A/B-tested).

    Attributes:
        omega_0 (float): Constant part of the runtime multiplier (same as the
            ``omega_0`` constructor argument), stored for diagnostics.
        omega_0_scale_min (float): Lower clamp bound on ``omega_0_scale``.
        omega_0_scale_max (float): Upper clamp bound on ``omega_0_scale``.
        omega_0_const (torch.Tensor): Non-persistent float32 scalar buffer
            holding ``2*pi*omega_0``; applied to the linear output at every
            forward pass.
        omega_0_scale (torch.nn.Parameter): Learnable per-row scale of shape
            ``[embedding_dim]``.  Clamped to
            ``[omega_0_scale_min, omega_0_scale_max]`` by a forward pre-hook
            before each forward call.
        linear (torch.nn.Linear): First-layer weight ``W`` with *unscaled*
            SIREN-1 init ``U(-1/d, +1/d)`` (no ``2*pi*omega_0`` factor in
            the bound).  Shape ``[embedding_dim, data_dim]``.
        grid_cache, step_sizes, L_cache_per_axis, L_cache:
            Inherited from ``SIRENPositionalEmbeddingND``; see that class.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        L_cache: int | Sequence[int],
        omega_0: float,
        omega_0_scale_init: float | Sequence[float] | torch.Tensor = 1.0,
        omega_0_scale_min: float = 1e-2,
        omega_0_scale_max: float = 2.0,
        use_bias: bool = True,
        apply_lr_scale: bool = False,
    ):
        """Initialize the learnable-ω₀ SIREN positional embedding; see the class docstring."""
        if omega_0 <= 0:
            raise ValueError(f"omega_0 must be strictly positive, got {omega_0}")
        if omega_0_scale_max <= 0:
            raise ValueError(f"omega_0_scale_max must be strictly positive, got {omega_0_scale_max}")
        if omega_0_scale_min <= 0:
            # Strictly positive: at omega_0_scale = 0 the first-layer
            # sine of that row collapses to sin(bias) (a constant), its
            # contribution to the kernel output vanishes, and the
            # optimizer receives no signal to push the scale back up.
            # The default lower bound is 1e-2 for exactly this reason.
            raise ValueError(
                f"omega_0_scale_min must be strictly positive (a row's effective "
                f"ω₀ must never reach 0, otherwise that row's first-layer sine "
                f"becomes a constant and the gradient signal through its scale "
                f"collapses); got omega_0_scale_min={omega_0_scale_min}."
            )
        if omega_0_scale_max < omega_0_scale_min:
            raise ValueError(
                f"omega_0_scale_max ({omega_0_scale_max}) must be >= omega_0_scale_min ({omega_0_scale_min})"
            )

        # Build the parent with the actual ``omega_0`` so ``self.omega_0``
        # diagnostics stay correct, then immediately overwrite the first-layer
        # weight with the *unscaled* SIREN-1 bound (no 2π·ω₀ factor).
        super().__init__(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            L_cache=L_cache,
            omega_0=float(omega_0),
            use_bias=use_bias,
        )

        with torch.no_grad():
            d = float(self.linear.in_features)
            # SIREN first-layer bound *without* the 2π·ω₀ factor: U(-1/d, +1/d).
            self.linear.weight.uniform_(-1.0 / d, 1.0 / d)
            # Bias was already zero-initialized by the parent.

        # Resolve scale init into a [embedding_dim] tensor.
        if isinstance(omega_0_scale_init, (int, float)):
            scale_init = torch.full((embedding_dim,), float(omega_0_scale_init), dtype=torch.float32)
        elif isinstance(omega_0_scale_init, torch.Tensor):
            scale_init = omega_0_scale_init.detach().to(dtype=torch.float32).flatten()
        else:
            scale_init = torch.as_tensor(list(omega_0_scale_init), dtype=torch.float32)
        if scale_init.ndim != 1 or scale_init.numel() != embedding_dim:
            raise ValueError(
                f"omega_0_scale_init must broadcast to shape ({embedding_dim},); "
                f"got tensor of shape {tuple(scale_init.shape)}"
            )
        if (scale_init < omega_0_scale_min).any() or (scale_init > omega_0_scale_max).any():
            raise ValueError(
                f"omega_0_scale_init values must lie in [{omega_0_scale_min}, {omega_0_scale_max}]; "
                f"got min={float(scale_init.min()):.4f}, max={float(scale_init.max()):.4f}"
            )

        self.omega_0_scale_min = float(omega_0_scale_min)
        self.omega_0_scale_max = float(omega_0_scale_max)
        # ``omega_0_const`` carries the constant 2π·ω₀ factor; storing it as
        # a non-persistent buffer keeps it on the right device/dtype without
        # adding another trainable parameter or hard-coding ``omega_0`` only
        # as a Python float.
        self.register_buffer(
            "omega_0_const",
            torch.tensor(2.0 * math.pi * float(omega_0), dtype=torch.float32),
            persistent=False,
        )
        self.omega_0_scale = torch.nn.Parameter(scale_init.clone())
        # Treat the per-row scale like the mask's ``std_param``: no weight
        # decay, and a forward pre-hook that clamps it in-place without
        # disrupting gradient flow inside the active range.
        self.omega_0_scale._no_weight_decay = True
        self._scale_clamp_hook = self.register_forward_pre_hook(self._clamp_omega_scale_pre_hook)

        if apply_lr_scale:
            # Compensate for the absent 2π·ω₀ factor in the first-layer
            # init bound.  AdamW with this scale yields the same per-step
            # update size as the standard SIREN init.
            self.linear.weight._lr_scale = 1.0 / (2.0 * math.pi * float(omega_0))
            if self.linear.bias is not None:
                self.linear.bias._lr_scale = 1.0 / (2.0 * math.pi * float(omega_0))

    def _clamp_omega_scale_pre_hook(self, module, inputs):
        """Clamp ``omega_0_scale`` into ``[scale_min, scale_max]`` in-place before forward.

        Registered as a ``register_forward_pre_hook`` so the clamp runs
        automatically at the start of every forward call without requiring the
        caller to call it explicitly.  This is a "direct" parametrisation — the
        clamping does not block gradient flow for values already inside the
        valid range.

        Args:
            module: The module instance (``self``); provided by PyTorch's hook
                mechanism and not used directly.
            inputs: The positional inputs tuple passed to ``forward``; not
                inspected or modified.

        Returns:
            None.  Modifies ``self.omega_0_scale.data`` in-place via
            ``clamp_``.
        """
        with torch.no_grad():
            self.omega_0_scale.data.clamp_(min=self.omega_0_scale_min, max=self.omega_0_scale_max)

    def forward(self, seq_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the positional embedding with a fp32-internal learnable-ω₀ first layer.

        Args:
            seq_lens: Lengths of the input grid for which to compute the positional embeddings.

        Returns:
            tuple:
                - torch.Tensor: The positional embedding ``sin(2π·ω₀·s·(W·x+b))``
                  cast back to the original linear-weight dtype, of shape
                  ``[1, *spatial_dims, embedding_dim]``.
                - torch.Tensor: The grid coordinates, shape
                  ``[1, *spatial_dims, data_dim]`` (fp32, as in the parent).
        """
        assert len(seq_lens) == self.data_dim, (
            f"seq_lens must be of length {self.data_dim}. Current length: {len(seq_lens)}"
        )

        # Per-axis cache extension: shared with the parent, grows each axis
        # independently while preserving its original step size.
        self._maybe_extend_grid_cache(tuple(seq_lens))

        assert self.grid_cache.dtype == torch.float32, (
            f"grid_cache must be float32 (got {self.grid_cache.dtype}) — see the parent class for the rationale."
        )

        # Slice the cached grid to the requested per-axis lengths.
        offsets = [L - sl for L, sl in zip(self.L_cache_per_axis, seq_lens)]
        slices = [slice(off, off + (sl * 2) - 1) for off, sl in zip(offsets, seq_lens)]
        grid = self.grid_cache[:, *slices]  # type: ignore

        # ── float32 inner block: linear → multiplier → sin ────────────────
        # We force fp32 for the matmul, the per-row multiplier, and the sine
        # to keep the SIREN's high-frequency content well-resolved at the
        # boundary (otherwise bf16 would quantize the phase ``2π·ω₀·s·...``
        # too coarsely along the grid).  We then cast back to whatever dtype
        # the first-layer weight was stored in (typically the surrounding
        # autocast dtype, e.g. bf16 in mixed-precision training).
        target_dtype = self.linear.weight.dtype
        weight_fp32 = self.linear.weight.to(torch.float32)
        bias_fp32 = self.linear.bias.to(torch.float32) if self.linear.bias is not None else None
        pre = torch_F.linear(grid, weight_fp32, bias_fp32)  # [1, *spatial, embedding_dim], fp32

        # Total multiplier inside the sine: ``2π·ω₀ · ω₀_scale``.
        # Both factors are fp32; broadcasting ``[embedding_dim]`` over the
        # leading spatial dims via the trailing axis is shape-safe for any N-D.
        scale_fp32 = self.omega_0_scale.to(torch.float32)
        pre = pre * (self.omega_0_const * scale_fp32)
        out = pre.sin()
        # ── end float32 inner block ───────────────────────────────────────

        return out.to(target_dtype), grid

    def extra_repr(self) -> str:
        """Diagnostic string showing ω₀, the scale range, and the current scale stats."""
        with torch.no_grad():
            scale = self.omega_0_scale.detach().to(torch.float32)
        scale_min, scale_max, scale_mean = (
            float(scale.min()),
            float(scale.max()),
            float(scale.mean()),
        )
        return (
            f"data_dim={self.data_dim}, embedding_dim={self.embedding_dim}, "
            f"omega_0={self.omega_0:.4g}, "
            f"omega_0_scale_init range=[{scale_min:.4g}, {scale_max:.4g}] mean={scale_mean:.4g}, "
            f"clamp=[{self.omega_0_scale_min:.4g}, {self.omega_0_scale_max:.4g}]"
        )


class LearnableOmegaSIRENKernelND(SIRENKernelND):
    """SIRENKernelND whose first-layer ω₀ is multiplied by a learnable per-row scale.

    Identical to :class:`SIRENKernelND` except the positional embedding is a
    :class:`LearnableOmegaSIRENPositionalEmbeddingND`.  The hidden and output
    layers retain the usual SIREN init at ``hidden_omega_0``.

    With ``omega_0_scale_init = 1.0`` (the default) the kernel produced at
    init is **bit-for-bit identical** to a :class:`SIRENKernelND` built with
    the same scalar ``omega_0`` and seed (modulo the float32-mid-cast in the
    new positional embedding's forward, which is numerically more accurate
    than the parent's path under autocast).  During training the model can
    learn an effective per-row ω₀ in
    ``[omega_0 · omega_0_scale_min, omega_0 · omega_0_scale_max]``.

    Args:
        omega_0: Constant scalar absorbed into the per-iteration ``2π·ω₀``
            factor inside the first-layer sine.
        omega_0_scale_init: Initial value of the learnable per-row scale —
            either a single float (default ``1.0``) or a 1-D sequence/tensor
            of length ``embedding_dim``.  See
            :class:`LearnableOmegaSIRENPositionalEmbeddingND` for details.
        omega_0_scale_min: Lower clamp on the per-row scale.
        omega_0_scale_max: Upper clamp on the per-row scale.
        apply_lr_scale: Forwarded to the positional embedding.  Default
            ``False``.

    All other constructor arguments (``out_dim``, ``data_dim``,
    ``mlp_hidden_dim``, ``num_layers``, ``embedding_dim``, ``L_cache``,
    ``use_bias``, ``hidden_omega_0``, ``film_cfg``, ``film_after_pos_embed``)
    have the same meaning as in :class:`SIRENKernelND`.

    Attributes:
        positional_embedding (LearnableOmegaSIRENPositionalEmbeddingND): First
            layer with learnable per-row omega_0 scale; replaces the parent's
            ``SIRENPositionalEmbeddingND``.

    The attributes ``hidden_linears``, ``sine``, ``out_linear`` and
    ``film_generator`` are inherited unchanged from :class:`SIRENKernelND`.
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        omega_0: float,
        L_cache: int | Sequence[int],
        use_bias: bool,
        omega_0_scale_init: float | Sequence[float] | torch.Tensor = 1.0,
        omega_0_scale_min: float = 1e-2,
        omega_0_scale_max: float = 2.0,
        hidden_omega_0: float = 1.0,
        apply_lr_scale: bool = False,
        film_cfg: LazyConfig | None = None,
        film_after_pos_embed: bool = False,
    ):
        """Initialize the learnable-omega SIREN kernel; see the class docstring for argument semantics."""
        # Build the parent with the same ``omega_0`` so all internal
        # bookkeeping (``self.omega_0``, ``self.hidden_omega_0``) lines up.
        super().__init__(
            out_dim=out_dim,
            data_dim=data_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            omega_0=float(omega_0),
            L_cache=L_cache,
            use_bias=use_bias,
            hidden_omega_0=hidden_omega_0,
            film_cfg=film_cfg,
            film_after_pos_embed=film_after_pos_embed,
        )

        # Replace the parent's positional embedding with the learnable-ω₀
        # variant.  RNG-wise we burn one extra ``Uniform(-1/d, 1/d)`` draw on
        # top of the parent's ``Uniform(-2π·ω₀/d, +2π·ω₀/d)`` draw, so the
        # full module's parameter trajectory differs from its parent at the
        # same seed *only* in the first-layer weight; hidden + output linears
        # are byte-identical to a fresh ``SIRENKernelND`` of the same seed.
        self.positional_embedding = LearnableOmegaSIRENPositionalEmbeddingND(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            L_cache=L_cache,
            omega_0=float(omega_0),
            omega_0_scale_init=omega_0_scale_init,
            omega_0_scale_min=omega_0_scale_min,
            omega_0_scale_max=omega_0_scale_max,
            use_bias=use_bias,
            apply_lr_scale=apply_lr_scale,
        )


class BlockDiagonalLearnableOmegaSIRENKernelND(LearnableOmegaSIRENKernelND):
    """Block-diagonal learnable-ω₀ SIREN kernel.

    Combines two ideas:

    1.  **Block-diagonal MLP init** (from :class:`BlockDiagonalMultiOmegaSIRENKernelND`):
        every hidden linear and the output linear have their weights multiplied
        by a block mask — block-diagonal entries kept at 1.0, off-block
        entries scaled by ``off_block_scale``.
    2.  **Learnable per-row ω₀ schedule** (from :class:`LearnableOmegaSIRENKernelND`):
        the first-layer scale is initialized to a per-block schedule.  We
        absorb the *largest* block's ω₀ into the constant
        ``2π · omega_0_max`` runtime factor and let the learnable scale
        carry the *relative* schedule, initialized to
        ``omega_0_per_block / omega_0_max`` so that the effective per-row ω₀
        at init equals the original block-diagonal schedule.  The scale is
        clamped to ``[omega_0_scale_min, omega_0_scale_max]`` (default
        ``[1e-2, 2]``), giving every row room to up to double its effective
        ω₀ during training without any row ever collapsing to zero
        frequency.

    With ``omega_0_scale_init`` left at its default and the schedule built
    from ``(omega_0_min, omega_0_max, schedule)``, the kernel at init matches
    :class:`BlockDiagonalMultiOmegaSIRENKernelND` (modulo the fp32-mid-cast
    in the positional embedding's forward).

    ``embedding_dim``, ``mlp_hidden_dim``, and ``out_dim`` must all be
    divisible by ``num_blocks``.

    When ``apply_lr_scale=True`` the first-layer weight gets
    ``_lr_scale = 1/(2π · omega_0_max)`` — a single conservative scalar
    chosen to match the highest-frequency block.  This is the SIREN-paper
    LR compensation; the lowest-ω₀ block trains relatively slower under
    this scheme but the gradient norm of every row is upper-bounded by
    that of the most-aggressive row, which is the dimension that sets the
    largest update step size in AdamW.

    Args:
        num_blocks: Number of ω₀ blocks; must divide ``embedding_dim``,
            ``mlp_hidden_dim``, and ``out_dim``.
        omega_0_min: Lower endpoint of the schedule.  Ignored if
            ``omega_0_per_block`` is supplied (schedule endpoints are then
            read from the supplied vector).
        omega_0_max: Upper endpoint of the schedule; also sets the constant
            runtime ``2π · omega_0_max`` factor that is pulled out of the
            weight init.
        schedule: ``"linear"`` or ``"log"``.
        off_block_scale: Off-diagonal scaling for the hidden + output linear
            block masks.  ``0.0`` → strict block-diagonal; ``1.0`` →
            equivalent to a dense :class:`LearnableOmegaSIRENKernelND`.
        omega_0_per_block: Optional explicit ω₀ schedule of length
            ``num_blocks``.  Overrides ``omega_0_min``/``omega_0_max``/
            ``schedule`` when supplied.
        omega_0_scale_min: Lower clamp on the per-row scale (default
            ``1e-2``).  The strictly-positive floor keeps every row's
            effective ω₀ above ``1e-2 · omega_0_max`` so no row's first-layer
            sine collapses to a constant.
        omega_0_scale_max: Upper clamp on the per-row scale (default ``2``).
        apply_lr_scale: When ``True``, attach
            ``_lr_scale = 1/(2π·omega_0_max)`` to the first-layer weight.
            Default ``False``.

    All other constructor arguments (``out_dim``, ``data_dim``,
    ``mlp_hidden_dim``, ``num_layers``, ``embedding_dim``, ``L_cache``,
    ``use_bias``, ``hidden_omega_0``, ``film_cfg``, ``film_after_pos_embed``)
    have the same meaning as in :class:`SIRENKernelND`.

    Attributes:
        num_blocks (int): Number of frequency blocks.
        off_block_scale (float): Off-diagonal weight scale applied at init.
        omega_0_per_block (torch.Tensor): Non-persistent float32 buffer of
            shape ``[num_blocks]`` holding the per-block omega_0 schedule.
        positional_embedding (LearnableOmegaSIRENPositionalEmbeddingND):
            First layer with learnable per-row omega_0 scale; ``omega_0_const``
            is set to ``max(omega_0_per_block)`` and ``omega_0_scale`` is
            initialised to ``omega_0_per_block / omega_0_const`` per row.
        hidden_linears, out_linear, film_generator:
            Inherited from :class:`SIRENKernelND`; see that class.
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        L_cache: int | Sequence[int],
        use_bias: bool,
        num_blocks: int = 8,
        omega_0_min: float = 1.0,
        omega_0_max: float = 12.0,
        schedule: str = "linear",
        off_block_scale: float = 0.1,
        omega_0_per_block: Sequence[float] | torch.Tensor | None = None,
        omega_0_scale_min: float = 1e-2,
        omega_0_scale_max: float = 2.0,
        hidden_omega_0: float = 1.0,
        apply_lr_scale: bool = False,
        film_cfg: LazyConfig | None = None,
        film_after_pos_embed: bool = False,
    ):
        """Initialize the block-diagonal learnable-omega SIREN kernel; see the class docstring."""
        for name, dim in [
            ("embedding_dim", embedding_dim),
            ("mlp_hidden_dim", mlp_hidden_dim),
            ("out_dim", out_dim),
        ]:
            if dim % num_blocks != 0:
                raise ValueError(f"{name}={dim} must be divisible by num_blocks={num_blocks}")

        # ── Build / validate the per-block ω₀ schedule.
        if omega_0_per_block is None:
            omega_per_block = _build_omega_0_per_block(
                num_blocks=num_blocks,
                omega_0_min=omega_0_min,
                omega_0_max=omega_0_max,
                schedule=schedule,
            )
        else:
            omega_per_block = _as_float_tensor(omega_0_per_block, name="omega_0_per_block")
            if omega_per_block.numel() != num_blocks:
                raise ValueError(
                    f"omega_0_per_block length ({omega_per_block.numel()}) must equal num_blocks ({num_blocks})"
                )

        # The constant runtime factor pulls out the *largest* ω₀; the learnable
        # scale carries the relative schedule (in [0, 1] at init).  This way
        # every block's effective ω₀ at init equals the schedule entry, and
        # ``omega_0_scale_max`` controls the global headroom for growth.
        omega_0_const = float(omega_per_block.max().item())
        rows_per_block = embedding_dim // num_blocks
        scale_init = (omega_per_block.to(torch.float32) / omega_0_const).repeat_interleave(rows_per_block)

        super().__init__(
            out_dim=out_dim,
            data_dim=data_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            omega_0=omega_0_const,
            L_cache=L_cache,
            use_bias=use_bias,
            omega_0_scale_init=scale_init,
            omega_0_scale_min=omega_0_scale_min,
            omega_0_scale_max=omega_0_scale_max,
            hidden_omega_0=hidden_omega_0,
            apply_lr_scale=apply_lr_scale,
            film_cfg=film_cfg,
            film_after_pos_embed=film_after_pos_embed,
        )

        # ── Apply the block mask to every hidden linear + the output linear.
        with torch.no_grad():
            for linear in self.hidden_linears:
                mask = BlockDiagonalMultiOmegaSIRENKernelND._block_mask(
                    linear.out_features,
                    linear.in_features,
                    num_blocks=num_blocks,
                    off_block_scale=off_block_scale,
                    device=linear.weight.device,
                    dtype=linear.weight.dtype,
                )
                linear.weight.data.mul_(mask)
            mask = BlockDiagonalMultiOmegaSIRENKernelND._block_mask(
                self.out_linear.out_features,
                self.out_linear.in_features,
                num_blocks=num_blocks,
                off_block_scale=off_block_scale,
                device=self.out_linear.weight.device,
                dtype=self.out_linear.weight.dtype,
            )
            self.out_linear.weight.data.mul_(mask)

        self.num_blocks = int(num_blocks)
        self.off_block_scale = float(off_block_scale)
        self.register_buffer(
            "omega_0_per_block",
            omega_per_block.to(dtype=torch.float32),
            persistent=False,
        )


if __name__ == "__main__":
    torch.set_default_device("cuda")

    # Grid in 1D.
    embedding = RandomFourierPositionalEmbeddingND(data_dim=1, embedding_dim=4, L_cache=10, omega_0=1.0)
    _, grid = embedding(seq_lens=(10,))
    _, grid_2 = embedding(seq_lens=(25,))
    _, grid_3 = embedding(seq_lens=(10,))
    torch.testing.assert_close(grid, grid_3)

    # Grid in 2D.
    embedding = RandomFourierPositionalEmbeddingND(data_dim=2, embedding_dim=4, L_cache=10, omega_0=1.0)
    _, grid = embedding(seq_lens=(10, 10))
    _, grid_2 = embedding(seq_lens=(25, 25))
    _, grid_3 = embedding(seq_lens=(10, 10))
    torch.testing.assert_close(grid, grid_3)

    # Grid in 3D.
    embedding = RandomFourierPositionalEmbeddingND(data_dim=3, embedding_dim=4, L_cache=10, omega_0=1.0)
    _, grid = embedding(seq_lens=(10, 10, 10))
    _, grid_2 = embedding(seq_lens=(25, 25, 25))
    _, grid_3 = embedding(seq_lens=(10, 10, 10))
    torch.testing.assert_close(grid, grid_3)

    # Random Fourier kernel in 1D.
    kernel = RandomFourierKernelND(
        out_dim=4,
        data_dim=1,
        mlp_hidden_dim=4,
        num_layers=2,
        embedding_dim=4,
        omega_0=1.0,
        L_cache=10,
        use_bias=True,
        nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
    )
    kernel, grid = kernel(seq_lens=(10,))
    print(kernel.shape, grid.shape)

    # Random Fourier kernel in 2D.
    kernel = RandomFourierKernelND(
        out_dim=4,
        data_dim=2,
        mlp_hidden_dim=4,
        num_layers=2,
        embedding_dim=4,
        omega_0=1.0,
        L_cache=10,
        use_bias=True,
        nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
    )
    kernel, grid = kernel(seq_lens=(10, 10))
    print(kernel.shape, grid.shape)

    # --- SIREN sanity checks ---
    # Shapes
    siren = SIRENKernelND(
        out_dim=4,
        data_dim=2,
        mlp_hidden_dim=32,
        num_layers=3,
        embedding_dim=16,
        omega_0=30.0,
        L_cache=10,
        use_bias=True,
        hidden_omega_0=1.0,
    )
    siren_kernel, siren_grid = siren(seq_lens=(10, 10))
    print("SIREN kernel shape, grid shape:", siren_kernel.shape, siren_grid.shape)

    # Gradients flow check through a simple loss
    siren_kernel.sum().backward()
    grads_ok = all(p.grad is not None for p in siren.parameters() if p.requires_grad)
    print("SIREN grads present:", grads_ok)

    # --- Multi-ω₀ SIREN sanity checks ---
    # Block-diagonal variant with production defaults (K=8, linear [1,12], off=0.1)
    bd = BlockDiagonalMultiOmegaSIRENKernelND(
        out_dim=32,
        data_dim=2,
        mlp_hidden_dim=32,
        num_layers=3,
        embedding_dim=32,
        L_cache=10,
        use_bias=True,
    )
    k, g = bd(seq_lens=(10, 10))
    print("BlockDiag SIREN kernel:", k.shape, "omega_0_per_block:", bd.omega_0_per_block.tolist())
    k.sum().backward()
    assert all(p.grad is not None for p in bd.parameters() if p.requires_grad)

    # Strict block-diagonal (off=0.0): off-block entries of hidden + out linears are exactly 0.
    bd_strict = BlockDiagonalMultiOmegaSIRENKernelND(
        out_dim=32,
        data_dim=2,
        mlp_hidden_dim=32,
        num_layers=3,
        embedding_dim=32,
        L_cache=10,
        use_bias=True,
        off_block_scale=0.0,
    )
    rows_per_block = 32 // bd_strict.num_blocks
    for linear in bd_strict.hidden_linears:
        w = linear.weight.data.cpu()
        for i in range(bd_strict.num_blocks):
            for j in range(bd_strict.num_blocks):
                if i == j:
                    continue
                block = w[i * rows_per_block : (i + 1) * rows_per_block, j * rows_per_block : (j + 1) * rows_per_block]
                assert block.abs().max().item() == 0.0, f"off-block ({i},{j}) nonzero with off_block_scale=0"
    print("BlockDiag SIREN off=0.0 strictly block-diagonal: OK")

    # --- Anisotropic L_cache sanity checks (per-axis kernel grid) ---
    # 3D SIREN kernel with grid (8, 64, 64) → cache shape (1, 15, 127, 127, 16)
    aniso_kernel = SIRENKernelND(
        out_dim=4,
        data_dim=3,
        mlp_hidden_dim=8,
        num_layers=2,
        embedding_dim=8,
        omega_0=10.0,
        L_cache=(8, 64, 64),
        use_bias=True,
    )
    aniso_out, aniso_grid = aniso_kernel(seq_lens=(8, 64, 64))
    assert aniso_out.shape == (1, 15, 127, 127, 4), aniso_out.shape
    assert aniso_grid.shape == (1, 15, 127, 127, 3), aniso_grid.shape
    print("Anisotropic SIREN kernel L_cache=(8,64,64): kernel shape", aniso_out.shape)

    # Slicing for a smaller seq_lens uses the per-axis cached extents.
    aniso_small_out, aniso_small_grid = aniso_kernel(seq_lens=(4, 32, 32))
    assert aniso_small_out.shape == (1, 7, 63, 63, 4), aniso_small_out.shape
    print("Anisotropic SIREN kernel sliced for (4,32,32): kernel shape", aniso_small_out.shape)

    # Per-axis cache extension grows only the axis that needs growing.
    aniso_kernel(seq_lens=(16, 64, 64))
    assert aniso_kernel.positional_embedding.L_cache_per_axis == (16, 64, 64), (
        aniso_kernel.positional_embedding.L_cache_per_axis
    )
    print(
        "Anisotropic SIREN kernel after seq_lens=(16,64,64): L_cache_per_axis =",
        aniso_kernel.positional_embedding.L_cache_per_axis,
    )
