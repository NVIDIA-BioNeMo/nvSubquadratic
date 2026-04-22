# TODO: Add license header here


"""Implicit Kernel Implementations for ND signals (based on Random Fourier Feature Networks).

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


class RandomFourierPositionalEmbeddingND(torch.nn.Module):
    """Implements a N-dimensional positional embedding using Random Fourier Features.

    This module generates positional embeddings by applying a linear transformation
    with randomized Fourier frequencies followed by sine and cosine functions.
    It is suitable for tasks where positional information needs to be encoded.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        L_cache: int,
        omega_0: float,
        use_bias: bool = True,
    ):
        """Initialize the RandomFourierPositionalEmbeddingND.

        Args:
            data_dim: Dimension of input data.
            embedding_dim: Dimensionality of the positional embedding. Must be even.
            L_cache: Number of cached time steps for the input positions.
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

        # Construct grid cache (cube) of size 2 * L_cache - 1.
        # TODO(@dwromero): We must make sure that the grid_cache is kept in float32.
        with torch.inference_mode(False):
            with torch.no_grad():
                t = torch.linspace(-1, 1, 2 * self.L_cache - 1, dtype=torch.float32)
                grid_cache = rearrange(
                    torch.stack(torch.meshgrid(*[t] * data_dim, indexing="ij"), dim=-1), "... -> 1 ..."
                )
        self.register_buffer("grid_cache", grid_cache, persistent=False)

        # Save the step size for the cache, so that subsequent calls keep equal distances between the elements of the cache grid.
        self.step_size = 1.0 / (L_cache - 1)

        # Add ._no_weight_decay flag to all parameters to avoid weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    def forward(self, seq_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the positional embeddings for a sequence of the given length.

        Args:
            seq_lens (tuple[int, ...]): Lengths of the input grid for which to compute the positional embeddings.

        Returns:
            tuple:
                - torch.Tensor: The positional embeddings, concatenated sine and cosine values (shape: [1, * spatial_dims, embedding_dim]).
                - torch.Tensor: The input positions normalized between [-1, 1] (shape: [1, * spatial_dims, 1]).

        Raises:
            AssertionError: If `seq_lens` is not of length `self.data_dim`.
            AssertionError: If `self.grid_cache` is not of type `torch.float32`.
        """
        # Check that the sequence lengths are of the correct length.
        assert len(seq_lens) == self.data_dim, (
            f"seq_lens must be of length {self.data_dim}. Current length: {len(seq_lens)}"
        )

        # Get the maximum sequence length.
        seq_len = max(seq_lens)

        # If the sequence is longer than the cache, create a new grid cache.
        if self.L_cache < seq_len:
            with torch.inference_mode(False):
                with torch.no_grad():
                    max_limit = 1.0 + self.step_size * (seq_len - self.L_cache)
                    t = torch.linspace(
                        -max_limit, max_limit, 2 * seq_len - 1, device=self.grid_cache.device, dtype=torch.float32
                    )
                    self.grid_cache = rearrange(
                        torch.stack(torch.meshgrid(*[t] * self.data_dim, indexing="ij"), dim=-1), "... -> 1 ..."
                    )
                    self.L_cache = seq_len

        # Ensure that the cached positions tensor has the correct data type.
        assert self.grid_cache.dtype == torch.float32, (
            f"grid_cache must be float32. At lower precision, indexes will be merged together. Current dtype: {self.grid_cache.dtype}"
        )

        # Calculate the offsets for the grid cache.
        offsets = [
            self.L_cache - seq_len for seq_len in seq_lens
        ]  # Values from which to start indexing the grid cache.

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
    """Implements a learnable ND-dimensional freeform convolutional kernel using implicit neural representations.

    This module combines positional embeddings, a feedforward neural network, and optional modulation
    and normalization to compute freeform filters over a grid of spatial dimensions.
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        omega_0: float,
        L_cache: int,
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
            L_cache: Number of cached time steps for the input positions.
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
        # This boils down to modulating the weight of the output layer by the expected kernel size.
        with torch.no_grad():
            self.out_linear.weight.data *= math.sqrt(1.0 / (L_cache**data_dim))

    def forward(self, seq_lens: tuple[int, ...], conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """Computes the random Fourier kernel for a given grid of spatial dimensions.

        Args:
            seq_lens (tuple[int, ...]): Lengths of the input grid for which to compute the positional embeddings.
            conditioning: Unused. Accepted for API compatibility with FiLM-enabled kernels.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed random Fourier kernel and the corresponding grid values.
                The kernel is a tensor of shape (1, * spatial_dims, out_dim)
                The grid is a tensor of shape (1, * spatial_dims, data_dim)
        """
        # Generate positional embeddings and corresponding grid values
        pos_emb, grid = self.positional_embedding(seq_lens)
        # Pass embeddings through the kernel network and output layer
        kernel = self.out_linear(self.kernel_network(pos_emb))
        return kernel, grid


def _init_siren_weights(layer: torch.nn.Linear, is_first_layer: bool, w0: float) -> None:
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
    """Sine activation used in SIREN with configurable frequency scaling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Sine activation."""
        return torch.sin(x)


class SIRENPositionalEmbeddingND(torch.nn.Module):
    """Implements a N-dimensional positional embedding using Sine features.

    This module generates positional embeddings by applying a linear transformation
    with randomized frequencies followed by sine activation.
    It is suitable for tasks where positional information needs to be encoded.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        L_cache: int,
        omega_0: float,
        use_bias: bool = True,
    ):
        """Initialize the SIRENPositionalEmbeddingND class.

        Args:
            data_dim: Dimension of input data.
            embedding_dim: Dimensionality of the positional embedding.
            L_cache: Number of cached time steps for the input positions.
            omega_0: Frequency scaling factor for the Fourier features.
            use_bias: Whether to use a bias term in the linear layer.
        """
        super().__init__()
        self.data_dim = data_dim
        self.embedding_dim = embedding_dim
        self.L_cache = L_cache
        self.omega_0 = omega_0
        self.use_bias = use_bias

        # Construct linear projection
        self.linear = torch.nn.Linear(in_features=data_dim, out_features=embedding_dim, bias=use_bias)

        # Initialize linear projection following SIREN initialization.
        _init_siren_weights(self.linear, is_first_layer=True, w0=self.omega_0)

        # Construct grid cache (cube) of size 2 * L_cache - 1.
        with torch.inference_mode(False):
            with torch.no_grad():
                t = torch.linspace(-1, 1, 2 * self.L_cache - 1, dtype=torch.float32)
                grid_cache = rearrange(
                    torch.stack(torch.meshgrid(*[t] * data_dim, indexing="ij"), dim=-1), "... -> 1 ..."
                )
        self.register_buffer("grid_cache", grid_cache, persistent=False)

        # Save the step size for the cache, so that subsequent calls keep equal distances between the elements of the cache grid.
        self.step_size = 1.0 / (L_cache - 1)

        # Add ._no_weight_decay flag to all parameters to avoid weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    def forward(self, seq_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the positional embeddings for a sequence of the given length.

        Args:
            seq_lens (tuple[int, ...]): Lengths of the input grid for which to compute the positional embeddings.

        Returns:
            tuple:
                - torch.Tensor: The positional embeddings, concatenated sine and cosine values (shape: [1, * spatial_dims, embedding_dim]).
                - torch.Tensor: The input positions normalized between [-1, 1] (shape: [1, * spatial_dims, 1]).

        Raises:
            AssertionError: If `seq_lens` is not of length `self.data_dim`.
            AssertionError: If `self.grid_cache` is not of type `torch.float32`.
        """
        # Check that the sequence lengths are of the correct length.
        assert len(seq_lens) == self.data_dim, (
            f"seq_lens must be of length {self.data_dim}. Current length: {len(seq_lens)}"
        )

        # Get the maximum sequence length.
        seq_len = max(seq_lens)

        # If the sequence is longer than the cache, create a new grid cache.
        if self.L_cache < seq_len:
            with torch.inference_mode(False):
                with torch.no_grad():
                    max_limit = 1.0 + self.step_size * (seq_len - self.L_cache)
                    t = torch.linspace(
                        -max_limit, max_limit, 2 * seq_len - 1, device=self.grid_cache.device, dtype=torch.float32
                    )

                    self.grid_cache = rearrange(
                        torch.stack(torch.meshgrid(*[t] * self.data_dim, indexing="ij"), dim=-1), "... -> 1 ..."
                    )
                    self.L_cache = seq_len

        # Ensure that the cached positions tensor has the correct data type.
        assert self.grid_cache.dtype == torch.float32, (
            f"grid_cache must be float32. At lower precision, indexes will be merged together. Current dtype: {self.grid_cache.dtype}"
        )

        # Calculate the offsets for the grid cache.
        offsets = [
            self.L_cache - seq_len for seq_len in seq_lens
        ]  # Values from which to start indexing the grid cache.

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
    """Kernel parameterized by a SIREN (sinusoidal representation network) MLP.

    The network maps coordinates in an N-D grid directly to kernel values.
    Optionally supports FiLM (Feature-wise Linear Modulation) conditioning:
    when ``film_cfg`` is provided, a ``KernelFiLMGenerator`` produces per-layer
    (gamma, beta) pairs that modulate hidden activations, making the kernel
    input-dependent.

    Args:
        out_dim: Number of output channels for the generated kernel.
        data_dim: Number of spatial/temporal input dimensions (size of coordinate vector).
        mlp_hidden_dim: Hidden width of the SIREN network.
        num_layers: Total number of layers including the first and hidden layers (>= 2).
        L_cache: Cache extent controlling the maximum supported grid size before cache growth.
        use_bias: Whether to include biases in linear layers.
        omega_0: Frequency scaling for the first SIREN layer.
        hidden_omega_0: Frequency scaling for subsequent SIREN layers.
        film_cfg: Optional LazyConfig for KernelFiLMGenerator. When provided, enables
            input-dependent FiLM conditioning of all hidden SIREN layers.
        film_after_pos_embed: If True, the first FiLM (gamma, beta) pair modulates
            the positional embedding *after* the sine activation (i.e. scales/shifts
            the ``sin(omega_0 * x)`` output).  Requires
            ``embedding_dim == mlp_hidden_dim`` and one extra FiLM layer in ``film_cfg``
            (i.e. ``num_film_layers = num_layers - 1 + 1 = num_layers``).
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        omega_0: float,
        L_cache: int,
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
        # Add Wang initialization to the output layer (to account for the fact that the output is used as a convolutional kernel)
        with torch.no_grad():
            self.out_linear.weight.data *= math.sqrt(1.0 / (L_cache**data_dim))  # Modulation by expected kernel size.

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
        """Count FLOPs for SIREN kernel generation on the positional grid.

        At ``inference=True`` with no FiLM generator, returns 0 because the
        kernel is input-independent and can be precomputed once and cached.
        When a ``film_generator`` exists, the kernel is input-dependent (via
        register-conditioned FiLM modulation) and must be recomputed every
        forward pass regardless of the inference flag.

        Let G = prod(2 * L_i - 1 for L_i in grid_lens) = total grid points.

        FLOPs breakdown:
          1. Positional embedding (``SIRENPositionalEmbeddingND``):
             Linear(``self.data_dim``, ``self.embedding_dim``) on G points:
               2 * G * data_dim * embedding_dim
             + sin activation: G * embedding_dim

          2. Hidden SIREN layers (``len(self.hidden_linears)`` = num_layers - 1):
             First:  Linear(embedding_dim, mlp_hidden_dim) + sin
               2 * G * embedding_dim * mlp_hidden_dim  +  G * mlp_hidden_dim
             Rest:   Linear(mlp_hidden_dim, mlp_hidden_dim) + sin  (each)
               2 * G * mlp_hidden_dim * mlp_hidden_dim  +  G * mlp_hidden_dim

          3. Output linear:
             Linear(``self.mlp_hidden_dim``, ``self.out_dim``) on G points:
               2 * G * mlp_hidden_dim * out_dim

          4. FiLM conditioning (only when ``self.film_generator`` is not None):
             a. FiLM generator MLP:  ``self.film_generator.flop_count()``
             b. Per modulated layer:  gamma * h + beta = 2 * G * mlp_hidden_dim
                Applied to each hidden layer, plus the positional embedding
                when ``self.film_after_pos_embed`` is True.
                (Note: film_after_pos_embed requires embedding_dim == mlp_hidden_dim.)

        Args:
            grid_lens: Spatial extents passed to the positional embedding.
                The kernel grid has size ``(2 * L - 1)`` per dimension.
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
            conditioning: Optional [B, C] conditioning vector for FiLM modulation.
                When provided and a film_generator exists, SIREN hidden layers are
                modulated, making the output kernel batch-dependent: [B, *spatial, out_dim].
                When None, behaves identically to the original SIREN: [1, *spatial, out_dim].

        Returns:
            tuple: (kernel, grid) where kernel has shape [1|B, *spatial, out_dim]
                and grid has shape [1, *spatial, data_dim].
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
    """Normalize an ω₀ schedule into a 1D float tensor.

    Accepts any sequence of floats (``list``, ``tuple``) or a 1D tensor.  Used
    by the multi-ω₀ SIREN classes so that LazyConfig-built instantiations (which
    naturally represent schedules as lists of floats) work out of the box.
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
    """Build a per-block ω₀ vector from (min, max, num_blocks, schedule).

    ``schedule`` is either ``"linear"`` (evenly spaced between min and max) or
    ``"log"`` (log-spaced — evenly spaced in log-10).  Returned tensor has
    ``dtype=torch.float64`` and length ``num_blocks``.
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

    Args:
        data_dim: Number of spatial/temporal input dimensions.
        embedding_dim: Dimensionality of the positional embedding.
        L_cache: Cache extent (controls the initial grid cache size).
        omega_0_per_row: Sequence of ``embedding_dim`` strictly-positive floats
            (or a 1-D tensor) giving the ω₀ used for row *k* of the first
            linear.
        use_bias: Whether to include a bias term.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        L_cache: int,
        omega_0_per_row: Sequence[float] | torch.Tensor,
        use_bias: bool = True,
    ):
        """Initialize the per-row multi-ω₀ SIREN positional embedding; see the class docstring."""
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

    Args:
        out_dim: Number of output channels for the generated kernel.
        data_dim: Number of spatial/temporal input dimensions.
        mlp_hidden_dim: Hidden width of the SIREN MLP.
        num_layers: Total number of SIREN layers (>= 2).
        embedding_dim: Positional-embedding dimensionality.
        omega_0_per_row: Sequence of ``embedding_dim`` strictly-positive floats
            giving the per-row ω₀ in the first layer.
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
        L_cache: int,
        use_bias: bool,
        hidden_omega_0: float = 1.0,
        film_cfg: LazyConfig | None = None,
        film_after_pos_embed: bool = False,
    ):
        """Initialize the per-row multi-ω₀ SIREN kernel; see the class docstring for argument semantics."""
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
    grid used by the vit5_hybrid config):
        ``num_blocks=8``, ``omega_0_min=1.0``, ``omega_0_max=12.0``,
        ``schedule="linear"``, ``off_block_scale=0.1``.

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
    """

    def __init__(
        self,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        L_cache: int,
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
        ``out_dim // num_blocks`` rows by ``in_dim // num_blocks`` columns;
        ``out_dim`` and ``in_dim`` must both be divisible by ``num_blocks``.
        """
        rows_per_block = out_dim // num_blocks
        cols_per_block = in_dim // num_blocks
        mask = torch.full((out_dim, in_dim), float(off_block_scale), device=device, dtype=dtype)
        for k in range(num_blocks):
            r0, r1 = k * rows_per_block, (k + 1) * rows_per_block
            c0, c1 = k * cols_per_block, (k + 1) * cols_per_block
            mask[r0:r1, c0:c1] = 1.0
        return mask


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
