# David W. Romero, 2025-09-09

"""Implicit Kernel Implementations for ND signals (based on Random Fourier Feature Networks)."""


import math
from typing import Callable

import torch
import torch.nn.functional as torch_F
from einops import rearrange

from nvsubquadratic.src.utils.lazy_config import LazyConfig, instantiate


class RandomFourierPositionalEmbeddingND(torch.nn.Module):
    """
    Implements a N-dimensional positional embedding using Random Fourier Features.

    This module generates positional embeddings by applying a linear transformation
    with randomized Fourier frequencies followed by sine and cosine functions.
    It is suitable for tasks where positional information needs to be encoded.

    Args:
        data_dim (int): Dimension of input data (1D for sequences, 2D for images, 3D for videos, etc.).
        embedding_dim (int): Dimensionality of the positional embedding. Must be even.
        L_cache (int): Number of cached time steps for the input positions.
        omega_0 (float): Frequency scaling factor for the Fourier features.
        use_bias (bool, optional): Whether to use a bias term in the linear layer. Default is False.

    Raises:
        ValueError: If `embedding_dim` is not an even number.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        L_cache: int,
        omega_0: float,
        use_bias: bool = True,
    ):
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
        t = torch.linspace(-1, 1, 2 * self.L_cache - 1, dtype=torch.float32)
        grid_cache = rearrange(torch.stack(torch.meshgrid(*[t] * data_dim, indexing="ij"), dim=-1), "... -> 1 ...")
        self.register_buffer("grid_cache", grid_cache)

        # Save the step size for the cache, so that subsequent calls keep equal distances between the elements of the cache grid.
        self.step_size = 1.0 / (L_cache - 1)

        # Add ._no_wd flag to all parameters to avoid weight decay
        for param in self.parameters():
            param._no_wd = True

    def forward(self, seq_lens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the positional embeddings for a sequence of the given length.

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
    """
    Implements a learnable ND-dimensional freeform convolutional kernel using implicit neural representations.

    This module combines positional embeddings, a feedforward neural network, and optional modulation
    and normalization to compute freeform filters over a grid of spatial dimensions.

    Args:
        out_dim (int): Number of output channels for the generated kernel.
        data_dim (int): Dimension of input data (1D for sequences, 2D for images, 3D for videos, etc.).
        mlp_hidden_dim (int): Dimensionality of the hidden layers in the kernel network.
        num_layers (int): Number of layers in the kernel network (including embedding layer and output layer).
        embedding_dim (int): Dimensionality of the positional embeddings.
        omega_0 (float): Frequency scaling factor for the positional embeddings.
        L_cache (int): Number of cached time steps for the input positions.
        use_bias (bool): Whether to use bias in the network and embedding layers.
        nonlinear_cfg (LazyConfig): Configuration for the nonlinear activation function.
        init_method (Callable[[int], Callable[[torch.Tensor], torch.Tensor]], optional): Initialization method for the kernel network.
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

    def forward(self, seq_lens: tuple[int, ...]) -> torch.Tensor:
        """
        Computes the random Fourier kernel for a given grid of spatial dimensions.

        Args:
            seq_lens (tuple[int, ...]): Lengths of the input grid for which to compute the positional embeddings.

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
