# David W. Romero, 2025-09-09

"""Initialization functions."""

# Based off of savanna's implementation.

import math
from functools import partial
from typing import Callable

import torch


def small_init(dim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates an initialization function that fills a tensor with values sampled from a normal distribution **in_place**.

    The standard deviation is calculated based on the method described in the paper:
    "Transformers without Tears: Improving the Normalization of Self-Attention" by Nguyen, T. & Salazar, J. (2010).

    The formula for standard deviation (std) is:
        std = sqrt(2 / (5 * dim))

    Args:
        dim (int): The dimensionality of the tensor to be initialized. This typically corresponds to the
                   number of input features or the size of the hidden layer in a neural network.

    Returns:
        function: A function that takes a tensor as input and initializes it with a normal distribution
                  having mean 0.0 and the calculated standard deviation.
    """
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init(dim: int, num_layers: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates an initialization function that fills a tensor with values sampled from a normal distribution
    based on Wang's initialization method  **in_place**.

    The standard deviation is calculated using the formula:
        std = 2 / (num_layers * sqrt(dim))

    This method accounts for the number of layers and the dimensionality to maintain appropriate scaling
    of weights, which can help in stabilizing training for deep neural networks.

    Args:
        dim (int): The dimensionality of the tensor to be initialized. Typically corresponds to the
                   number of input features or the size of the hidden layer.
        num_layers (int): The total number of layers in the neural network. This helps in scaling the
                        standard deviation appropriately as the network depth increases.

    Returns:
        function: A function that takes a tensor as input and initializes it with a normal distribution
                  having mean 0.0 and the calculated standard deviation.
    """
    std = 2 / num_layers / math.sqrt(dim)

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def partial_wang_init_fn_with_num_layers(num_layers: int) -> Callable[[int], Callable[[torch.Tensor], torch.Tensor]]:
    """
    Factory that returns a function equivalent to ``partial(wang_init, num_layers=...)``.

    This is used with LazyConfig so that ``num_layers`` can be provided via
    OmegaConf interpolation (e.g., "${net.num_blocks}") and resolved before
    constructing the callable.
    """
    return partial(wang_init, num_layers=num_layers)
