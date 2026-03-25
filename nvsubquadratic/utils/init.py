"""Weight initialization factories.

All factories follow the same pattern: accept configuration parameters,
compute the target standard deviation, and return a ``partial`` that
initializes a tensor in-place when called.

Example usage::

    from nvsubquadratic.utils.init import trunc_normal_init, small_init

    # Direct use (e.g. ViT5Attention.init_fn)
    init_fn = trunc_normal_init(std=0.02)
    init_fn(some_linear.weight)

    # With MLP's curried signature fn(dim) -> fn(tensor)
    init_method_in = trunc_normal_init_factory(std=0.02)

    # Dim-dependent init (already matches MLP's curried signature)
    init_method_in = small_init  # small_init(dim) -> fn(tensor)
"""

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn


def trunc_normal_init(std: float = 0.02) -> Callable[[torch.Tensor], torch.Tensor]:
    """Truncated-normal initializer with fixed standard deviation.

    Args:
        std: Standard deviation for the truncated normal distribution.

    Returns:
        A callable ``fn(tensor) -> tensor`` that initializes the tensor
        in-place with ``trunc_normal_(mean=0, std=std)``.
    """
    return partial(nn.init.trunc_normal_, std=std)


def small_init(dim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Dim-dependent initializer from "Transformers without Tears" (Nguyen & Salazar, 2019).

    Computes ``std = sqrt(2 / (5 * dim))`` and returns a normal initializer
    with that standard deviation.

    Args:
        dim: Layer width used to compute the standard deviation.

    Returns:
        A callable ``fn(tensor) -> tensor`` that initializes the tensor
        in-place with ``normal_(mean=0, std=sqrt(2 / (5 * dim)))``.
    """
    std = math.sqrt(2 / (5 * dim))
    return partial(nn.init.normal_, mean=0.0, std=std)


def wang_init(dim: int, num_layers: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Depth-scaled initializer (Wang et al.).

    Computes ``std = 2 / (num_layers * sqrt(dim))`` and returns a normal
    initializer with that standard deviation.

    Args:
        dim: Layer width used to compute the standard deviation.
        num_layers: Total number of layers in the network.

    Returns:
        A callable ``fn(tensor) -> tensor`` that initializes the tensor
        in-place with ``normal_(mean=0, std=2 / (num_layers * sqrt(dim)))``.
    """
    std = 2 / num_layers / math.sqrt(dim)
    return partial(nn.init.normal_, mean=0.0, std=std)


def trunc_normal_init_factory(std: float = 0.02) -> Callable[[int], Callable[[torch.Tensor], torch.Tensor]]:
    """Factory that returns ``fn(dim) -> fn(tensor)`` for truncated-normal init.

    The ``dim`` argument is accepted but ignored — the standard deviation is
    fixed.  This makes the returned callable compatible with MLP's
    ``init_method_in`` / ``init_method_out`` curried signature.

    Args:
        std: Standard deviation for the truncated normal distribution.

    Returns:
        A callable ``fn(dim) -> fn(tensor)`` compatible with MLP's
        ``init_method_in`` / ``init_method_out`` signature.
    """
    _init = trunc_normal_init(std=std)

    def _factory(_dim: int) -> Callable[[torch.Tensor], torch.Tensor]:
        return _init

    return _factory


def partial_wang_init_fn_with_num_layers(num_layers: int) -> Callable[[int], Callable[[torch.Tensor], torch.Tensor]]:
    """Factory that returns ``partial(wang_init, num_layers=...)``.

    Useful with LazyConfig so that ``num_layers`` can be provided via
    OmegaConf interpolation (e.g., ``"${net.num_blocks}"``) and resolved
    before constructing the callable.

    Args:
        num_layers: Total number of layers, baked into the returned factory.

    Returns:
        A callable ``fn(dim) -> fn(tensor)`` compatible with MLP's
        ``init_method_in`` / ``init_method_out`` signature.
    """
    return partial(wang_init, num_layers=num_layers)
