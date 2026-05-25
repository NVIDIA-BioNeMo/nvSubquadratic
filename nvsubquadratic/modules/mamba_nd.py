# TODO: Add license header here


r"""Mamba-ND: selective state-space mixer for 1D/2D/3D signals.

Background
----------
Mamba (Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
Spaces", arXiv:2312.00752) is a **selective state-space model (SSM)** that
achieves linear time complexity in sequence length — O(N) — while retaining
the long-range modelling capacity of attention.

The core SSM recurrence is:

.. math::

    h_t &= \bar{A}_t \, h_{t-1} + \bar{B}_t \, x_t \\
    y_t &= C_t \, h_t

where :math:`h_t \in \mathbb{R}^{d \times N}` is a latent state, and
:math:`\bar{A}_t`, :math:`\bar{B}_t` are **input-dependent** discretised
transition matrices.  The key departure from classical linear SSMs (e.g.
S4, S5) is that :math:`B`, :math:`C`, and the step size :math:`\Delta` are
*functions of the input* :math:`x_t` rather than fixed parameters:

.. math::

    \Delta_t,\ B_t,\ C_t = \mathrm{Linear}(x_t)

The continuous-time state matrix :math:`A` is discretised using two rules:

* :math:`\bar{A}_t = e^{\Delta_t A}` via the **zero-order hold (ZOH)** rule.
* :math:`\bar{B}_t = \Delta_t B_t` via the **Euler (first-order)** rule.

The Euler rule for :math:`\bar{B}_t` is the discretisation actually used by
``mamba_ssm`` (Eq. 4 in arXiv:2312.00752); the full ZOH formula
:math:`(e^{\Delta A} - I) A^{-1} B` is an alternative that the paper mentions
but does not use in the default implementation.

This selectivity allows Mamba to focus on relevant tokens and ignore
irrelevant context, giving it an advantage over fixed-kernel convolutions
(Hyena, CKConv) on tasks requiring content-based filtering, while remaining
subquadratic unlike attention.

Comparison with other mixers
-----------------------------
+-----------+-------------------+--------------------+--------------------------------------------+
| Mixer     | Sequence-mixing   | Kernel             | Complexity (in N)                          |
+===========+===================+====================+============================================+
| Attention | pairwise dot-prod | input-dependent    | O(N^2)                                     |
+-----------+-------------------+--------------------+--------------------------------------------+
| Hyena     | FFT convolution   | fixed (learned MLP)| O(N log N)                                 |
+-----------+-------------------+--------------------+--------------------------------------------+
| Mamba     | SSM recurrence    | input-dependent    | O(N) training; O(1)/step inference         |
+-----------+-------------------+--------------------+--------------------------------------------+

O(N) training requires the hardware-aware parallel scan in ``mamba_ssm``
(custom CUDA extension).  A naive sequential or parallel-scan implementation
is O(N log N).  At inference time the recurrent form costs O(1) per step with
a fixed-size state, making Mamba particularly attractive for autoregressive
generation.

ND generalisation strategy
--------------------------
The Mamba recurrence is inherently sequential and 1D.  This module extends it
to arbitrary spatial rank (1D sequences, 2D images, 3D volumes) by **flattening
all spatial axes into a single sequence dimension** before running the core Mamba
layer:

.. code-block:: none

    [B, *spatial, C]
         |  rearrange "b ... c -> b (...) c"
         v
    [B, S, C]          where S = prod(spatial_dims)
         |  Mamba1D core (or bidirectional pair)
         v
    [B, S, C]
         |  reshape back to original spatial layout
         v
    [B, *spatial, C]

The scan order for multi-dimensional inputs follows the default PyTorch /
``einops`` row-major (C-contiguous) flattening: for a 2D ``[H, W]`` input the
tokens are visited in raster-scan order (row 0, col 0 to row 0, col W-1 to
row 1, col 0 and so on).  For 3D ``[D, H, W]`` inputs the outermost axis varies
slowest.  This ordering is fixed and is not learned; future work could explore
Hilbert-curve or zigzag orderings for improved spatial locality.

**Vertical anisotropy warning**: in a 2D ``[H, W]`` input, vertically adjacent
pixels (same column, adjacent rows) are ``W`` tokens apart in the flattened
sequence.  The SSM must propagate information across ``W`` state-update steps
to relate them, which may lose spatial correlation for wide images.
Bidirectional mode partially mitigates this by letting the reverse scan see
vertical neighbours in the forward direction.

Bidirectional mode
------------------
Setting ``bidirectional=True`` instantiates a second Mamba layer
(``core_layer_rev``) that processes the token sequence in **reverse order**.
The reversed output is flipped back and added to the forward output:

.. math::

    \text{out} = \mathrm{Mamba}(x)
               + \mathrm{flip}(\mathrm{Mamba}_\mathrm{rev}(\mathrm{flip}(x)))

This makes the effective receptive field of each position span the entire
sequence in both directions, at the cost of 2x parameters and compute.
For non-causal spatial tasks (images, volumes) bidirectional mode is strongly
recommended.

Integration with the rest of the library
-----------------------------------------
:class:`Mamba` is designed to be used as the ``inner_mixer`` inside
:class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer` (which adds
shared QKV and output linear projections).  It is also listed as a supported
mixer in :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`'s
dispatch table.

Related modules
---------------
* ``nvsubquadratic.modules.hyena_nd`` - Hyena (fixed-kernel gated conv)
* ``nvsubquadratic.modules.attention`` - multi-head self-attention
* ``nvsubquadratic.modules.sequence_mixer`` - operator-agnostic dispatch layer

References:
----------
Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective
State Spaces*. arXiv:2312.00752.
"""

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class Mamba(torch.nn.Module):
    r"""Selective state-space mixer for ND signals.

    Wraps a 1D Mamba core layer (e.g. ``mamba_ssm.Mamba``) and extends it to
    arbitrary spatial rank by flattening all spatial axes into a single
    sequence dimension before the SSM recurrence and reshaping back afterward.

    The SSM recurrence computed by the core layer is:

    .. math::

        h_t &= \bar{A}_t \, h_{t-1} + \bar{B}_t \, x_t \\
        y_t &= C_t \, h_t

    where the transition matrices :math:`\bar{A}_t`, :math:`\bar{B}_t` and
    the readout matrix :math:`C_t` are all *functions of* :math:`x_t`,
    derived via linear projections inside the core layer.  The step size
    :math:`\Delta_t` (also input-dependent) controls the discretisation:
    :math:`\bar{A}_t = e^{\Delta_t A}` (ZOH) and
    :math:`\bar{B}_t = \Delta_t B_t` (Euler).

    **Scan order for ND inputs**: spatial axes are flattened in row-major
    (C-contiguous) order, i.e. for 2D ``[H, W]`` the sequence visits tokens
    as (0,0), (0,1), ..., (0,W-1), (1,0), ... (raster-scan).  For 3D
    ``[D,H,W]`` the depth axis varies slowest.  This ordering is fixed
    (not learned).  Vertically adjacent pixels are ``W`` steps apart in the
    flattened sequence; see the module docstring for the anisotropy implication.

    **Bidirectional mode**: when ``bidirectional=True`` a second core layer
    processes the flattened sequence in reverse, and its (re-reversed) output
    is summed with the forward output.  This gives every position a full-sequence
    receptive field in both causal directions, which is beneficial for
    non-causal spatial tasks such as image or volume modelling.

    Attributes:
        bidirectional (bool): Whether to apply a second reversed Mamba pass.
        core_layer (torch.nn.Module): The forward (or only) Mamba core.
            Must accept input of shape ``[B, S, C]`` and return ``[B, S, C]``.
        core_layer_rev (torch.nn.Module): The reverse Mamba core, instantiated
            only when ``bidirectional=True``.  When ``bidirectional=False`` this
            attribute is not registered and accessing it raises
            :class:`AttributeError` by design, keeping the module's parameter
            count and ``state_dict`` unaffected.

    Example::

        import torch
        from nvsubquadratic.lazy_config import LazyConfig
        from nvsubquadratic.modules.mamba_nd import Mamba
        from mamba_ssm import Mamba as MambaCore

        mamba = Mamba(
            mamba_layer_cfg=LazyConfig(MambaCore)(d_model=128, d_state=16, d_conv=4, expand=2),
            bidirectional=True,
        )

        # 2D input: batch=2, spatial=(16, 16), channels=128
        x = torch.randn(2, 16, 16, 128)
        y = mamba(x)   # [2, 16, 16, 128]
    """

    def __init__(
        self,
        mamba_layer_cfg: LazyConfig,
        bidirectional: bool = False,
    ):
        """Initialise the Mamba-ND wrapper.

        Args:
            mamba_layer_cfg: :class:`~nvsubquadratic.lazy_config.LazyConfig`
                for the underlying 1D Mamba core.  The target class must
                accept a 3-D tensor of shape ``[B, S, C]`` (batch, sequence
                length, channels) and return a tensor of the same shape.
                Typical targets include ``mamba_ssm.Mamba`` and
                ``mamba_ssm.Mamba2``.  ``instantiate(mamba_layer_cfg)`` is
                called twice when ``bidirectional=True``; each call constructs
                a fresh ``nn.Module`` with newly initialised weights, so the
                two directions do not share parameters.
            bidirectional: If ``True``, run a second Mamba core on the
                reversed sequence and sum both outputs.  This doubles
                parameter count and compute but gives non-causal coverage
                of the full sequence -- strongly recommended for spatial
                tasks (images, volumes).  Defaults to ``False``.

        Raises:
            Exception: Propagated from
                :func:`~nvsubquadratic.lazy_config.instantiate` if
                ``mamba_layer_cfg`` cannot be constructed.  Check that the
                target class accepts ``[B, S, C]`` tensors and that all
                required constructor arguments are provided in the config.
        """
        super().__init__()
        self.bidirectional = bidirectional

        self.core_layer = instantiate(mamba_layer_cfg)
        # If bidirectional, we need to instantiate a reversed Mamba layer
        if self.bidirectional:
            self.core_layer_rev = instantiate(mamba_layer_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the Mamba SSM to an ND input signal.

        The forward pass performs the following steps:

        1. **Flatten** all spatial axes into one sequence dimension:
           ``[B, *spatial, C]`` to ``[B, S, C]``, where ``S = prod(spatial)``.
           The flattening follows row-major (C-contiguous) order.
        2. **Forward SSM**: ``out = core_layer(x)`` -- applies the selective
           SSM recurrence :math:`y_t = C_t(\bar{A}_t h_{t-1} + \bar{B}_t x_t)`.
        3. **Reverse SSM** (only when ``bidirectional=True``):
           ``out_rev = core_layer_rev(flip(x))`` -- runs the SSM on the
           reversed sequence, then flips back and adds to ``out``:

           .. math::

               \text{out} \mathrel{+}=
                   \mathrm{flip}(\mathrm{Mamba}_\mathrm{rev}(\mathrm{flip}(x)))

        4. **Reshape** back to the original spatial layout:
           ``[B, S, C]`` to ``[B, *spatial, C]``.

        Implementation note
        -------------------
        The local variable ``x`` is rebound to the flattened ``[B, S, C]``
        view after the ``rearrange`` call; the original spatial shape is
        preserved in ``x_shape`` for the final ``reshape``.

        Args:
            x: Input tensor of shape ``(B, *spatial, C)`` where ``B`` is
                batch size, ``spatial`` is one or more spatial dimensions
                (e.g. ``(T,)`` for 1D sequences, ``(H, W)`` for 2D images,
                ``(D, H, W)`` for 3D volumes), and ``C`` is the channel
                (hidden) dimension.  The tensor must be in channels-last
                (BHC / BHWc) layout, consistent with the rest of the library.

        Returns:
            Output tensor of shape ``(B, *spatial, C)`` -- same shape and
            layout as the input.  When ``bidirectional=True`` the output is
            the element-wise sum of the forward and reverse SSM outputs,
            which doubles the effective output magnitude compared to a
            unidirectional pass; downstream normalisation layers (e.g.
            ``RMSNorm`` inside the residual block) absorb this scale.
        """
        x_shape = x.shape
        # Reshape input to [B, flatten (* spatial_dims), hidden_dim
        x = rearrange(x, "b ... c -> b (...) c")

        # Forward pass through the core layer. It expects an input of shape [B, seq_len, hidden_dim].
        out = self.core_layer(x)

        # If bidirectional, reverse the input, apply the inverted layer, reverse back and add to
        # output of the core (forward) layer
        if self.bidirectional:
            out_rev = self.core_layer_rev(torch.flip(x, dims=[1]))
            out = out + torch.flip(out_rev, dims=[1])

        # Reshape output to original [B, * spatial_dims, hidden_dim] shape
        out = out.reshape(*x_shape)
        return out
