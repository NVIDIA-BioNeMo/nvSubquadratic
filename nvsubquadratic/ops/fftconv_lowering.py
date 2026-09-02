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

r"""``torch.compile`` lowering that rewrites 2D FFT convs onto the fused CUDA kernel.

Motivation
----------
Inductor cannot generate code for complex operators — it emits
``UserWarning: Torchinductor does not support code generation for complex
operators`` and falls back to eager cuFFT for the whole
``rfft2 → multiply → irfft2`` chain.  So ``torch.compile`` on its own buys
essentially nothing for :func:`nvsubquadratic.ops.fftconv.fftconv2d_fp32_bhl`.
This pass detects that chain and swaps in
:func:`nvsubquadratic.ops.fftconv_custom.fused_fftconv2d_bhl`, which runs the
whole pipeline as a single cuFFTDx launch in the input dtype.

It exists so that models already written against ``fft_backend="torch_fft"``
pick up the kernel without a config change.  Setting
``fft_backend="subq_ops_fused"`` explicitly is the more predictable route and
does not need this pass — see :mod:`nvsubquadratic.ops.fftconv_custom`.

Enabling it
-----------
Pass it to ``torch.compile`` directly::

    from nvsubquadratic.ops.fftconv_lowering import fused_fftconv2d_options

    compiled = torch.compile(model, options=fused_fftconv2d_options())

:func:`fused_fftconv2d_options` is scoped to that one compiled callable.  The
:class:`fused_fftconv2d_lowering` context manager does the same thing by
patching global inductor config, for when a trainer or framework owns the
``torch.compile`` call and you cannot pass ``options``.

Where the pass runs
-------------------
It is registered as an inductor **pre-grad** pass, which operates on the Dynamo
graph before AOTAutograd.  That placement matters: the replacement node is a
plain forward call, and autograd for it is derived from the custom op's
registered backward.  A post-grad pass would have to rewrite the forward and
the backward subgraphs consistently, which is far more fragile.

Matched pattern
---------------
The Dynamo graph for ``fftconv2d_fp32_bhl`` is fully static — the FFT sizes and
crop offsets are Python ints baked in as constants — so the pass can verify the
whole recipe rather than pattern-matching shapes structurally::

    x    ─[.to(float32)]─┐
                         ├─ rfft2(s=S, dim=(2,3)) ─┐
    k    ─[.to(float32)]─┘                         ├─ mul ─ irfft2(s=S, dim=(2,3))
                                                   ┘
        ─ [..., a:a+X, b:b+Y] ─ [.to(orig_dtype)]

A rewrite only happens when every one of these holds:

* ``S == (min(X + (Kx+1)//2, 2X), min(Y + (Ky+1)//2, 2Y))`` — the exact padding
  recipe from ``fftconv.py``, so an unrelated FFT conv is never touched.
* ``a == Kx // 2`` and ``b == Ky // 2`` — the reference's 'same' crop.
* The shapes fit the fused kernel (spatial <= 64 per axis; see
  :func:`~nvsubquadratic.ops.fftconv_custom.resolve_fused_fft_size`).
* The device is CUDA and the compute capability supports the required FFT tile.

Anything else is left alone, so the fallback is the unmodified eager-cuFFT path.

.. warning::
   ``fftconv.py`` has a ``COMPILE_COMPATIBLE`` flag that replaces the complex
   multiply with a real-valued ``_complex_mul_real``.  That variant produces a
   different subgraph and is **not** matched; the pass reports it via
   :func:`lowering_stats` rather than rewriting it.
"""

from __future__ import annotations


__all__ = [
    "FusedFFTConv2dLowering",
    "fused_fftconv2d_lowering",
    "fused_fftconv2d_options",
    "lowering_stats",
    "reset_lowering_stats",
]

import logging
import operator
from typing import Any

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass

from nvsubquadratic.ops.fftconv_custom import (
    FUSED_FFT_SIZE_128_MIN_ARCH,
    fused_fftconv2d_bhl,
    load_fused_fft_conv2d,
    resolve_fused_fft_size,
)


logger = logging.getLogger(__name__)

# Bumped whenever the matching or replacement logic changes. Inductor mixes this
# into its cache key, so a stale compiled artifact is never reused across a
# change to this file.
_PASS_VERSION = "1"

# Shared with the eager backend, which raises on the same condition; a silent
# graph rewrite must skip rather than raise, so the check is duplicated in
# behaviour but not in definition.
_FFT_SIZE_128_MIN_ARCH = FUSED_FFT_SIZE_128_MIN_ARCH

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# Rewrite counters, keyed by outcome. Useful in tests and when a user asks "did
# it actually fire?", which is the failure mode a silent pass is prone to.
_STATS: dict[str, int] = {}


def lowering_stats() -> dict[str, int]:
    """Return a snapshot of rewrite counters since the last reset.

    Keys are ``"rewritten"`` plus one ``"skipped:<reason>"`` entry per rejection
    cause (e.g. ``"skipped:spatial-too-large"``). A pass that silently does
    nothing is hard to debug; this is how you tell "did not match" from
    "matched but rejected", and why.

    .. note::
       Counters only advance when the pass actually runs. Inductor caches
       compiled artifacts on disk, and a cache hit skips every pre-grad pass —
       so empty counters can mean "served from cache" rather than "did not
       fire". Set ``torch._inductor.config.force_disable_caches = True`` when
       using this to verify behaviour.

    Returns:
        A copy of the counter dict.
    """
    return dict(_STATS)


def reset_lowering_stats() -> None:
    """Clear the rewrite counters returned by :func:`lowering_stats`."""
    _STATS.clear()


def _bump(key: str) -> None:
    _STATS[key] = _STATS.get(key, 0) + 1


def _skip(reason: str) -> None:
    _bump(f"skipped:{reason}")
    logger.debug("fused_fftconv2d lowering skipped a candidate: %s", reason)


def _meta_tensor(node: Any) -> torch.Tensor | None:
    """Return the example tensor Dynamo attached to ``node``, if any."""
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("example_value")
    if value is None:
        value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _unwrap_cast(node: torch.fx.Node) -> torch.fx.Node:
    """Strip a ``.to(float32)`` / ``.float()`` hop, returning the tensor behind it.

    The reference upcasts both operands before the FFT; the fused kernel takes
    the original dtype directly, so the cast is exactly what we want to drop.
    """
    if node.op == "call_method" and node.target in ("to", "float") and node.args:
        source = node.args[0]
        if isinstance(source, torch.fx.Node):
            return source
    if node.op == "call_function" and node.target is torch.Tensor.to and node.args:
        source = node.args[0]
        if isinstance(source, torch.fx.Node):
            return source
    return node


def _is_fft(node: Any, target) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "call_function" and node.target is target


def _fft_kwarg(node: torch.fx.Node, name: str, position: int) -> Any:
    """Read an FFT argument that may have been passed positionally or by keyword."""
    if name in node.kwargs:
        return node.kwargs[name]
    return node.args[position] if len(node.args) > position else None


def _as_int_pair(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _find_spectrum_operands(irfft: torch.fx.Node) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    """Recover the two ``rfft2`` nodes feeding an ``irfft2``.

    Handles both multiply spellings the reference can produce: the default
    in-place ``fft_x.mul_(fft_kernel)`` (where ``irfft2``'s own input is the
    *input* spectrum, mutated in place) and an out-of-place ``a * b``.
    """
    spectrum = irfft.args[0] if irfft.args else irfft.kwargs.get("input")
    if not isinstance(spectrum, torch.fx.Node):
        return None

    # In-place: irfft2 reads the mutated input spectrum directly.
    if _is_fft(spectrum, torch.fft.rfft2):
        for user in spectrum.users:
            if user.op == "call_method" and user.target == "mul_" and user.args and user.args[0] is spectrum:
                other = user.args[1] if len(user.args) > 1 else None
                if _is_fft(other, torch.fft.rfft2):
                    return spectrum, other
        return None

    # Out-of-place: irfft2 reads the product node.
    is_mul = (spectrum.op == "call_function" and spectrum.target in (operator.mul, torch.mul)) or (
        spectrum.op == "call_method" and spectrum.target == "mul"
    )
    if is_mul and len(spectrum.args) == 2:
        lhs, rhs = spectrum.args
        if _is_fft(lhs, torch.fft.rfft2) and _is_fft(rhs, torch.fft.rfft2):
            return lhs, rhs
    return None


def _parse_crop(irfft: torch.fx.Node) -> tuple[torch.fx.Node, int, int, int, int] | None:
    """Parse the ``[..., a:a+X, b:b+Y]`` crop that consumes ``irfft2``.

    Returns:
        ``(getitem_node, a, X, b, Y)``, or ``None`` if the sole consumer is not
        a trailing-two-axis slice of that exact form.
    """
    users = list(irfft.users)
    if len(users) != 1:
        return None
    getitem = users[0]
    if getitem.op != "call_function" or getitem.target is not operator.getitem:
        return None

    index = getitem.args[1]
    if not isinstance(index, tuple) or len(index) != 3:
        return None
    ellipsis, slice_x, slice_y = index
    if ellipsis is not Ellipsis or not isinstance(slice_x, slice) or not isinstance(slice_y, slice):
        return None
    if slice_x.step not in (None, 1) or slice_y.step not in (None, 1):
        return None
    if any(v is None for v in (slice_x.start, slice_x.stop, slice_y.start, slice_y.stop)):
        return None

    start_x, start_y = int(slice_x.start), int(slice_y.start)
    return getitem, start_x, int(slice_x.stop) - start_x, start_y, int(slice_y.stop) - start_y


def _arch_supports(device: torch.device, fft_size: int) -> bool:
    """Whether ``device`` can run the fused kernel at this FFT tile size."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if fft_size < 128:
        return True
    return torch.cuda.get_device_capability(device.index) >= _FFT_SIZE_128_MIN_ARCH


def _try_rewrite(graph: torch.fx.Graph, irfft: torch.fx.Node, allow_reduced_precision: bool) -> bool:
    """Attempt to replace one ``rfft2/mul/irfft2/crop`` chain. Returns whether it fired."""
    if _as_int_pair(_fft_kwarg(irfft, "dim", 2)) != (2, 3):
        _skip("irfft-dim-not-2-3")
        return False
    fft_shape = _as_int_pair(_fft_kwarg(irfft, "s", 1))
    if fft_shape is None:
        _skip("irfft-dynamic-shape")
        return False

    operands = _find_spectrum_operands(irfft)
    if operands is None:
        _skip("no-rfft2-product")
        return False

    crop = _parse_crop(irfft)
    if crop is None:
        _skip("no-same-crop")
        return False
    getitem, start_x, out_x, start_y, out_y = crop

    # Identify which spectrum is the input: its spatial extent equals the crop
    # length. (For the in-place spelling the order is already known, but keying
    # off shapes covers the out-of-place spelling too.)
    x_rfft = kernel_rfft = None
    for candidate, other in (operands, operands[::-1]):
        meta = _meta_tensor(candidate.args[0] if candidate.args else None)
        if meta is not None and meta.ndim == 4 and tuple(meta.shape[-2:]) == (out_x, out_y):
            x_rfft, kernel_rfft = candidate, other
            break
    if x_rfft is None:
        _skip("operand-roles-ambiguous")
        return False

    for rfft in (x_rfft, kernel_rfft):
        if _as_int_pair(_fft_kwarg(rfft, "s", 1)) != fft_shape or _as_int_pair(_fft_kwarg(rfft, "dim", 2)) != (2, 3):
            _skip("rfft-args-mismatch")
            return False

    x_node = _unwrap_cast(x_rfft.args[0])
    kernel_node = _unwrap_cast(kernel_rfft.args[0])
    x_meta, kernel_meta = _meta_tensor(x_node), _meta_tensor(kernel_node)
    if x_meta is None or kernel_meta is None:
        _skip("missing-shape-metadata")
        return False
    if x_meta.ndim != 4 or kernel_meta.ndim != 4:
        _skip("not-4d")
        return False

    dim_x, dim_y = int(x_meta.shape[-2]), int(x_meta.shape[-1])
    k_x, k_y = int(kernel_meta.shape[-2]), int(kernel_meta.shape[-1])

    # The recipe check. This is what keeps the pass from touching an unrelated
    # FFT convolution that merely has the same node types.
    expected_shape = (min(dim_x + (k_x + 1) // 2, 2 * dim_x), min(dim_y + (k_y + 1) // 2, 2 * dim_y))
    if fft_shape != expected_shape or (start_x, start_y) != (k_x // 2, k_y // 2):
        _skip("not-the-reference-recipe")
        return False
    if (out_x, out_y) != (dim_x, dim_y):
        _skip("crop-is-not-same-size")
        return False
    if int(kernel_meta.shape[1]) != int(x_meta.shape[1]) or int(kernel_meta.shape[0]) not in (1, int(x_meta.shape[0])):
        _skip("channel-or-batch-mismatch")
        return False

    if x_meta.dtype not in _SUPPORTED_DTYPES:
        _skip("unsupported-dtype")
        return False
    if not allow_reduced_precision and x_meta.dtype is not torch.float32:
        _skip("reduced-precision-disabled")
        return False

    try:
        fft_size = resolve_fused_fft_size(dim_x, dim_y, k_x, k_y)
    except ValueError:
        _skip("spatial-too-large")
        return False
    if not _arch_supports(x_meta.device, fft_size):
        _skip("arch-unsupported")
        return False

    # The reference casts its fp32 result back to the input dtype. The fused
    # kernel already returns the input dtype, so when that cast is present it is
    # the node to replace — otherwise the rewrite would leave a dangling cast.
    replace_target = getitem
    consumers = list(getitem.users)
    if len(consumers) == 1 and consumers[0].op == "call_method" and consumers[0].target == "to":
        cast = consumers[0]
        cast_meta = _meta_tensor(cast)
        if cast_meta is not None and cast_meta.dtype == x_meta.dtype:
            replace_target = cast

    with graph.inserting_before(replace_target):
        fused = graph.call_function(fused_fftconv2d_bhl, args=(x_node, kernel_node))
    fused.meta.update(replace_target.meta)
    replace_target.replace_all_uses_with(fused)

    _bump("rewritten")
    logger.debug(
        "fused_fftconv2d lowering rewrote a %sx%s conv (kernel %sx%s, fft_size=%s, dtype=%s)",
        dim_x,
        dim_y,
        k_x,
        k_y,
        fft_size,
        x_meta.dtype,
    )
    return True


class FusedFFTConv2dLowering(CustomGraphPass):
    """Inductor pre-grad pass replacing the 2D FFT-conv chain with the fused kernel.

    Install it with :func:`fused_fftconv2d_lowering`, which handles the inductor
    config plumbing and restores any previously registered pass on exit.

    Args:
        allow_reduced_precision: When ``True`` (default) fp16/bf16 graphs are
            rewritten too, which changes the convolution from fp32-internal to
            native-dtype — the main source of the speedup, and a real numerics
            change (~2e-3 normwise in bf16). Set ``False`` to restrict the
            rewrite to fp32 graphs, where it is numerically neutral.
    """

    def __init__(self, allow_reduced_precision: bool = True):
        """Build the pass and force the fused kernel's operators to register.

        Args:
            allow_reduced_precision: See the class docstring.

        Raises:
            ImportError: If ``subquadratic_ops_torch`` is not installed.
        """
        # Registration has to happen up front: a compiled artifact restored from
        # inductor's on-disk cache skips this pass entirely and calls
        # torch.ops.subquadratic_ops_torch.* directly, so relying on the
        # wrappers' lazy import would fail with an op-namespace AttributeError
        # on that first (cache-hit) call. Doing it here also turns a missing
        # install into a clear ImportError at setup time.
        load_fused_fft_conv2d()
        self.allow_reduced_precision = allow_reduced_precision

    def __call__(self, graph: torch.fx.Graph) -> torch.fx.Graph:
        """Rewrite every matching FFT-conv chain in ``graph``, in place."""
        candidates = [n for n in graph.nodes if _is_fft(n, torch.fft.irfft2)]
        if not candidates:
            return graph

        if any(_try_rewrite(graph, node, self.allow_reduced_precision) for node in candidates):
            # Drops the now-unused rfft2/mul/irfft2/crop nodes. The in-place
            # ``mul_`` is dead too: its only consumer was the irfft2 we replaced.
            graph.eliminate_dead_code()
            graph.lint()
        return graph

    def uuid(self) -> str:
        """Cache key contribution, so inductor never reuses artifacts across changes here."""
        return f"nvsubquadratic-fused-fftconv2d-lowering-v{_PASS_VERSION}-rp{int(self.allow_reduced_precision)}"


def fused_fftconv2d_options(allow_reduced_precision: bool = True) -> dict[str, Any]:
    """Return a ``torch.compile(options=...)`` dict that enables the lowering.

    This is the preferred way to enable the pass. Inductor's ``options`` are
    applied as a config patch scoped to that one compiled callable, so unlike
    :class:`fused_fftconv2d_lowering` it neither mutates global inductor state
    nor conflicts with a ``pre_grad_custom_pass`` registered elsewhere.

    Args:
        allow_reduced_precision: See :class:`FusedFFTConv2dLowering`.

    Returns:
        A dict to pass as ``torch.compile(..., options=...)``.

    Raises:
        ImportError: If ``subquadratic_ops_torch`` is not installed.

    Example:
        >>> import torch
        >>> from nvsubquadratic.ops.fftconv_lowering import fused_fftconv2d_options
        >>> compiled = torch.compile(model, options=fused_fftconv2d_options())  # doctest: +SKIP
        >>> out = compiled(x)  # doctest: +SKIP
    """
    return {"pre_grad_custom_pass": FusedFFTConv2dLowering(allow_reduced_precision)}


class fused_fftconv2d_lowering:  # Lowercase: only ever used as a context manager.
    """Context manager that installs :class:`FusedFFTConv2dLowering` globally.

    Prefer :func:`fused_fftconv2d_options` when you control the
    ``torch.compile`` call — it is scoped to a single callable instead of
    patching global inductor config. Reach for this context manager when
    something else compiles the model for you (a trainer, a framework entry
    point) and you cannot pass ``options``.

    ``torch.compile`` caches compiled artifacts, so enter this **before** the
    first compiled call on the model you want rewritten; a function already
    compiled without the pass keeps its cached code.

    Args:
        allow_reduced_precision: See :class:`FusedFFTConv2dLowering`.

    Example:
        >>> import torch
        >>> from nvsubquadratic.ops.fftconv_lowering import (
        ...     fused_fftconv2d_lowering,
        ...     lowering_stats,
        ... )
        >>> with fused_fftconv2d_lowering():  # doctest: +SKIP
        ...     compiled = torch.compile(model)
        ...     out = compiled(x)
        >>> lowering_stats().get("rewritten", 0)  # doctest: +SKIP
        1

    .. note::
       Inductor exposes a single ``pre_grad_custom_pass`` slot. Any pass already
       registered is saved and restored on exit, but the two do not compose
       while this one is active.
    """

    def __init__(self, allow_reduced_precision: bool = True):
        """Build the pass eagerly, so a missing install fails here, not at compile time.

        Args:
            allow_reduced_precision: See :class:`FusedFFTConv2dLowering`.
        """
        self.pass_ = FusedFFTConv2dLowering(allow_reduced_precision)
        self._previous: Any = None

    def __enter__(self) -> FusedFFTConv2dLowering:
        """Install the pass, saving whatever was registered before.

        Returns:
            The installed :class:`FusedFFTConv2dLowering` instance.
        """
        from torch._inductor import config as inductor_config

        self._previous = inductor_config.pre_grad_custom_pass
        if self._previous is not None:
            logger.warning(
                "Replacing an already-registered inductor pre_grad_custom_pass (%r) for the "
                "duration of fused_fftconv2d_lowering; it will be restored on exit.",
                self._previous,
            )
        inductor_config.pre_grad_custom_pass = self.pass_
        return self.pass_

    def __exit__(self, *exc_info: object) -> None:
        """Restore the previously registered pass, including on an exception."""
        from torch._inductor import config as inductor_config

        inductor_config.pre_grad_custom_pass = self._previous
        self._previous = None
