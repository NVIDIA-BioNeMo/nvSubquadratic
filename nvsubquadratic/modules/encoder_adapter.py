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

r"""Attach zero-init parallel adapters to arbitrary pretrained ViT-style encoders.

Generalises :class:`~nvsubquadratic.modules.vit5_parallel_hyena_adapter.ViT5ParallelHyenaSequenceMixer`
from the in-repo ViT-5 stack to external encoders (timm, HuggingFace, DINOv2).
Two problems have to be solved that the in-repo case avoids by construction:

**1. Calling convention.** The adapter needs attention to be a pure
``[B, T, C] -> [B, T, C]`` function.  Real modules differ: ``timm.models.vision_transformer.Attention``
already matches, while HuggingFace self-attention returns a tuple
``(context, attn_weights)``.  :class:`AttentionShim` normalises both.

**2. Prefix tokens.** Pretrained ViTs prepend a CLS token and (in DINOv2-with-registers)
several register tokens, giving ``T = num_prefix + grid_h * grid_w``.  That total
generally does **not** factor into a rectangle — DINOv2-with-registers at 224/14 has
``T = 1 + 4 + 256 = 261`` and ``261 % 16 == 5`` — so the sequence cannot be reshaped to
a spatial grid directly.

:class:`PrefixAwareParallelAdapter` handles this by splitting the prefix off inside the
adapter branch, mixing only the patch tokens, and writing **exactly zero** to the prefix
positions.  The base model's own attention branch continues to update CLS and registers
normally; the adapter never touches them.  This preserves the zero-init bit-exactness
guarantee trivially at the prefix positions and avoids assigning CLS a meaningless
spatial coordinate.

**Registers are not required, but conditioning is.**  Hyena needs only a well-formed
grid; registers are irrelevant to that.  Input dependence, however, is *constitutive*
of HyenaND — Wessels et al. (arXiv:2607.19378) define the operator as

.. code-block:: text

    K_i(x) = w(c_i) ⊙ f_θ(c_i; z(x))

and implement the dependence on the control variable ``z(x)`` with FiLM over the SIREN
kernel's hidden layers.  An adapter built with ``film_cfg=None`` is therefore a *static*
implicit long convolution, **not** HyenaND, and should not be reported as one.

``conditioning_source`` selects where ``z(x)`` comes from.  The default,
``"patch_mean"``, pools the patch tokens the adapter already reads — the right choice
for a frozen encoder, since it relies on nothing the base model had to learn on the
adapter's behalf.  ``"prefix"`` mirrors ``RegisterPooling`` in the in-repo ViT-5
ImageNet configs, which is sound when the registers were trained as part of a Hyena
model but weaker for an off-the-shelf encoder whose registers serve another purpose.

Two properties make conditioning cheap and safe here: ``z(x)`` is pooled once per
instance, so the inner mixer still issues a single ND FFT call; and FiLM initialises to
``(γ, β) = (1, 0)``, leaving the kernel unmodulated at init and preserving the zero-init
bit-exactness guarantee.

Caveat worth carrying: neither the paper nor this repo contains a clean FiLM-on vs
FiLM-off ablation at matched settings.  The evidence supports "input-dependent kernels
are the proposed operator", not "conditioning has been shown to beat a static kernel by
X".  If that comparison matters to a result, run it explicitly with
``conditioning_source="none"`` as the static-kernel arm.

Typical use::

    from nvsubquadratic.modules.encoder_adapter import (
        attach_parallel_adapters, freeze_except_adapters, verify_zero_init_identity,
    )

    ok, report = verify_zero_init_identity(
        model, sample_input,
        lambda m: attach_parallel_adapters(m, inner_mixer_factory=..., hidden_dim=384,
                                           rank=32, grid_w=16, num_prefix_tokens=1),
    )
    assert ok, report
    freeze_except_adapters(model, also_train=("head",))
"""

import math
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn


# Leaf attribute names that commonly hold a self-attention module.
DEFAULT_ATTENTION_PATTERNS = ("attn", "attention", "self_attn", "self_attention")


class AttentionShim(nn.Module):
    r"""Normalise a pretrained attention module to ``[B, T, C] -> [B, T, C]``.

    Wraps an arbitrary attention module and returns only its hidden-state output,
    discarding auxiliary returns.  Handles the two conventions seen across the
    ViT ecosystem:

    * ``timm`` ``Attention.forward(x) -> Tensor`` — returned unchanged.
    * HuggingFace ``ViTSelfAttention`` / ``Dinov2SelfAttention`` —
      ``forward(hidden_states, ...) -> tuple[Tensor, ...]``; element 0 is taken.

    Objects exposing a ``last_hidden_state`` attribute (HF ``ModelOutput``) are
    also unwrapped.

    Args:
        attn: The pretrained attention module to wrap.

    Attributes:
        attn (nn.Module): The wrapped module.  Its parameters are untouched, so
            state-dict keys gain one ``.attn.`` level of nesting.
    """

    def __init__(self, attn: nn.Module) -> None:
        """Store the wrapped attention module."""
        super().__init__()
        self.attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Call the wrapped attention and return its hidden states.

        Args:
            x: Token sequence ``[B, T, C]``.

        Returns:
            Tensor ``[B, T, C]`` — the attention output with any auxiliary
            returns (attention probabilities, caches) discarded.
        """
        out = self.attn(x)
        if isinstance(out, tuple):
            out = out[0]
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        return out


class PrefixAwareParallelAdapter(nn.Module):
    r"""Attention plus a zero-init bottleneck branch that skips prefix tokens.

    Computes::

        out[:, prefix] = attn(x)[:, prefix]                              # + 0
        out[:, patch]  = attn(x)[:, patch] + W_up(mixer(W_down(x_patch)))

    ``W_up`` is zero-initialised, so the whole module is bit-exactly equal to the
    wrapped attention at step 0.  Because the adapter writes literal zeros at the
    prefix positions, that guarantee holds there unconditionally — even after
    ``W_up`` moves off zero during training.

    Data flow::

        x: [B, T, C]                    T = P + N,  N = grid_h * grid_w
          │
          ├─► shim(attn(x)) ─────────────────────────────────────────►┐
          │                                                            │
          └─► split ──► x_patch: [B, N, C]                            │
                          │                                            │
                          ▼ w_down          [B, N, r]                 │
                          ▼ inner_mixer     [B, N, r]                 │
                          ▼ w_up            [B, N, C]                 │
                          │                                            │
                     pad prefix with zeros ──► [B, T, C] ────────────►(+)
                                                                       ▼
                                                                 [B, T, C]

    Args:
        attn: Pretrained attention module.  Wrapped in :class:`AttentionShim`
            unless ``shim_attention=False``.
        hidden_dim: Encoder hidden width ``C``.
        rank: Bottleneck width ``r``.  ``W_down: C -> r``, ``W_up: r -> C``.
        inner_mixer: Token mixer operating on ``[B, N, r]`` and returning the
            same shape.  ``None`` gives ``nn.Identity`` (the C0 capacity
            ablation).
        num_prefix_tokens: Count of non-spatial tokens ``P`` (CLS + registers).
            ``0`` for a pure patch sequence.
        prefix_position: ``"first"`` for the ``[CLS, registers, patches]`` layout
            used by timm / HuggingFace / DINOv2, ``"last"`` for the
            ``[patches, CLS, registers]`` layout used by
            :class:`~nvsubquadratic.networks.vit5_classification.ViT5ClassificationNet`.
        shim_attention: Wrap ``attn`` in :class:`AttentionShim`.  Set ``False``
            when the module already returns a bare tensor and you want to keep
            state-dict keys flat.
        conditioning_source: Where the FiLM control variable ``z(x)`` comes from.

            * ``"patch_mean"`` (default) — mean-pool the patch tokens the adapter
              already reads into ``(B, hidden_dim)``.  This is the recommended
              source when adapting a **frozen** encoder: it depends on nothing
              the base model had to learn on the adapter's behalf.
            * ``"prefix"`` — mean-pool the CLS/register tokens instead, mirroring
              ``RegisterPooling`` in the in-repo ViT-5 ImageNet configs.  Sound
              when the encoder's registers were themselves trained as part of a
              Hyena model; weaker for an off-the-shelf encoder whose registers
              were trained for another purpose (in DINOv2, absorbing attention
              artifacts).  Requires ``num_prefix_tokens > 0``.
            * ``"none"`` — unconditional kernel.  Note this makes the branch a
              *static* implicit long convolution, **not** HyenaND: input
              dependence is constitutive of the operator in Wessels et al.
              (arXiv:2607.19378), which defines
              ``K_i(x) = w(c_i) ⊙ f_θ(c_i; z(x))``.  Useful only as an explicit
              static-kernel ablation.

            Conditioning requires an inner mixer that accepts a ``conditioning``
            keyword — the :class:`~nvsubquadratic.modules.vit5_hyena_adapter.ViT5HyenaAdapter`
            → :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`
            → :class:`~nvsubquadratic.modules.hyena_nd.Hyena` chain — whose kernel
            must be built with a ``film_cfg`` sized for ``cond_dim=hidden_dim``.
            ``nn.Identity`` and a plain depthwise conv do **not** accept the kwarg
            and will raise; conditioning is meaningless for them, so pair them
            with ``"none"``.

            ``z(x)`` is pooled once per instance, so the single ND FFT call is
            preserved — conditioning costs one MLP, not per-position kernels.
            FiLM initialises to ``(γ, β) = (1, 0)``, i.e. unmodulated at init, so
            it composes with the zero-init ``W_up`` guarantee.

            Prefix outputs stay exactly zero under every setting — the adapter
            may *read* prefix tokens but never writes to them.

    Attributes:
        attn (nn.Module): Shimmed (or raw) attention module.
        w_down (nn.Linear): ``C -> r``, ``bias=False``.
        inner_mixer (nn.Module): Bottleneck-width token mixer.
        w_up (nn.Linear): ``r -> C``, ``bias=False``, zero-initialised.
    """

    def __init__(
        self,
        attn: nn.Module,
        hidden_dim: int,
        rank: int,
        inner_mixer: Optional[nn.Module] = None,
        num_prefix_tokens: int = 0,
        prefix_position: Literal["first", "last"] = "first",
        shim_attention: bool = True,
        conditioning_source: Literal["patch_mean", "prefix", "none"] = "patch_mean",
        grid_hw: Optional[tuple[int, int]] = None,
        grid_mixer: bool = True,
    ) -> None:
        """Wrap ``attn``, build the bottleneck projections, and zero ``W_up``."""
        super().__init__()
        if prefix_position not in ("first", "last"):
            raise ValueError(f"prefix_position must be 'first' or 'last', got {prefix_position!r}")
        if conditioning_source not in ("patch_mean", "prefix", "none"):
            raise ValueError(
                f"conditioning_source must be 'patch_mean', 'prefix', or 'none', got {conditioning_source!r}"
            )
        if conditioning_source == "prefix" and num_prefix_tokens == 0:
            raise ValueError("conditioning_source='prefix' requires num_prefix_tokens > 0")

        self.attn = AttentionShim(attn) if shim_attention else attn
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_prefix_tokens = num_prefix_tokens
        self.prefix_position = prefix_position
        self.conditioning_source = conditioning_source
        self.grid_hw: Optional[tuple[int, int]] = grid_hw

        self.w_down = nn.Linear(hidden_dim, rank, bias=False)
        self.inner_mixer = inner_mixer if inner_mixer is not None else nn.Identity()
        # Grid mixers (Hyena, 2-D conv) consume [B, H, W, r] and need the patch
        # tokens reshaped; nn.Identity and per-token maps consume [B, N, r].
        # Forced off for Identity so the C0 ablation needs no special-casing.
        self.grid_mixer: bool = grid_mixer and inner_mixer is not None
        self.w_up = nn.Linear(rank, hidden_dim, bias=False)

        nn.init.zeros_(self.w_up.weight)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Return the patch-token slice and the offset at which it starts."""
        p = self.num_prefix_tokens
        if p == 0:
            return x, 0
        if self.prefix_position == "first":
            return x[:, p:], p
        return x[:, :-p], 0

    def _prefix(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the prefix-token slice, or ``None`` when there are none."""
        p = self.num_prefix_tokens
        if p == 0:
            return None
        return x[:, :p] if self.prefix_position == "first" else x[:, -p:]

    def _grid(self, num_patches: int) -> tuple[int, int]:
        r"""Resolve the ``(H, W)`` patch grid for this forward pass.

        Derived per call rather than stored, so a single adapter instance serves
        every resolution an any-resolution encoder produces (C-RADIO's CPE
        supports inputs up to 2048 px in steps of the patch size).  The mixers
        below are already resolution-agnostic — a Hyena kernel is regenerated at
        ``2L-1`` per axis on each call — so nothing else needs to change.

        Resolution order:

        1. ``self.grid_hw`` when set explicitly (use for non-square inputs, via
           :meth:`set_grid_hw`).
        2. ``isqrt(num_patches)`` when the count is a perfect square.

        Args:
            num_patches: Number of patch tokens ``N`` (prefix already removed).

        Returns:
            ``(H, W)`` with ``H * W == num_patches``.

        Raises:
            ValueError: If ``num_patches`` is not a perfect square and no
                explicit grid was set — the shape is genuinely ambiguous and
                guessing would silently mix over a wrong layout.
        """
        if self.grid_hw is not None:
            h, w = self.grid_hw
            if h * w != num_patches:
                raise ValueError(
                    f"grid_hw={self.grid_hw} implies {h * w} patches but got {num_patches}. "
                    "Call set_grid_hw() with the current resolution, or clear it to infer."
                )
            return h, w

        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError(
                f"{num_patches} patch tokens is not a perfect square, so the grid is ambiguous. "
                "Call set_grid_hw((H, W)) before the forward pass for non-square inputs."
            )
        return side, side

    def set_grid_hw(self, grid_hw: Optional[tuple[int, int]]) -> None:
        r"""Set (or clear) the explicit patch grid for non-square inputs.

        Args:
            grid_hw: ``(H, W)`` patch-grid dimensions, or ``None`` to go back to
                inferring a square grid from the token count.
        """
        self.grid_hw = grid_hw

    def _scatter(self, h: torch.Tensor, total_tokens: int, offset: int) -> torch.Tensor:
        """Place patch-token output ``h`` into a zero tensor of length ``total_tokens``."""
        if h.shape[1] == total_tokens:
            return h
        out = h.new_zeros(h.shape[0], total_tokens, h.shape[2])
        out[:, offset : offset + h.shape[1]] = h
        return out

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        r"""Apply attention plus the prefix-skipping zero-init adapter.

        Args:
            x: Token sequence ``[B, T, C]`` with ``T = num_prefix_tokens + N``
                and ``N`` divisible by the inner mixer's grid width.
            **_kwargs: Accepted for interface compatibility and ignored.  Any
                conditioning is derived internally from ``conditioning_source``,
                not taken from the caller.

        Returns:
            Tensor ``[B, T, C]``.  Prefix positions receive the attention output
            plus exactly ``0`` from the adapter branch.
        """
        attn_out = self.attn(x)

        patches, offset = self._split(x)
        B, N, _ = patches.shape
        h = self.w_down(patches)

        # Reshape to the 2-D patch grid, sized from THIS forward pass so one
        # adapter instance serves every resolution the encoder emits.
        if self.grid_mixer:
            H, W = self._grid(N)
            h = h.reshape(B, H, W, self.rank)

        # z(x): pooled once per instance, so a Hyena inner mixer still issues a
        # single ND FFT call regardless of conditioning.
        if self.conditioning_source == "patch_mean":
            h = self.inner_mixer(h, conditioning=patches.mean(dim=1))
        elif self.conditioning_source == "prefix":
            # Read the prefix for FiLM only; the adapter still writes zero there.
            h = self.inner_mixer(h, conditioning=self._prefix(x).mean(dim=1))
        else:
            h = self.inner_mixer(h)

        if self.grid_mixer:
            h = h.reshape(B, N, self.rank)

        h = self._scatter(self.w_up(h), x.shape[1], offset)

        return attn_out + h

    def extra_repr(self) -> str:
        """Return a summary of width, rank, prefix handling, and conditioning."""
        return (
            f"hidden_dim={self.hidden_dim}, rank={self.rank}, "
            f"num_prefix_tokens={self.num_prefix_tokens}, prefix_position={self.prefix_position}, "
            f"conditioning_source={self.conditioning_source}"
        )


def find_attention_modules(
    model: nn.Module,
    patterns: tuple[str, ...] = DEFAULT_ATTENTION_PATTERNS,
) -> list[tuple[str, nn.Module]]:
    r"""Locate outermost attention modules by leaf attribute name.

    Matches modules whose final path component is in ``patterns``, then drops any
    match nested inside another match.  This keeps ``blocks.0.attn`` and discards
    the inner ``blocks.0.attn.attention`` that HuggingFace models expose, so each
    block is adapted exactly once.

    Args:
        model: Module to search.
        patterns: Leaf attribute names treated as attention modules.

    Returns:
        List of ``(dotted_path, module)`` in ``named_modules`` order, outermost
        matches only.  Empty if nothing matched — inspect the module tree and
        pass an explicit ``patterns`` tuple in that case.
    """
    matches = [(name, mod) for name, mod in model.named_modules() if name and name.split(".")[-1] in patterns]
    return [(n, m) for n, m in matches if not any(n.startswith(other + ".") for other, _ in matches)]


def _set_submodule(model: nn.Module, dotted_path: str, new_module: nn.Module) -> None:
    """Replace the submodule at ``dotted_path`` with ``new_module`` in place."""
    parts = dotted_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def attach_parallel_adapters(
    model: nn.Module,
    hidden_dim: int,
    rank: int,
    inner_mixer_factory: Optional[Callable[[], nn.Module]] = None,
    num_prefix_tokens: int = 0,
    prefix_position: Literal["first", "last"] = "first",
    patterns: tuple[str, ...] = DEFAULT_ATTENTION_PATTERNS,
    shim_attention: bool = True,
    conditioning_source: Literal["patch_mean", "prefix", "none"] = "patch_mean",
) -> list[str]:
    r"""Wrap every attention module in ``model`` with a zero-init parallel adapter.

    Mutates ``model`` in place.  Because ``W_up`` is zero-initialised in every
    adapter, the model's function is unchanged immediately after this call —
    verify that with :func:`verify_zero_init_identity` before training.

    Args:
        model: Encoder to adapt, modified in place.
        hidden_dim: Encoder hidden width ``C``.
        rank: Bottleneck width ``r``.
        inner_mixer_factory: Zero-argument callable returning a fresh
            ``[B, N, r] -> [B, N, r]`` mixer.  Called once per attention module,
            so each block gets independent weights.  ``None`` gives
            ``nn.Identity`` everywhere (C0 ablation).
        num_prefix_tokens: Count of CLS + register tokens.
        prefix_position: ``"first"`` (timm / HF / DINOv2) or ``"last"`` (ViT-5).
        patterns: Leaf attribute names identifying attention modules.
        shim_attention: Wrap each attention module in :class:`AttentionShim`.
        conditioning_source: Source of the FiLM control variable ``z(x)`` —
            ``"patch_mean"`` (default), ``"prefix"``, or ``"none"``.  Use
            ``"none"`` only with a mixer that rejects the kwarg (``nn.Identity``,
            depthwise conv), and note that it makes a Hyena branch a *static*
            long convolution rather than HyenaND.  See
            :class:`PrefixAwareParallelAdapter`.

    Returns:
        Dotted paths of the wrapped modules, in attachment order.

    Raises:
        ValueError: If no attention module matches ``patterns``.
    """
    targets = find_attention_modules(model, patterns=patterns)
    if not targets:
        raise ValueError(
            f"No attention modules found with leaf names {patterns}. "
            "Inspect the module tree and pass an explicit `patterns` tuple."
        )

    for path, attn in targets:
        _set_submodule(
            model,
            path,
            PrefixAwareParallelAdapter(
                attn=attn,
                hidden_dim=hidden_dim,
                rank=rank,
                inner_mixer=inner_mixer_factory() if inner_mixer_factory is not None else None,
                num_prefix_tokens=num_prefix_tokens,
                prefix_position=prefix_position,
                shim_attention=shim_attention,
                conditioning_source=conditioning_source,
            ),
        )
    return [path for path, _ in targets]


def freeze_except_adapters(
    model: nn.Module,
    also_train: tuple[str, ...] = ("head",),
    adapter_markers: tuple[str, ...] = ("w_down", "w_up", "inner_mixer"),
) -> tuple[int, int]:
    r"""Freeze the backbone, leaving adapter parameters and the head trainable.

    Sets ``requires_grad = False`` on every parameter, then re-enables those whose
    name contains an adapter marker or any substring in ``also_train``.

    Args:
        model: Model with adapters already attached.
        also_train: Name substrings to keep trainable alongside the adapters
            (typically the classification head).
        adapter_markers: Name substrings identifying adapter parameters.

    Returns:
        ``(trainable_params, total_params)`` element counts, for reporting the
        trainable fraction.
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(marker in name for marker in adapter_markers) or any(key in name for key in also_train):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _as_tensor(output) -> torch.Tensor:
    """Reduce a model output (tensor / dict / tuple / ModelOutput) to one tensor."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "last_hidden_state", "pooler_output"):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Cannot extract a tensor from model output of type {type(output)}")


def verify_zero_init_identity(
    model: nn.Module,
    sample_input,
    attach_fn: Callable[[nn.Module], object],
) -> tuple[bool, str]:
    r"""Assert that attaching adapters leaves the model bit-exactly unchanged.

    Runs ``model`` in eval mode before and after ``attach_fn``, comparing with
    :func:`torch.equal` rather than :func:`torch.allclose` — an approximate match
    would hide a real wiring error such as a non-zero ``W_up`` or a misplaced
    prefix split.

    Eval mode matters: dropout and stochastic depth break determinism
    independently of the adapter.

    Args:
        model: Encoder without adapters, mutated in place by ``attach_fn``.
        sample_input: Input accepted by ``model.forward`` — a tensor, or a dict
            splatted as keyword arguments.
        attach_fn: Callable applying the adapters to ``model``.

    Returns:
        ``(passed, report)``.  ``report`` names the max absolute deviation and
        the likely cause when the check fails.
    """

    def run() -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            out = model(**sample_input) if isinstance(sample_input, dict) else model(sample_input)
        return _as_tensor(out)

    y_before = run()
    attach_fn(model)
    y_after = run()

    if y_before.shape != y_after.shape:
        return False, f"FAIL: output shape changed {tuple(y_before.shape)} -> {tuple(y_after.shape)}"

    if torch.equal(y_before, y_after):
        return True, "PASS: outputs are bit-exactly identical after attachment."

    max_diff = (y_before - y_after).abs().max().item()
    return False, (
        f"FAIL: outputs differ, max|delta| = {max_diff:.3e}. "
        "Check that every w_up.weight is exactly zero and has bias=False, that the "
        "prefix split matches the encoder's token layout, and that the model was in "
        "eval mode (dropout / drop-path break determinism on their own)."
    )


def check_gradient_unlock(
    model: nn.Module,
    sample_input,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[bool, str]:
    r"""Check the expected step-0 gradient pattern for a zero-init adapter.

    With ``W_up = 0`` the gradient reaching ``W_down`` is exactly zero, because
    the path from ``W_down`` to the loss is multiplied by ``W_up``.  Only
    ``W_up`` moves on the first step; the rest of the branch unlocks once
    ``W_up`` is non-zero.  A branch that stays dead is indistinguishable from
    "the adapter did not help" in a loss curve, so this is worth asserting.

    Args:
        model: Model with adapters attached.
        sample_input: Input accepted by ``model.forward``.
        loss_fn: Maps the extracted output tensor to a scalar loss.

            **Do not pass ``torch.sum``.** Most ViTs end in LayerNorm, whose
            output is zero-mean along the channel axis, so ``sum()`` is
            identically ~0 for any input and its gradient is *exactly zero* --
            indistinguishable from a dead adapter. Use ``(o**2).sum()``, a random
            linear head, or the real task loss.

    Returns:
        ``(passed, report)`` describing the observed gradient pattern.
    """
    model.zero_grad(set_to_none=True)
    out = model(**sample_input) if isinstance(sample_input, dict) else model(sample_input)
    loss_fn(_as_tensor(out)).backward()

    up_grads, down_grads = [], []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "w_up" in name:
            up_grads.append(param.grad.abs().sum().item())
        elif "w_down" in name:
            down_grads.append(param.grad.abs().sum().item())

    if not up_grads:
        return False, "FAIL: no w_up gradients found — adapters may not be attached or not on the forward path."
    if all(g == 0.0 for g in up_grads):
        return False, "FAIL: all w_up gradients are zero — the adapter branch is not reaching the loss."
    if any(g != 0.0 for g in down_grads):
        return False, (
            "FAIL: a w_down gradient is non-zero at step 0. With w_up == 0 it must be exactly zero; "
            "a non-zero value means w_up was not zero-initialised."
        )
    return True, (
        f"PASS: {len(up_grads)} w_up tensors have non-zero grad, "
        f"all {len(down_grads)} w_down tensors are exactly zero (expected at step 0)."
    )
