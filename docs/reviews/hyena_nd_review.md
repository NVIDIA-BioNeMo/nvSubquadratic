# Reviewer feedback: `nvsubquadratic/modules/hyena_nd.py`

Reviewed against the Phase-1 docstrings added in commit `docs(write/hyena_nd)`.

______________________________________________________________________

## 1. Module docstring — paper citation is incomplete and imprecise

**Text:** `"Poli et al., 'Hyena Hierarchy: Towards Larger Convolutional Language Models', ICML 2023."`

**Issue:** No arXiv ID or DOI. Collaborators without institutional access cannot look the paper up quickly. Also, the paper was published at ICML 2023 but first appeared on arXiv — give both.

**Fix:** Change to:

```
Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language Models",
ICML 2023.  arXiv:2302.10866.
```

______________________________________________________________________

## 2. Module docstring — "Section 3" reference is too vague

**Text (class docstring):** `"The two-gate structure corresponds to the H3-style decomposition described in Section 3."`

**Issue:** The Hyena paper is 30+ pages; "Section 3" does not pin the exact location. H3 is described in a *related-work* paper (Fu et al., 2023), not in the Hyena paper itself. An external reader will spend time searching.

**Fix:** Replace with:

```
The two-gate structure follows the H3 block (Fu et al., "Hungry Hungry Hippos",
ICLR 2023, arXiv:2212.14052, Section 3.2) and is generalised in Hyena
(Poli et al., arXiv:2302.10866, Section 3, "The Hyena Recurrence").
```

______________________________________________________________________

## 3. Module docstring — ND generalisation section does not mention boundary conditions

**Text:** `"For 2D/3D signals the convolution is non-causal by default; causal 1D mode is preserved."`

**Issue:** A new collaborator will not know whether 2D/3D uses zero-padding (linear conv) or circular conv. This matters because the choice affects the FLOP count and which `fftconv` function is invoked. The `mixed_fftconv` path (#120) also exists now.

**Fix:** Add after the sentence:

```
By default the 2D/3D path uses zero-padded (linear) FFT convolution
(``fftconv2d`` / ``fftconv3d``), matching ``torch.nn.ConvNd(padding='same')``
semantics.  Set the ``circular`` flag on ``CKConvND`` to switch to periodic
boundary conditions, or use ``mixed_fftconv`` for per-axis mixed BCs
(see ``nvsubquadratic.ops.mixed_fftconv``).
```

______________________________________________________________________

## 4. Class docstring — `Attributes` block: `k_norm` type annotation is contradictory

**Text:**

```
k_norm (torch.nn.Module): Per-channel normalisation for K.
    ``Identity`` when the gate is nonlinear (magnitude already bounded
    by :math:`\sigma`); a fresh instance of ``qk_norm_cfg`` otherwise.
    ``None`` when ``qk_norm_cfg`` is ``None``.
```

**Issue:** The type is listed as `torch.nn.Module` but the final sentence says it can be `None`. The actual code sets `self.k_norm = None` when `qk_norm_cfg is None`. The type should be `torch.nn.Module | None`.

**Fix:** Change to `k_norm (torch.nn.Module | None):` and restate the None case first:

```
k_norm (torch.nn.Module | None): Per-channel normalisation for K.
    ``None`` when ``qk_norm_cfg`` is ``None`` (QK-norm entirely disabled).
    ``torch.nn.Identity`` when the gate is nonlinear (σ already bounds
    K's magnitude); a fresh instance of ``qk_norm_cfg`` when the gate is
    ``Identity`` (linear gating).
```

______________________________________________________________________

## 5. `__init__` docstring — `output_norm_cfg` default value is misleading

**Text:** ``` "Defaults to ``Identity`` (no normalisation)." ```

**Issue:** The actual default is `LazyConfig(torch.nn.Identity)()`, not a plain `torch.nn.Identity`. This distinction matters because `LazyConfig(torch.nn.Identity)()` is a frozen lazy config object, not a module; the module is only created by `instantiate` inside `__init__`. An external caller who tries to pass `torch.nn.Identity()` (an already-instantiated module) will get a confusing error from `instantiate`.

**Fix:** Clarify the type contract:

```
output_norm_cfg: ``LazyConfig`` for the normalisation applied after the second
    gate.  Defaults to a ``LazyConfig`` wrapping ``torch.nn.Identity`` (no
    normalisation).  Do **not** pass an already-instantiated module — pass a
    ``LazyConfig`` object that wraps the class.
```

______________________________________________________________________

## 6. `flop_count` docstring — FLOP count for QK-Norm assumes RMSNorm always

**Text:** `"3·C·S for Q; additional 3·C·S for K"`

**Issue:** The factor of 3 (mean, variance, scale) is correct for RMSNorm / LayerNorm but is silently assumed here. A GroupNorm or a simple scalar multiply would have different counts. The docstring presents this as exact without flagging the assumption.

**Fix:** Add a note:

```
The factor of 3 assumes an RMSNorm-like norm (sum-of-squares + rsqrt +
elementwise scale).  Other norm types will differ; this is an approximation.
```

______________________________________________________________________

## 7. `flop_count` docstring — missing explanation of why `out_ch` appears in the depthwise conv formula

**Text:**

```
2 · (in_ch / groups) · out_ch · S · k_prod
```

**Issue:** For a true depthwise conv `in_ch == out_ch == groups`, so `(in_ch / groups) = 1` and the expression collapses to `2 · out_ch · S · k_prod`. The formula as written is actually the general grouped-conv formula, and it happens to be correct for depthwise but the derivation is unclear. An external reader will wonder why `out_ch` appears when they expect `in_ch // groups = 1`.

**Fix:** Add an inline note:

```
For a pure depthwise conv (groups == in_ch == out_ch) this simplifies to
``2 · out_ch · S · k_prod``; the grouped formula is written here to handle
partially-grouped convolutions (e.g. ``DistributedDepthwiseConvNd``).
```

______________________________________________________________________

## 8. `forward` docstring — CP AllToAll semantics need more precision

**Text:**

```
1. Before short conv: ``split_to_full`` — gather spatial shards, split along the channel dim.
2. After short conv: ``full_to_split`` — scatter spatial, gather channels back.
```

**Issue:** The terms `split_to_full` and `full_to_split` are internal string constants from `AllToAllSingleFunction`. A new collaborator does not know which spatial axis is sharded or what "split along channel" means quantitatively. Is it the *first* spatial axis? What is the shard size?

**Fix:** Add:

```
The AllToAll shards along ``dim=2`` (the first spatial axis) and gathers
along ``dim=1`` (the channel axis).  After the AllToAll, each device holds
the full spatial extent but only ``C / cp_size`` channels.  After the reverse
AllToAll, the original ``C`` channels are restored with ``spatial / cp_size``
positions per device.
```

______________________________________________________________________

## 9. `forward` docstring — `query` variable re-use is confusing and undocumented

**Text:** (no docstring for this; it is inline code)

In the body, `query` is overwritten twice:

```python
query = query * self.gate_nonlinear(key)  # now z, not Q
...
# then passed to global_conv as z
```

The variable is named `query` but after the first gate it represents the *gated intermediate* `z`. This is not mentioned anywhere.

**Fix:** Add a note in the `forward` docstring under a "Implementation note" or "Variable naming" paragraph:

```
Implementation note: ``query`` is overwritten in-place (semantically) after
the first gate to hold the gated intermediate ``z = Q ⊙ σ(K)``.  The original
Q tensor is no longer accessible after that point.  This is intentional to
avoid an extra allocation.
```

______________________________________________________________________

## 10. `extra_repr` — `k_norm` can be `None` but is accessed unconditionally

**Text:**

```python
k_norm_str = self.k_norm.__class__.__name__ if self.k_norm is not None else "None"
```

**Issue:** This is correct code, but the docstring does not mention that `k_norm` may be `None` here (consistent with Issue 4). The docstring for `extra_repr` currently says nothing about when fields are omitted.

**Fix:** Add to the `extra_repr` docstring:

```
When ``self.q_norm`` and ``self.k_norm`` are ``None`` (QK-norm disabled),
the strings ``"q_norm=None"`` and ``"k_norm=None"`` are still included so
the disabled state is explicit in ``repr(module)``.
```

______________________________________________________________________

## 11. Missing usage example anywhere in the file

**Issue:** There is no `Example:` block anywhere in the module or class docstring. An external collaborator cannot quickly see how to wire up a `Hyena` block. The `kernels_nd.py` module even has a `# For test, please run:` note, showing that examples are expected.

**Fix:** Add an `Example:` section to the `Hyena` class docstring showing the minimal construction with `CKConvND`, e.g.:

```python
Example:
    >>> import torch
    >>> from nvsubquadratic.lazy_config import LazyConfig
    >>> from nvsubquadratic.modules.hyena_nd import Hyena
    >>> from nvsubquadratic.modules.ckconv_nd import CKConvND
    >>> # Minimal 2D Hyena block (non-causal, no normalisation)
    >>> hyena = Hyena(
    ...     global_conv_cfg=LazyConfig(CKConvND, hidden_dim=64, data_dim=2, ...),
    ...     short_conv_cfg=LazyConfig(torch.nn.Conv2d, 192, 192, 3, padding=1, groups=192),
    ...     gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU),
    ...     pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity),
    ...     qk_norm_cfg=None,
    ... )
    >>> B, H, W, C = 2, 16, 16, 64
    >>> q = k = v = torch.randn(B, H, W, C)
    >>> y = hyena(q, k, v)  # [2, 16, 16, 64]
```

______________________________________________________________________

## 12. Module docstring — context parallelism section does not clarify which spatial axis is sharded

**Text:** `"shard the spatial dimension across devices"`

**Issue:** For 2D/3D inputs there are multiple spatial axes. Which one? The code shards `dim=2` (the first spatial axis), but this is never stated in the docstring.

**Fix:** Change to:

```
The module shards along ``dim=2`` (the first spatial axis of the
channels-first ``[B, C, *spatial]`` tensor) while gathering the channel dim.
For 2D inputs ``[B, C, H, W]`` this means row-wise sharding (across H).
```
