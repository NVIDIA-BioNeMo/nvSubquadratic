# Reviewer Feedback: `nvsubquadratic/modules/attention.py`

Reviewed after the Phase 1 write pass. Issues are ordered roughly by severity
(correctness hazard → missing information → style).

______________________________________________________________________

## 1. `_unflatten_spatial` 3D rearrange has misleading variable names

**Location:** `_unflatten_spatial`, line `return rearrange(x, "b (h w d) c -> b h w d c", ...)`

**Problem:** The einops pattern uses variable names `h`, `w`, `d` but the
actual mapping — given `spatial_shape = (D, H, W)` — is `h` ↔ depth,
`w` ↔ height, `d` ↔ width.  This is a semantic mismatch that could mislead
anyone reading the code.  The docstring note added in Phase 1 documents the
mismatch but the correct fix is to rename the variables in the rearrange
pattern to `"b (d h w) c -> b d h w c"` with `d=spatial_shape[0], h=spatial_shape[1], w=spatial_shape[2]`, which makes the code self-consistent
with the established `(D, H, W)` convention used everywhere else.

**Fix:** Change the rearrange string and keyword argument names:

```python
return rearrange(
    x,
    "b (d h w) c -> b d h w c",
    d=spatial_shape[0],
    h=spatial_shape[1],
    w=spatial_shape[2],
)
```

and remove the warning note from the docstring since the code will then be
unambiguous.

______________________________________________________________________

## 2. Module-level docstring FLOP formula omits QKV projections

**Location:** Module docstring, "Background" section, FLOP formula:

> `FLOPs ≈ 4 · B · H · L² · d_k`

**Problem:** The formula accounts only for the attention matrix products
(QK^T and attention·V).  It omits the QKV input projections that live in
`QKVSequenceMixer`, which cost `3 · B · L · C²` additional FLOPs.  The
docstring should either (a) state explicitly that these are only the
*attention kernel* FLOPs excluding projections, or (b) give the full formula.

**Fix:** Add a clarifying parenthetical, e.g.:

> "(attention kernel only; QKV input/output projections in
> `QKVSequenceMixer` add another `~6·B·L·C²` FLOPs)"

______________________________________________________________________

## 3. Class docstring "Context parallelism (CP)" section contradicts forward docstring

**Location:** Class docstring, "Context parallelism (CP)" section:

> "When `cp_group` is passed at forward time and has size > 1, the module
> **gathers the full spatial sequence** via zigzag all-gather before attention
> and splits it back afterwards."

**Problem:** The `forward` method immediately raises `ValueError("Context parallelism must be revisited.")` before any all-gather is attempted.  The
class docstring says the gather-and-split *happens*, when in fact it never
does because the `raise` is unconditional.  An external collaborator reading
only the class docstring will believe CP works.

**Fix:** Rewrite the class docstring CP section to match the forward
docstring's wording — state that `cp_group.size() > 1` raises `ValueError`
and that the implementation is a sketch for future compatibility, not
functional code.

______________________________________________________________________

## 4. `_rope_ndim` attribute is undocumented in the `Attributes` block

**Location:** Class docstring, `Attributes` section.

**Problem:** `self._rope_ndim` is set in `__init__` when `use_rope=True` and
used in every branch of the `forward` RoPE block, yet it does not appear in
the `Attributes` block.  A collaborator reading the class docstring to
understand the module's state will miss it.

**Fix:** Add to the Attributes block:

```
_rope_ndim (int | None): Spatial rank for which RoPE was initialised
    (1, 2, or 3). Present only when ``use_rope=True``; not defined
    otherwise.
```

______________________________________________________________________

## 5. `forward` step 1 description is misleading (CP is non-functional)

**Location:** `forward` docstring, pipeline step list:

> "1. (Optional CP) Zigzag all-gather Q/K/V along the spatial axis."

**Problem:** Step 1 implies the all-gather actually executes when `cp_group`
is provided.  In reality, the code raises immediately.  This is inconsistent
and confusing.

**Fix:** Replace step 1 with:

> "1. (Optional CP, not yet implemented) Raises `ValueError` if
> `cp_group.size() > 1`; pass `cp_group=None` for all current use cases."

______________________________________________________________________

## 6. `rope_spatial_dims` not listed as an attribute when `use_rope=True`

**Location:** Class docstring, `Attributes` section; `__init__` docstring.

**Problem:** `rope_spatial_dims` is accepted as an `__init__` argument but
not stored as `self.rope_spatial_dims`, which means it cannot be inspected
after construction.  This is fine for internal use but the `__init__`
docstring says it "must match the spatial shape seen during `forward`" without
explaining how a caller should verify or recover this shape post-construction.
`extra_repr` also does not expose it.

**Fix:** Either store `self.rope_spatial_dims = rope_spatial_dims` and add it
to `Attributes` + `extra_repr`, or add a note in the `__init__` docstring
explaining that `rope_spatial_dims` is intentionally not stored and that the
caller is responsible for tracking it.

______________________________________________________________________

## 7. Missing shape annotation on `_flatten_spatial` return type for batch-head input

**Location:** `_flatten_spatial` docstring, `Args` section:

> "x (torch.Tensor): Input tensor of shape `[B, *spatial_dims, C]` where
> `len(spatial_dims)` is 1, 2, or 3."

**Problem:** At the point `_flatten_spatial` is called in `forward`, `x` has
already been rearranged to `[B*H, *spatial_dims, d_k]` — the leading `B`
dimension is actually `B*H`.  The docstring implies the batch axis is always
`B`, which is misleading to someone reading it in the context of `forward`.

**Fix:** Add a note:

> "Note: in `forward`, this is called after the channel → head split, so the
> leading dimension is `B * H` and `C` is `d_k = head_dim`."

______________________________________________________________________

## 8. No `Example` block anywhere in the class

**Problem:** External collaborators benefit from a minimal usage example
showing how to instantiate `Attention` for a typical 2D image use case.
The sibling module `hyena_nd.py` already has inline code examples.

**Fix:** Add an `Example:` block to the class docstring, e.g.:

```python
# 2D image attention with 8 heads, RoPE, and cosine-attention QK norm
attn = Attention(
    hidden_dim=256,
    num_heads=8,
    apply_qk_norm=True,
    use_rope=True,
    rope_spatial_dims=(32, 32),
)
q = k = v = torch.randn(2, 32, 32, 256)  # [B, H, W, C]
out = attn(q, k, v)  # [B, H, W, C]
```
