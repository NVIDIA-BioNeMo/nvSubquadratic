# Review: `nvsubquadratic/modules/position_encoding.py`

Reviewed as a critical second reader targeting external collaborators reading alongside the paper.

______________________________________________________________________

## Issue 1 — Module docstring mentions "sinusoidal, learned, RFF" variants but only one is implemented

**Location**: Module docstring, "Variants" section.

**Quoted text**:

> "describe the variants present (sinusoidal, learned, RFF, etc.)"

The module docstring (in the prose leading up to the `PositionEmbeddingND` variant block) does not explicitly acknowledge that sinusoidal and RFF encodings are *absent* from the module. The `PositionEmbeddingND` class description says "For variable-resolution or resolution-generalisation use-cases, prefer a sinusoidal or RFF-based encoding (not yet included in this module)." This is good, but the module-level "Variants" heading implies more variants are present. **Fix**: rename the "Variants" section to "Implemented variants" or explicitly add a "Planned / not yet implemented" sub-list (sinusoidal, RFF) so readers are not confused by the heading.

______________________________________________________________________

## Issue 2 — The factorisation parameter-count formula in the module docstring has a misleading simplification

**Location**: Module docstring, "ND generalisation strategy" section.

**Quoted text**:

> "The total parameter count is therefore `data_dim * max(max_dim_lengths) * (embedding_dim // data_dim) = embedding_dim * max(max_dim_lengths)` — linear in the embedding dimension and the largest axis length."

The formula `data_dim * max(max_dim_lengths) * (embedding_dim // data_dim)` is only correct when all axes have the *same* maximum length. The true total is `sum(max_dim_lengths[d] * (embedding_dim // data_dim) for d in range(data_dim))`, which is already stated correctly in the class docstring's "Parameter count" section. The module-level simplification introduces a subtle error for non-square grids (e.g. a 2D 56×14 grid). **Fix**: replace the over-simplified formula with the exact sum, or explicitly state "assuming all axes have the same maximum length `M`".

______________________________________________________________________

## Issue 3 — No mention of the `param._no_weight_decay` convention in the module docstring

**Location**: Module docstring (overview / cross-references).

The module docstring says nothing about the `_no_weight_decay` tagging convention. An external collaborator writing a custom optimiser builder will need to know that the codebase uses `param._no_weight_decay = True` as a signal. The class docstring explains it, but readers skimming the module overview will miss it. **Fix**: add a one-sentence note in the module docstring (e.g. in the `PositionEmbeddingND` variant description or in a new "Optimiser integration" note) that all embedding parameters are tagged with `_no_weight_decay = True` and that optimiser builders should respect this flag.

______________________________________________________________________

## Issue 4 — `forward` docstring does not explain the broadcast-expand mechanism

**Location**: `PositionEmbeddingND.forward` docstring, "Returns" section.

The docstring says the returned tensor has "the same shape as `x`" but does not explain *how* the 1-D per-axis embedding slices are broadcast-expanded to the full `[B, *spatial_dims, per_dim_embedding_dim]` shape before concatenation. A reader inspecting the code will see `emb_axis.view(1, *shape, ...)` followed by `.expand(batch_size, *spatial_dims, ...)` and may not immediately understand why this is correct. **Fix**: add one sentence in the "Returns" section describing the broadcast-expand step: "Each 1-D per-axis embedding of shape `[L_d, per_dim]` is reshaped to `[1, ..., L_d, ..., per_dim]` (with 1 in all axes except `d`) and then broadcast-expanded to `[B, *spatial_dims, per_dim]`; the resulting tensors are concatenated on the last dimension."

______________________________________________________________________

## Issue 5 — The `forward` method validates `x.shape[-1] == embedding_dim` but the docstring's `Args` does not mention this constraint explicitly

**Location**: `PositionEmbeddingND.forward` docstring, "Args" block for `x`.

**Quoted text**:

> "Shape: `[B, *spatial_dims, embedding_dim]`, where the number of spatial axes must equal `data_dim` and the last dimension must equal `embedding_dim`."

The constraint is present but buried in the shape annotation prose. The `Raises` block lists `ValueError` for channel-dimension mismatch, but does not state what the mismatch condition is ("`x.shape[-1] != embedding_dim`"). **Fix**: in the `Raises` entry for the channel mismatch error, state the condition explicitly: "If `x.shape[-1] != self.embedding_dim` (channel count does not match the embedding dimension passed at construction)."

______________________________________________________________________

## Issue 6 — Missing `device` / `dtype` propagation note in `forward`

**Location**: `PositionEmbeddingND.forward` docstring.

The method internally calls `torch.arange(..., device=x.device)` but does not use `x.dtype` for the position indices (correctly using `torch.long`, since they are integer indices). However, the embedding lookup result inherits the dtype of `nn.Embedding.weight`. There is no note clarifying that the output dtype matches the embedding weight dtype (typically `float32`) and that callers working in mixed precision (e.g. `bfloat16`) may need to cast the result. **Fix**: add a `Note` block: "The returned tensor has the same `dtype` as the embedding weight parameters (typically `torch.float32`). In mixed-precision training the result should be cast to match `x.dtype` before addition: `x = x + pos_enc(x).to(x.dtype)`."

______________________________________________________________________

## Issue 7 — Class docstring uses `‖` (U+2016 DOUBLE VERTICAL LINE) for concatenation; this is non-standard and may not render in all docstring viewers

**Location**: `PositionEmbeddingND` class docstring.

**Quoted text**:

> `PE(i_0, ..., i_{D-1}) = [E_0(i_0) ‖ E_1(i_1) ‖ ... ‖ E_{D-1}(i_{D-1})]`

The `‖` character is ambiguous (it is also the norm operator). Standard ML papers use `concat` or `[· ; ·]`. **Fix**: replace `‖` with `concat(...)` or add a brief note "(where `‖` denotes channel-wise concatenation)" immediately after the formula.

______________________________________________________________________

## Issue 8 — `__init__` docstring lists `AssertionError` in `Raises` but the checks could be `ValueError` for better API ergonomics

**Location**: `PositionEmbeddingND.__init__` docstring, `Raises` block.

**Quoted text**:

> "Raises: AssertionError: If `data_dim < 1`. AssertionError: If ..."

The implementation uses `assert` statements, so technically the docstring is correct. However, documenting `AssertionError` as a public API contract is poor practice: `assert` statements are disabled under `python -O`, making the checks silently disappear in optimised deployments. The `Raises` documentation signals to callers that these are expected error conditions, not internal invariants. **Fix**: convert the four `assert` statements in `__init__` to `ValueError` raises (as is already done in `forward`) and update the `Raises` block accordingly.

______________________________________________________________________
