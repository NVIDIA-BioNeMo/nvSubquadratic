# Reviewer feedback: `nvsubquadratic/modules/kernels_nd.py`

Second-pass review after Phase 1 docstring additions. Issues are numbered; each
quotes the offending text and says exactly what to fix.

______________________________________________________________________

## 1. Module docstring: `RandomFourierKernelND` is not listed in the "Kernel classes" table

**Location**: module-level docstring, "Kernel classes in this module" section.

The table lists six consumer-facing kernel classes but omits
`RandomFourierKernelND` — the table starts with `SIRENKernelND`. The class is
described in the prose above, but a new reader scanning the table will not find
it.

**Fix**: add `RandomFourierKernelND` as the first entry in the list, before
`SIRENKernelND`.

______________________________________________________________________

## 2. `_normalize_l_cache`: missing `Raises` block

**Location**: `_normalize_l_cache` docstring.

The function raises `TypeError` (for bool or non-int/sequence input) and
`ValueError` (for wrong sequence length or values \< 2), but the docstring has
no `Raises:` section.  An external collaborator reading this to decide whether
to guard against exceptions will not know what to catch.

**Fix**: add a `Raises:` section listing both `TypeError` and `ValueError`
with the conditions that trigger each.

______________________________________________________________________

## 3. `RandomFourierPositionalEmbeddingND.forward`: wrong shape annotation for grid return value

**Location**: `forward` docstring, `Returns` block:

> `torch.Tensor: The input positions normalized between [-1, 1] (shape: [1, * spatial_dims, 1]).`

The trailing dimension is `data_dim` (the number of spatial axes), not `1`.
A 2D model has a grid of shape `[1, H, W, 2]`, not `[1, H, W, 1]`.  The
`_build_grid_cache` code confirms: `dim=-1` of the stacked meshgrid has size
`data_dim`.

**Fix**: change the shape annotation to `[1, *spatial_dims, data_dim]`.

The same bug appears in `SIRENPositionalEmbeddingND.forward` — fix it there
too.

______________________________________________________________________

## 4. `RandomFourierPositionalEmbeddingND.forward` and `SIRENPositionalEmbeddingND.forward`: stale `"concatenated sine and cosine values"` description for SIREN

**Location**: `SIRENPositionalEmbeddingND.forward`, `Returns` block:

> `torch.Tensor: The positional embeddings, concatenated sine and cosine values (shape: [1, * spatial_dims, embedding_dim]).`

The SIREN embedding is `sin(Wx+b)` — a single sine, no cosine concatenation.
The description was copied verbatim from the RFF embedding, where `[cos, sin]`
concatenation is correct.

**Fix**: change the SIREN `forward` return description to:
"The positional embeddings, `sin(W x + b)`, shape `[1, *spatial_dims, embedding_dim]`."

______________________________________________________________________

## 5. `RandomFourierKernelND.forward`: `Returns` annotation says "tuple" but the function actually returns two tensors described as a single `tuple[torch.Tensor, torch.Tensor]`

**Location**: `RandomFourierKernelND.forward`, `Returns` block:

> `tuple[torch.Tensor, torch.Tensor]: The computed random Fourier kernel and the corresponding grid values.     The kernel is a tensor of shape (1, * spatial_dims, out_dim)     The grid is a tensor of shape (1, * spatial_dims, data_dim)`

The format mixes the inline `tuple[…]` type-hint style with a multi-line
description, making it hard to parse. The return type annotation in the
`Returns:` block should follow Google style: one sub-item per returned object.

**Fix**: rewrite as:

```
Returns:
    tuple:
        - torch.Tensor: Kernel values of shape ``[1, *spatial_dims, out_dim]``.
        - torch.Tensor: Coordinate grid of shape ``[1, *spatial_dims, data_dim]``.
```

______________________________________________________________________

## 6. `SIRENKernelND` class docstring: `flop_count` method has no standalone docstring header describing its purpose in one sentence

**Location**: `SIRENKernelND.flop_count`.

The docstring jumps directly into the "At `inference=True`…" explanation
without a one-sentence summary. Google style requires a short summary line
first.

**Fix**: add as the first line:

> "Return an integer FLOP estimate for one kernel generation forward pass."

______________________________________________________________________

## 7. `SIRENKernelND.flop_count`: `grid_lens` description is ambiguous

**Location**: `flop_count` docstring, `Args:` block:

> ``` grid_lens: Spatial extents passed to the positional embedding. The kernel grid has size ``(2 * L - 1)`` per dimension. ```

This does not say what unit `grid_lens` is in (seq_lens? cache extents?).  The
code uses `for L in grid_lens: G *= 2 * L - 1`, so `L` here is the *output*
sequence length (each axis), not the cache extent.  The description "Spatial
extents passed to the positional embedding" is correct but the parenthetical
`(2 * L - 1)` makes it look like `grid_lens` are the half-lengths.

**Fix**: replace with:

> ``` grid_lens: Per-axis output sequence lengths, i.e. the same tuple you would pass to ``forward`` as ``seq_lens``. The FLOP count uses the grid volume ``G = prod(2*L - 1 for L in grid_lens)`` as the number of coordinate points the MLP processes. ```

______________________________________________________________________

## 8. `_build_grid_cache`: missing Args/Returns in both `RandomFourierPositionalEmbeddingND` and `SIRENPositionalEmbeddingND` copies

**Location**: `RandomFourierPositionalEmbeddingND._build_grid_cache` and
`SIRENPositionalEmbeddingND._build_grid_cache`.

Both static methods have a one-paragraph description but no `Args:` or
`Returns:` sections.  The parameters `L_per_axis`, `max_limits`, and `device`
are not described, and the return shape is not stated.

**Fix**: add:

```
Args:
    L_per_axis: Per-axis cache extents.  The cache size along axis ``i``
        is ``2 * L_per_axis[i] - 1`` grid points.
    max_limits: Per-axis coordinate limits.  Each axis spans
        ``[-max_limits[i], +max_limits[i]]``.  Defaults to ``1.0`` on all
        axes.
    device: Target device for the returned tensor.  Defaults to CPU.

Returns:
    Float32 tensor of shape ``[1, 2*L_0-1, ..., 2*L_{d-1}-1, data_dim]``
    representing the coordinate meshgrid, with a leading batch dimension of 1.
```

(The two copies are byte-identical; apply the same fix to both.)

______________________________________________________________________

## 9. `_maybe_extend_grid_cache`: missing Args/Returns in both copies

**Location**: `RandomFourierPositionalEmbeddingND._maybe_extend_grid_cache` and
`SIRENPositionalEmbeddingND._maybe_extend_grid_cache`.

Same issue as item 8: no `Args:` or `Returns:` sections.

**Fix**: add:

```
Args:
    seq_lens: Requested per-axis output sequence lengths.  Any axis where
        ``seq_lens[i] > self.L_cache_per_axis[i]`` triggers a cache
        extension.

Returns:
    None.  Modifies ``self.grid_cache`` and ``self.L_cache_per_axis``
    in-place when an extension is needed.
```

______________________________________________________________________

## 10. `LearnableOmegaSIRENPositionalEmbeddingND._clamp_omega_scale_pre_hook`: missing Args/Returns

**Location**: `_clamp_omega_scale_pre_hook`.

The hook is registered as a forward pre-hook and takes `(module, inputs)` but
the docstring has no `Args:` block, which is required for any public/protected
method.

**Fix**: add:

```
Args:
    module: The module instance (``self``); provided by PyTorch's hook
        mechanism.
    inputs: The positional inputs tuple; not used.

Returns:
    None.  Modifies ``self.omega_0_scale.data`` in-place.
```

______________________________________________________________________

## 11. `BlockDiagonalMultiOmegaSIRENKernelND._block_mask`: `out_dim` / `in_dim` shape contract not stated precisely enough

**Location**: `_block_mask` docstring:

> "`out_dim` and `in_dim` must both be divisible by `num_blocks`."

This constraint is documented but neither a `Raises:` nor a note says what
happens if violated (currently it silently produces wrong-sized blocks because
integer division truncates).

**Fix**: add a `Raises:` section:

```
Raises:
    ZeroDivisionError: If ``num_blocks == 0``.
    Note: if ``out_dim`` or ``in_dim`` is not divisible by ``num_blocks``,
        the block sizes are silently truncated via integer division; the
        caller (``BlockDiagonalMultiOmegaSIRENKernelND.__init__``) enforces
        divisibility before calling this method.
```

and add `Args:` / `Returns:` sections (currently absent):

```
Args:
    out_dim: Number of output features (rows of the weight matrix).
    in_dim: Number of input features (columns of the weight matrix).
    num_blocks: Number of blocks.
    off_block_scale: Scalar value for off-diagonal block entries.
    device: Target device.
    dtype: Target dtype.

Returns:
    Float tensor of shape ``[out_dim, in_dim]`` with ``1.0`` on the
    block diagonal and ``off_block_scale`` elsewhere.
```

______________________________________________________________________

## 12. `SIRENKernelND` class docstring: "Wang initialisation" is unexplained

**Location**: `SIRENKernelND` class docstring, "Initialisation" section:

> "The output layer applies **Wang initialisation**: weights are scaled by `sqrt(1 / kernel_volume)`…"

An external collaborator unfamiliar with this term will not know where it comes
from.  The SIREN paper (Sitzmann et al. 2020) does not use the name "Wang init"
— this is a practice from continuous convolution literature.

**Fix**: add a brief parenthetical reference, e.g.:

> "(introduced for implicit kernel networks by \[Wang et al.\]; equivalent to dividing the output layer's weights by the square root of the total number of kernel grid points, so that the initial filter energy is independent of grid size)"

or cite the CKConv paper where this practice originates.

______________________________________________________________________

## 13. `LearnableOmegaSIRENPositionalEmbeddingND` class docstring: `apply_lr_scale` mentions `_build_param_groups` without context

**Location**: class docstring, `Args:` for `apply_lr_scale`:

> "attach `_lr_scale = 1/(2π·omega_0)` to `self.linear.weight` so that `_build_param_groups` lowers its learning rate…"

`_build_param_groups` is an internal optimizer helper not visible from this
file.  A reader who does not know the codebase's optimizer infrastructure will
not understand how `_lr_scale` is consumed.

**Fix**: replace with:

> "When True, attach `_lr_scale = 1/(2*pi*omega_0)` to `self.linear.weight`. The optimizer utility `_build_param_groups` (in `experiments/`) reads this attribute and multiplies the layer's learning rate by `_lr_scale`, compensating for the missing `2*pi*omega_0` factor in the SIREN-1 init bound."

______________________________________________________________________

## 14. Module docstring: no mention of where the `conditioning` tensor comes from

**Location**: module docstring, no mention of FiLM conditioning flow.

The module docstring describes the two kernel families (RFF and SIREN) but
does not mention that `SIRENKernelND` and its descendants support a
`conditioning` argument that makes the kernel input-dependent.  This is the
FiLM path, and a new collaborator reading the module overview has no idea this
feature exists or how to trigger it.

**Fix**: add a short paragraph in the Overview section:

> "Optionally, `SIRENKernelND` (and all SIREN subclasses) accept a `conditioning` tensor of shape `[B, C]` via their `forward` method. When a `KernelFiLMGenerator` is wired in via `film_cfg`, the kernel becomes input-dependent — each sample in the batch gets a different filter. This is used in diffusion models and other conditional generation tasks where the kernel must respond to a global context signal."

______________________________________________________________________
