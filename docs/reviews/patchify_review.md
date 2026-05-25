# Review: `nvsubquadratic/modules/patchify.py`

Reviewer pass on commit `docs(write/patchify): add module and class docstrings`.

______________________________________________________________________

## Issues

### 1. Module docstring — overlapping-patch semantics need a concrete example

**Location:** Module docstring, "Overlapping patches" paragraph in `Patchify`.

**Problem:** The docstring says "setting `stride < patch_size` produces overlapping patches
with the same formula above" but does not show the formula for that case.  An external
collaborator who only reads the module docstring (not the class docstring) sees no formula at all.

**Fix:** Add the general output-size formula `floor((s - patch_size) / stride) + 1` to
the module docstring overview, directly after the description of the output layout convention.

______________________________________________________________________

### 2. `Patchify.__init__` — `patch_size` docstring does not mention the per-patch pixel count

**Location:** `Patchify.__init__` docstring, `patch_size` arg entry.

**Quote:** `"Side length P of each patch.  The convolution uses kernel_size = patch_size along every spatial axis, so each patch covers P^data_dim input pixels."`

**Problem:** The phrase "each patch covers P^data_dim input pixels" uses caret notation that
renders as plain text in most docstring viewers.  Readers unfamiliar with the notation may
think `^` is a bitwise XOR.

**Fix:** Replace `P^data_dim` with `P**data_dim` (Python exponentiation) or spell it out
as "`P × P` pixels (2D)" / "`P × P × P` voxels (3D)" with a note that the general
formula is `P ** data_dim`.

______________________________________________________________________

### 3. `Patchify` class docstring — no mention of what happens when spatial size is not divisible by `patch_size`

**Location:** `Patchify` class docstring, immediately after the output shape formula.

**Problem:** The formula `s / patch_size` (non-overlapping case) implicitly assumes exact
divisibility, but the layer silently truncates if `s % patch_size != 0` (standard Conv
floor semantics).  A new collaborator who passes a 224×224 image with `patch_size=16`
gets 14×14 tokens as expected, but a 225×224 image would give 14×14 tokens silently
dropping the last pixel column.  This is a frequent source of bugs.

**Fix:** Add a warning note under the output shape formula, e.g.:

> **Note:** If `spatial_dim % patch_size != 0`, the last pixels in that axis are silently
> discarded (standard floor-division Conv semantics).  Callers are responsible for ensuring
> the spatial dimensions are divisible by `patch_size` (e.g. via padding before this layer).

______________________________________________________________________

### 4. `Unpatchify.forward` — `output_spatial_shape` description buries the most important use-case

**Location:** `Unpatchify.forward` docstring, `output_spatial_shape` arg description.

**Quote:** `"Must have length data_dim.  When None, PyTorch infers the output size (may differ from the original input spatial size if spatial_dim % patch_size != 0)."`

**Problem:** The caveat about size ambiguity is in the middle of a long sentence.  It is the
most important reason to pass `output_spatial_shape`, but a skimming reader will miss it.

**Fix:** Restructure the description to lead with the ambiguity problem:

> When `stride > 1` multiple patch-grid sizes map to the same output size (floor in the
> forward direction discards remainders).  Pass `output_spatial_shape` to resolve this
> ambiguity and guarantee recovery of the exact original spatial dimensions.  Must have
> length `data_dim`.  When `None`, the output size is inferred by PyTorch and may not
> match the original spatial size if `spatial_dim % patch_size != 0`.

______________________________________________________________________

### 5. `Unpatchify` class docstring — overlapping-patch adjoint semantics are ambiguous

**Location:** `Unpatchify` class docstring, second paragraph.

**Quote:** `"When stride < patch_size (overlapping), contributions from overlapping patches are *summed* by the transposed convolution — this is the adjoint of the overlapping-patch forward pass."`

**Problem:** The summation is correct mathematically, but "adjoint" may mislead readers into
thinking `Unpatchify` perfectly reconstructs the original signal.  For overlapping patches
the transposed conv is only the linear adjoint (backward map), not an inverse; pixel values
are *accumulated*, not averaged, so the output scale grows with the overlap ratio.  This is
especially unexpected for users who pair `Patchify` and `Unpatchify` in an autoencoder.

**Fix:** Add an explicit note:

> For overlapping patches this accumulation means `Unpatchify(Patchify(x))` does **not**
> recover `x` exactly; the output is a blurred, scaled version of `x`.  Only for
> non-overlapping patches (`stride == patch_size`) does the round-trip preserve spatial
> alignment (up to the learned weights).

______________________________________________________________________

### 6. `Unpatchify.__init__` — `weight_init` tradeoffs are missing for the `"default"` option

**Location:** `Unpatchify.__init__` docstring, `weight_init` arg, `"default"` option.

**Quote:** `'"default" uses PyTorch's built-in kaiming_uniform (fan computed from out_features; can cause output-variance blow-up for large in_features).'`

**Problem:** The existing text warns about variance blow-up for `"default"` but does not tell
the reader *when* to choose it anyway.  An external collaborator who has not read the
DiT/ViT literature will default to `"default"` without knowing the risk.

**Fix:** Add a recommendation, e.g.:

> Prefer `"fan_in"` for new architectures.  `"default"` is retained for loading
> pre-trained checkpoints whose weights were saved under PyTorch's default init.

______________________________________________________________________

### 7. Missing cross-reference to `PositionEmbeddingND` in module docstring

**Location:** Module docstring.

**Problem:** The module docstring mentions that the output layout matches what
"PositionEmbeddingND and the mixer blocks" expect, but does not tell the reader
where to find `PositionEmbeddingND`.

**Fix:** Add an explicit cross-reference:

> See `nvsubquadratic.modules.position_encoding.PositionEmbeddingND` for the positional
> encoding layer that is typically applied immediately after `Patchify`.

______________________________________________________________________

### 8. `Patchify` and `Unpatchify` class docstrings — no ND usage example

**Location:** Both class docstrings.

**Problem:** The class-level docstrings show only implied 2D usage (the shapes use `H, W`
notation).  A reader working on 1D sequences or 3D volumes must infer the generalisation
from the `data_dim` parameter alone, with no concrete example.

**Fix:** Add a brief "Examples" section to at least one of the classes showing a 1D case
(`[B, L, C]` → `[B, L/P, C_embed]`) and referencing the `__main__` block for the 2D
demonstration.
