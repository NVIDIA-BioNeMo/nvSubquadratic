# Review: `nvsubquadratic/ops/mixed_fftconv.py`

Reviewed as a new external collaborator reading alongside the research paper.
All item numbers below are concrete fixes to apply.

______________________________________________________________________

## Module docstring

**1. The "same" alignment example in the module-level "Algorithm overview" section
uses the tuple form `(True, False)` in the code snippet but the call signature
accepts `Sequence[bool]`, not a plain tuple label.**

Quote: `y = mixed_fftconv2d_fp32_bhl(x, kernel, periodic=(True, False))`

No issue with the Python itself, but the module docstring does not mention that
`CKConvND` uses the *string* form (`fft_padding=["circular", "zero"]`) and that
the `periodic` bool-tuple is the *internal* normalised form. A reader glancing at
this module and then looking at a config YAML will be confused about why their
config uses strings but the function takes bools.

**Fix:** Add a one-sentence note at the end of the module-level "When to use this"
section, e.g.:

> Note: `CKConvND` accepts `fft_padding=["circular", "zero"]` (string form).
> It internally converts this to the boolean `periodic` tuple before calling
> these ops.

______________________________________________________________________

## `_mixed_recipe`

**2. The docstring says the crop for a non-periodic axis is
`[K_d // 2, K_d // 2 + N_d)`, but the code uses integer floor division. For an
even kernel `K_d = 4`, the crop starts at index 2 — which is the "left-aligned"
half — but the docstring never explains the asymmetry for even kernels.**

Quote from docstring:

> crop `[K_d // 2, K_d // 2 + N_d)`

The crop is correct for odd kernels (symmetric). For even kernels it applies the
"left half of even kernel" convention (matching `torch.nn.Conv*d(padding='same')`
with `dilation=1`), which is a non-obvious choice. If a collaborator tries to
reproduce results with manual `F.pad` they will get the wrong result unless they
know this.

**Fix:** Add a note after the crop description:

> For even `K_d`, floor division gives a left-biased crop (one extra sample on
> the right), matching the convention of `torch.nn.ConvNd(padding='same')`.
> Tests verify this against the spatial reference.

______________________________________________________________________

**3. The function signature has `kshape` as a parameter name but the docstring
calls it "Kernel spatial dims". The abbreviation `kshape` is fine, but the
parameter name in the Args block should match the actual argument name exactly.**

Quote from docstring:

> `kshape:  Kernel spatial dims `(K_0, ..., K\_\{D-1})`.`

This is correct — just flag that the double space before "Kernel" is a minor
formatting inconsistency that ruff/mdformat may flag in future.

**Fix:** Remove the extra space: `kshape: Kernel spatial dims ...`

______________________________________________________________________

## `_MixedPhaseRamp1DCache`

**4. The class docstring says "Non-periodic axes use `s_d = 0` and contribute
no ramp." but it does not explain *why* `s_d = 0` for non-periodic axes. A reader
unfamiliar with the algorithm will not know that non-periodic alignment is handled
by the centered crop instead.**

Quote:

> Non-periodic axes use `s_d = 0` and contribute no ramp.

**Fix:** Append the explanation:

> Non-periodic axes use `s_d = 0` because their "same" alignment is handled
> entirely by the centered crop `[K_d // 2, K_d // 2 + N_d)` in the IFFT
> output — no frequency-domain shift is needed.

______________________________________________________________________

## `_MixedPhaseRamp1DCache.get`

**5. The Returns line says "1-D complex tensor of shape `[F // 2 + 1]` (rfft
axis) or `[F]` (non-rfft axis)". But when `s == 0` the method still builds
and returns a tensor of all-ones (as the docstring says "returns all-ones for
completeness"). This is never actually reached in practice because
`_build_nd_phase_ramp` skips `s == 0` axes — so the "all-ones" note implies a
code path that never runs. This could mislead a reader into thinking `s == 0`
is a valid call site.**

Quote:

> Callers with `s == 0` are expected to skip the multiply entirely; this
> method still handles `s == 0` (returns all-ones).

**Fix:** Remove the parenthetical "returns all-ones" claim (or move it to a
comment in the code), and instead say:

> In practice `_build_nd_phase_ramp` skips axes where `s == 0` before
> calling this method.

______________________________________________________________________

## `_build_nd_phase_ramp`

**6. The docstring says "The N-D ramp is not cached; only the 1-D per-axis ramps
are." but does not mention that the returned tensor is a *view* of the 1-D cached
tensors (broadcast, not materialised as a separate allocation). A reader
concerned about memory may not realise that the N-D tensor is cheap — it is a
view chain, not a full materialisation.**

Quote:

> The N-D ramp is **not** cached; only the 1-D per-axis ramps are. The
> product is materialised here (via broadcasted multiplication) so that the
> downstream multiply with `fft_x` is a single fused op.

The word "materialised" is ambiguous — it sounds like the whole N-D tensor is
allocated, but the product loop in the code multiplies the broadcast views,
producing a new (small, single-axis-varying) tensor each time.

**Fix:** Clarify:

> The N-D ramp itself is not cached. The loop multiplies broadcast 1-D views
> together, producing a compact tensor (not a full `(F_0, ..., F_{D-1})`
> allocation) that covers only the periodic axes with non-zero shifts.

______________________________________________________________________

## `_mixed_fftconv_nd_fp32_bhl`

**7. Step 6 in the docstring says "apply `torch.roll` on the periodic axes"
but the code actually filters to only the axes where `s != 0` (i.e., it
excludes size-1 periodic kernels):**

Quote from code:

```python
roll_shifts = tuple(s for s in shifts if s != 0)
roll_dims = tuple(2 + d for d, s in enumerate(shifts) if s != 0)
```

The docstring says "apply torch.roll on the periodic axes" which is slightly
inaccurate — it applies to periodic axes *with non-zero shifts*.

**Fix:** Change step 6 to:

> 6. If `use_phase_shift=False`, apply `torch.roll` on periodic axes that
>    have a non-zero shift (`s_d != 0`); size-1 periodic kernels (shift 0)
>    are skipped.

______________________________________________________________________

**8. The kernel-size limit comments in the code (explaining why periodic axes
use `K <= N` and non-periodic use `K <= 2*N`) are inside `_mixed_fftconv_nd_fp32_bhl`
as inline comments but are not surfaced in the docstring's Raises section.
An external collaborator who passes an oversized kernel will get an
`AssertionError` with a message but no hint about what "double-grid" means.**

Quote from Raises:

> AssertionError: On shape mismatches or out-of-range kernel sizes.

**Fix:** Expand the Raises block:

> AssertionError: On shape mismatches or out-of-range kernel sizes.
> Specifically: `K_d > N_d` on a periodic axis (circular FFT length is
> `N_d`, so the kernel must fit); `K_d > 2*N_d` on a non-periodic axis
> (the padded FFT length is at most `2*N_d`, matching the standard
> "double-grid" SIREN kernel size `2*N_d - 1`).

______________________________________________________________________

## Public 1D/2D/3D BHL entry points

**9. `mixed_fftconv1d_fp32_bhl` and its 2D/3D siblings each reference
`use_phase_shift` but the 2D and 3D variants say "See
:func:`mixed_fftconv1d_fp32_bhl`" for its meaning. That cross-reference works,
but the reader must jump to the 1D function to understand the parameter — and
the 1D function's `use_phase_shift` description does not mention the performance
trade-off explicitly.**

Quote from `mixed_fftconv1d_fp32_bhl`:

> use_phase_shift: If True, align periodic axes via frequency-domain phase
> ramps. If False, align via :func:`torch.roll` on periodic axes after
> the inverse transform. The output is mathematically equivalent.

**Fix:** Add a performance hint to the 1D description:

> `use_phase_shift=True` (default) is faster — the shift is fused into the
> frequency-domain multiply with no additional data movement. Use `False`
> only as a reference or when `torch.compile` cannot handle complex ops.

______________________________________________________________________

## BLH `_w_reshape` wrappers

**10. The kernel argument docstring for `mixed_fftconv1d_fp32_bhl_w_reshape`
says ``` kernel: Kernel tensor of shape ``[B, K, H]`` (BLH) ```. But the BHL
wrappers also support a shared-kernel leading dim of 1 (`[1, K, H]`), which
is the standard depthwise case. The docstring is inconsistent with the BHL
entry points (which say `[1|B, H, K]`) and with the module-level shape
conventions.**

Quote:

> kernel: Kernel tensor of shape `[B, K, H]` (BLH).

**Fix:** Change to `[1|B, K, H]` in all `_w_reshape` wrapper docstrings (1D,
2D, 3D, and their chunked counterparts).

______________________________________________________________________

## Chunked variants

**11. `_DEFAULT_MIXED_CHUNK_SIZE = 128` is a module-level constant referenced in
the chunked function docstrings but has no docstring or comment of its own. A
reader tuning memory usage will not know why 128 was chosen.**

The related `fftconv_chunked.py` module mentions "~26% memory savings for ~11%
overhead" for chunk size 128 — that context is missing here.

**Fix:** Add a comment above the constant:

```python
# Default channel chunk size for the memory-efficient variants.
# A chunk of 128 channels typically gives ~26% peak-memory savings
# with ~11% throughput overhead relative to the non-chunked path
# (measured on H100; see fftconv_chunked.py for profiling details).
_DEFAULT_MIXED_CHUNK_SIZE = 128
```

______________________________________________________________________

## Missing usage example

**12. None of the public entry points include a minimal runnable example. The
module docstring has a snippet but it shows only the function call, not the
tensor shapes, so a new user cannot copy-paste-and-run it. A short `Example:`
block on `mixed_fftconv2d_fp32_bhl` (the most common use case) would help
significantly.**

**Fix:** Add to `mixed_fftconv2d_fp32_bhl`:

```python
Example:
    >>> import torch
    >>> from nvsubquadratic.ops.mixed_fftconv import mixed_fftconv2d_fp32_bhl
    >>> B, H, X, Y, Kx, Ky = 2, 64, 32, 64, 63, 127
    >>> x = torch.randn(B, H, X, Y)
    >>> kernel = torch.randn(1, H, Kx, Ky)
    >>> # x periodic, y zero-padded
    >>> y = mixed_fftconv2d_fp32_bhl(x, kernel, periodic=(True, False))
    >>> y.shape
    torch.Size([2, 64, 32, 64])
```

______________________________________________________________________

## Summary of fixes to apply

| #   | Location                     | Fix type                                     |
| --- | ---------------------------- | -------------------------------------------- |
| 1   | Module docstring             | Add note about string form in CKConvND       |
| 2   | `_mixed_recipe`              | Clarify even-kernel asymmetry                |
| 3   | `_mixed_recipe`              | Remove double space in Args                  |
| 4   | `_MixedPhaseRamp1DCache`     | Explain why `s_d = 0` for non-periodic       |
| 5   | `_MixedPhaseRamp1DCache.get` | Remove misleading "returns all-ones" claim   |
| 6   | `_build_nd_phase_ramp`       | Clarify "materialised" language              |
| 7   | `_mixed_fftconv_nd_fp32_bhl` | Fix step 6 (roll only on `s != 0` axes)      |
| 8   | `_mixed_fftconv_nd_fp32_bhl` | Expand Raises block with kernel-size context |
| 9   | `mixed_fftconv1d_fp32_bhl`   | Add performance hint to `use_phase_shift`    |
| 10  | All `_w_reshape` wrappers    | Fix kernel shape to `[1\|B, K, H]`           |
| 11  | `_DEFAULT_MIXED_CHUNK_SIZE`  | Add comment with memory-savings context      |
| 12  | `mixed_fftconv2d_fp32_bhl`   | Add runnable `Example:` block                |
