# Reviewer feedback: `nvsubquadratic/modules/ckconv_nd.py`

Second-reader pass against the Phase 1 docstrings. Issues are listed in
reading order (module docstring → module-level functions → class).

______________________________________________________________________

## Module docstring

**Issue 1 — `References` heading uses a non-Google-style RST adornment.**

The heading reads:

```
References:
----------
```

The colon is inside the heading text, which is non-standard Google style
(Google style omits the colon on section names like `Args:`, `Returns:`,
etc., but those are *field* sections, not free-form RST sections).  More
critically, the separator `----------` has one more dash than the heading
text (`References:`), which breaks RST rendering.  Fix: use a plain
`References` heading with the correct separator length, or follow the
pattern already used by `kernels_nd.py` and write the references as a
bullet list inside the `Related modules` section without a separate heading.

______________________________________________________________________

## `_parse_padding_mode`

**Issue 2 — Missing explanation of what the return value (`True`/`False`)
means in context.**

The Returns block says:

> `True` if the mode is `"circular"` (periodic axis), `False` if it is `"zero"` (zero-padded axis).

This is correct, but a new reader has no idea why a boolean is returned
here or how it maps to the `periodic` flag used downstream.  Add one
sentence linking the return value to `_PADDING_MODE_TO_PERIODIC` and
clarifying that `True` means "this axis uses circular (wrap-around) FFT
convolution."

______________________________________________________________________

## `_resolve_periodic`

**Issue 3 — `data_dim` arg doc does not mention OmegaConf / `ListConfig`
subtlety.**

The `fft_padding` arg doc mentions that the list form works with
OmegaConf's `ListConfig`, but the *reason* this matters (why the
`isinstance(fft_padding, Sequence)` check is used instead of
`isinstance(fft_padding, list)`) is documented only as a comment in
`__init__`, not in `_resolve_periodic`.  A reader calling `_resolve_periodic`
directly cannot see that context.  Add a note to the `fft_padding` arg
docstring in `_resolve_periodic` that says: "OmegaConf `ListConfig`
objects are accepted because `Sequence` matching is used rather than
`isinstance(…, list)`."

**Issue 4 — Return type annotation is missing.**

The function signature declares `-> tuple[bool, ...]` but the Returns
block only says "a tuple of `data_dim` booleans."  This is fine, but
the docstring should explicitly state the meaning of each boolean
(`True` = circular, `False` = zero-padded) to be self-contained — a
reader consulting just the docstring without seeing `_parse_padding_mode`
would not know this.

______________________________________________________________________

## `_wrap_mixed_op`

**Issue 5 — The inner `_wrapped` closure has no docstring.**

`_wrap_mixed_op` is a factory that returns a callable.  The returned
`_wrapped` function has no docstring of its own, so `help(_wrapped)`
(and any IDE inspection) shows nothing.  Add a one-line docstring to
`_wrapped`: "Call `op_fn(x, kernel, periodic, shortcut)` with bound `periodic`."

______________________________________________________________________

## `_grid_is_single_per_axis`

**Issue 6 — The correspondence between `grid_type="single"` / `"double"`
and the actual kernel spatial sizes is not stated precisely.**

The current docstring says `grid_type='single'` means "the SIREN kernel
grid spans `(N+1)//2` per axis so the produced kernel size equals the
input size on that axis," but it does not explain *why* `(N+1)//2` grid
points produce a kernel of size `N`.  The reason is that `SIRENKernelND`
evaluates on a `(2*L - 1)`-point grid, so `L = (N+1)//2` → kernel size
= `2*(N+1)//2 - 1 ≈ N`.  Without this the paragraph is confusing —
"grid spans `(N+1)//2` … so kernel size equals input size" sounds like
a contradiction.  Add the derivation: `kernel_size = 2*L - 1 = 2*(N+1)//2 - 1`.

______________________________________________________________________

## `CKConvND` — class docstring

**Issue 7 — The `shortcut` initialisation scheme is not documented.**

The class Attributes block lists `shortcut (nn.Parameter)` but gives no
information about how it is initialised.  Looking at the code:

```python
bounds = math.sqrt(1.0 / hidden_dim)
self.shortcut.data.uniform_(-bounds, bounds)
```

This is a Kaiming-uniform-style init (scaled by `1/sqrt(hidden_dim)`).
The Attributes entry should mention this: "Initialised with
`uniform(-1/√C, 1/√C)` (Kaiming-uniform scale)."

**Issue 8 — The `Attributes` block for `fftconv_fn` and
`fftconv_fn_bhl_input` does not document the call signature.**

Both are described as "(callable): Selected FFT convolution function for …"
without stating what arguments they accept.  A reader setting up a
subclass or unit test does not know the signature.  Add: "Signature:
`(x, kernel, shortcut) → output`."

______________________________________________________________________

## `CKConvND.__init__`

**Issue 9 — `kernel_cfg` arg doc does not mention the `out_dim`
constraint.**

`kernel_cfg` is described as "typically a `SIRENKernelND` or
`RandomFourierKernelND` lazy config."  But the kernel must also have
`out_dim = hidden_dim` (its output channel count must match the
operator's channel count).  This constraint is implicit in the code
(the convolution would silently produce wrong-shape tensors otherwise).
Add: "The kernel's `out_dim` must equal `hidden_dim`."

**Issue 10 — `fft_backend="subq_ops"` constraint about per-sample
(FiLM) weights is mentioned in the original `__init__` docstring but
removed from the Phase 1 rewrite.**

The original `__init__` docstring contained the sentence: "Per-sample
(FiLM) weights are supported on the 2D path only; the 1D causal CUDA
kernel does not accept batched weights."  This was dropped in Phase 1.
Reinstate it under the `fft_backend` arg description for `"subq_ops"`.

______________________________________________________________________

## `CKConvND.apply_convolution`

**Issue 11 — `conv_kernel` shape annotation says `(1_or_B, *kernel_spatial, C)` but does not explain what `kernel_spatial` is relative to `spatial`.**

For the zero-padded (double-grid) case `kernel_spatial ≈ 2 × spatial`
(the SIREN evaluates on `2*L - 1` points per axis); for the circular
(single-grid) case `kernel_spatial ≈ spatial`.  A new reader inspecting
a live tensor would be confused by a kernel that is twice as wide as the
input.  Add a one-sentence note: "`kernel_spatial` equals `spatial`
on single-grid (circular) axes and `2*N - 1` on double-grid (zero-padded)
axes."

______________________________________________________________________

## `CKConvND.forward`

**Issue 12 — Tensor shape annotation uses `C` but the class elsewhere
uses `hidden_dim`.**

The Args block writes `C = self.hidden_dim` in the description but the
shape annotations say `(B, *spatial, C)`.  For consistency with the
class Attributes (which use `hidden_dim`) the shape annotation should
use `hidden_dim`: `(B, *spatial, hidden_dim)`.  This avoids a reader
having to cross-reference to find that `C` is a synonym.

**Issue 13 — The `Raises` block documents a `ValueError` that is
actually a hard `raise ValueError` in a conditional but describes it as
an error about `cp_group + is_causal`, whereas the code uses the wording
"has not been verified to work" — the error is intentional and the
docstring should match: "Causal + CP is explicitly rejected because the
combination has not been verified for correctness."**

Current text:

> Raises:
> ValueError: If `cp_group` is provided together with `is_causal=True`
> (this combination has not been verified and may produce incorrect results).

This is close but slightly misleading — the error says "has not been
verified to work" which could imply it *might* work.  The docstring
should say it is **explicitly rejected** and the user must not rely on
the current behaviour silently passing through.

______________________________________________________________________

## Formatting / style

**Issue 14 — `flop_count` docstring uses non-Google-style free-form
section headers ("Phase 1", "Phase 2") inside the main description
without an `Args` separator.**

The `flop_count` docstring was inherited verbatim from the original
code.  Its structure mixes RST-style bullet headers with Google-style
field sections.  In the Phase 1 docstring the `Args` block appears
after a long free-form body that uses `**Phase 1 — …**` bold headers.
This is acceptable in Google style (the convention allows a free-form
description before `Args:`), but the `**Phase 1**` / `**Phase 2**`
headers lose their bold rendering in plain-text help() output because
they use RST syntax.  Convert them to plain headings or indented notes
consistent with the rest of the file (e.g. use `.. note::` blocks or
just indent as `Note:` Google-style).
