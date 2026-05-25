# Review: `nvsubquadratic/modules/vit5_hyena_adapter.py`

Reviewed after Phase 1 docstring pass. Issues are numbered and actionable.

______________________________________________________________________

## 1. Module docstring: "drop-in replacement" claim needs stronger qualification

The second summary line reads:

> "Drop-in replacement interface for `ViT5Attention`."

This is accurate but incomplete. `ViT5Attention` owns its own QKV/output projections; the
adapter delegates all projections to `inner_mixer`. A reader doing a config swap needs to
know that `inner_mixer_cfg` (e.g. `QKVSequenceMixer`) **must** be configured with the
correct `hidden_dim`/`num_heads` — the adapter accepts no such arguments itself. The
module docstring already says "Projection dimensions must be configured inside
`inner_mixer_cfg`" but this appears deep in the body. Move or repeat this caveat in the
opening summary paragraph so it is immediately visible.

______________________________________________________________________

## 2. Module docstring: shape-preservation contract lives only in item 3, not in the interface section

Item 3 of the adapter steps says the inner mixer must be shape-preserving, but the
"Interface contract" section at the bottom says nothing about this. A reader who reads
only that section will miss the constraint. Add a bullet there: "The inner mixer must
return a tensor of the same shape `[B, H, W, C]` it received; downsampling or strided
mixers are not supported."

______________________________________________________________________

## 3. Class docstring: `Attributes` block omits `inner_mixer` type information

The Attributes block reads:

> `inner_mixer (nn.Module): The instantiated 2-D sequence mixer.`

This is correct but too terse. Add the channels-last expectation: "Accepts and returns
`[B, H, W, C]` tensors (channels-last). Typically a `QKVSequenceMixer` wrapping
`Hyena`."

______________________________________________________________________

## 4. `__init__` docstring: `grid_w` arg does not mention what happens at hierarchical stages specifically enough

The docstring says "In a hierarchical network, pass the correct `grid_w` for each stage
(after patch merging)." but does not tell the reader where to find that value. Add:
"After a 2× patch-merging step, `grid_w` halves; the network's stage configuration
(e.g. `ViT5HierarchicalClassificationNet`) should be the source of truth for each
stage's `grid_w`."

______________________________________________________________________

## 5. `forward` docstring: `mixer_kwargs` description lists only two keys but there may be others

The docstring lists `conditioning` and `cp_group` as "common keys". This is fine, but
add a closing note: "Any additional kwargs accepted by the concrete inner mixer are
also forwarded; consult the inner mixer's docstring for the full list."

______________________________________________________________________

## 6. `forward` docstring: the `Raises` entry is imprecise about what `reshape` actually raises

The current text says `RuntimeError: If T % grid_w != 0 (implicit, from reshape)`.
PyTorch's `reshape` actually raises `RuntimeError` with the message "shape ... is
invalid for input of size ...". To be precise: "RuntimeError: Raised by `torch.Tensor.reshape`
if `T % grid_w != 0`, with a message reporting the mismatched total element count."

______________________________________________________________________

## 7. `extra_repr` docstring: does not mention the return value format

Current: "Return `grid_w=<value>` appended to PyTorch's default module repr."

"Appended" is slightly wrong — `extra_repr` returns a string that PyTorch *inserts*
inside the parentheses of the repr, not appended after. Fix to: "Return the string
`'grid_w=<value>'` inserted into PyTorch's module repr (inside the parentheses)."

______________________________________________________________________

## 8. Missing: note about non-contiguous output from inner mixer and `reshape` fallback

The `forward` Returns section mentions the non-contiguous-copy caveat for the final
`reshape`. However it does not say **which** inner mixers are likely to cause this.
Add: "In practice, `QKVSequenceMixer` returns a contiguous tensor (its output
projection is a `Linear` applied on the last axis), so the final `reshape` is
typically a free view."
