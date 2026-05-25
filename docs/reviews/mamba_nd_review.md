# Review: `nvsubquadratic/modules/mamba_nd.py`

Reviewed after Phase 1 docstring pass.  Issues are ordered from highest to
lowest priority.  Each item quotes the relevant text and states the exact fix.

______________________________________________________________________

## 1. ZOH approximation presented as equality

**Location**: module-level docstring, "Background" section.

**Quoted text**:

```
\bar{B}_t = (e^{\Delta_t A} - I) A^{-1} B_t \approx \Delta_t B_t
```

**Issue**: The full ZOH expression is exact, but the approximation
`≈ Δ_t B_t` is only valid for small `Δ_t`.  More precisely, the Mamba paper
(Eq. 4 in arXiv:2312.00752) uses the Euler discretisation `B̄_t = Δ_t B_t`
and notes that the ZOH formula is an alternative; the code in `mamba_ssm`
actually uses the Euler rule for `B`, not the full ZOH.  Presenting the full
ZOH formula and then approximating it may confuse readers into thinking the
implementation uses ZOH for both `A` and `B`.

**Fix**: Clarify that `Ā_t = exp(Δ_t A)` (ZOH) but `B̄_t = Δ_t B_t` (Euler
/ first-order), matching what `mamba_ssm` actually computes.  Remove the
intermediate exact-then-approximate chain or clearly label it "alternative
ZOH formula (not used in practice)".

______________________________________________________________________

## 2. Comparison table: Mamba complexity claim is misleading for training

**Location**: module-level docstring, "Comparison with other mixers" section.

**Quoted text**:

```
| Mamba     | SSM recurrence    | input-dependent    | O(N)                 |
```

**Issue**: O(N) is the *inference* (autoregressive) complexity via the
recurrent form.  During training Mamba uses a parallel associative scan whose
GPU-efficient implementation is O(N log N) in time, or O(N) with the
hardware-aware parallel scan (which requires special CUDA kernels — the whole
point of the `mamba_ssm` package).  Listing O(N) without qualification makes
it look strictly cheaper than Hyena at training time, which is only true with
the custom CUDA kernels.

**Fix**: Add a note distinguishing training (parallel scan, O(N) with custom
kernels or O(N log N) naively) from inference (recurrent, O(N) per step, O(1)
state size).  E.g. add a "Notes" column or a footnote: "O(N) with hardware-
aware parallel scan (requires `mamba_ssm` CUDA extension); O(1) per step at
inference".

______________________________________________________________________

## 3. Scan order: no mention of the implication for 2D spatial locality

**Location**: module-level docstring, "ND generalisation strategy" section, and
class-level docstring paragraph starting "**Scan order for ND inputs**".

**Quoted text**:

```
The scan order for multi-dimensional inputs follows the default PyTorch /
``einops`` row-major (C-contiguous) flattening: for a 2D ``[H, W]`` input the
tokens are visited in raster-scan order (row 0, col 0 → row 0, col W-1 →
row 1, col 0 → …).
```

**Issue**: This is accurate but omits the key practical consequence: tokens
that are spatially adjacent *vertically* (same column, adjacent rows) are far
apart in the flattened sequence (W steps away), so the SSM's effective
receptive field is anisotropic — it sees horizontal neighbours cheaply but
vertical neighbours only through W state-update steps.  An external
collaborator reading this to decide whether to use Mamba for a 2D task needs
this information.

**Fix**: Add one sentence explicitly warning about vertical anisotropy, e.g.:
"Note that vertically adjacent pixels (same column, adjacent rows) are W
tokens apart in the flattened sequence; for tall images this means the forward
SSM sees them only through many state transitions, potentially losing spatial
correlation.  Bidirectional mode partially mitigates this."

______________________________________________________________________

## 4. `forward` docstring: `x` is mutated (overwritten by `rearrange`)

**Location**: `Mamba.forward`, Args section and implementation.

**Quoted text** (implementation):

```python
x = rearrange(x, "b ... c -> b (...) c")
```

**Issue**: The `rearrange` result is assigned back to `x`, which shadows the
original argument.  The original spatial shape is captured in `x_shape` first,
so this is safe, but the docstring does not warn that the local variable `x`
changes meaning mid-function (it is the flattened view from line 254 onward).
This is a mild readability issue but analogous to the note in `hyena_nd.py`'s
`forward` docstring about `query` being overwritten — be consistent.

**Fix**: Add an "Implementation note" paragraph to `forward`:
"The local variable `x` is rebound to the flattened `[B, S, C]` view after
the `rearrange` call; the original spatial shape is preserved in `x_shape`
for the final `reshape`."

______________________________________________________________________

## 5. `__init__` docstring: no mention that `core_layer_rev` has independent parameters

**Location**: `Mamba.__init__`, Args section for `mamba_layer_cfg`.

**Quoted text**:

```
The config is instantiated once for ``core_layer`` and, when
``bidirectional=True``, a second independent instantiation is created for
``core_layer_rev`` so that the two directions have separate parameters.
```

**Issue**: The wording says "independent instantiation … separate parameters"
but does not say *how* independence is achieved — a reader unfamiliar with
`LazyConfig` might wonder if the second call shares weights via some internal
cache.

**Fix**: Add a clarifying phrase: "`instantiate(mamba_layer_cfg)` is called
twice with the same config; each call constructs a fresh `nn.Module` with
newly initialised weights, so the two directions do not share parameters."

______________________________________________________________________

## 6. Missing `Raises` section in `__init__`

**Location**: `Mamba.__init__` docstring.

**Issue**: If `mamba_layer_cfg` cannot be instantiated (wrong target class,
missing required arguments), `instantiate` will raise — most likely a
`RuntimeError`, `TypeError`, or an `omegaconf` exception, depending on the
`LazyConfig` backend.  The `QKVSequenceMixer.__init__` docstring in
`sequence_mixer.py` includes a `Raises` section for this; `Mamba.__init__`
should match.

**Fix**: Add:

```
Raises:
    Exception: Propagated from
        :func:`~nvsubquadratic.lazy_config.instantiate` if
        ``mamba_layer_cfg`` cannot be constructed.  Check that the target
        class accepts ``[B, S, C]`` tensors and that all required constructor
        arguments are provided in the config.
```

______________________________________________________________________

## 7. Class-level Attributes block: `core_layer_rev` uses vague conditional phrasing

**Location**: `Mamba` class, Attributes block.

**Quoted text**:

```
core_layer_rev (torch.nn.Module): The reverse Mamba core.  Only
    present when ``bidirectional=True``; accessing this attribute
    when ``bidirectional=False`` raises :class:`AttributeError`.
```

**Issue**: Saying "raises `AttributeError`" is accurate but alarming without
context — it looks like a bug.  It would be clearer to note the attribute is
intentionally absent (not registered) to keep `state_dict` / `parameters()`
clean when unused.

**Fix**: Reword to: "The reverse Mamba core, instantiated only when
`bidirectional=True`.  When `bidirectional=False` this attribute is not
registered and accessing it raises :class:`AttributeError` by design, keeping
the module's parameter count and `state_dict` unaffected."

______________________________________________________________________

## 8. Module docstring: `References` section uses non-standard heading style

**Location**: module-level docstring, last section.

**Quoted text** (after ruff fix):

```
References:
----------
```

**Issue**: Sphinx / NumPy / Google doc conventions all write `References` with
the underline immediately under the heading, not with a blank line.  The ruff
linter already fixed a trailing colon issue here but the section may still
render oddly in Sphinx because of the mixed `References:` + underline style.
The rest of the module docstring uses RST underlines without trailing colons
on the heading (e.g. `Background\n----------`).

**Fix**: Change to match the rest of the file:

```
References
----------
```

(remove the trailing colon from `References:`).
