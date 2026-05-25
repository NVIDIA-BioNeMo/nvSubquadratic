# Review: `nvsubquadratic/modules/sequence_mixer.py`

Reviewed as a new external collaborator reading alongside the research paper.

______________________________________________________________________

## Issues

### 1. Module docstring: "dispatch" contract is stated but not defined precisely

**Quoted text:** "Any class whose constructor accepts `(q, k, v, cp_group, **kwargs)` in its
`forward` method can be used as the inner mixer."

**Fix:** This describes the `forward` signature but says "constructor" — confusing wording.
Change "constructor" to "the `forward` method". Also, the positional nature of `cp_group`
(it is passed positionally, not as a keyword) should be stated explicitly, because an inner
mixer that defines `cp_group` as `**kwargs` would silently fail to receive it. A one-liner
protocol note would help:

```
The inner mixer forward must accept: forward(q, k, v, cp_group, **kwargs)
where cp_group is passed positionally as the fourth argument.
```

### 2. Module docstring: `MambaNd` class name is wrong

**Quoted text:** `:class:`~nvsubquadratic.modules.mamba_nd.MambaNd\`\`

**Fix:** Check the actual class name in `mamba_nd.py`. If it is `MambaND` (uppercase D), fix
the cross-reference. A broken `:class:` reference silently produces a literal string in Sphinx.

### 3. Class docstring: Attributes block uses wrong "shape" notation for Linear weights

**Quoted text:** "qkv_proj (torch.nn.Linear): Combined Q+K+V input projection, shape `(C, 3·C)`."

**Fix:** `torch.nn.Linear` stores weights with shape `(out_features, in_features)` = `(3C, C)`,
not `(C, 3C)`. The docstring currently states the transpose. Either correct to `(3C, C)` or
just drop the weight-shape hint and write "maps `C` → `3C`" to avoid the ambiguity.

### 4. `__init__` docstring: `Raises` block is under-specified

**Quoted text:** "RuntimeError: Propagated from `instantiate(mixer_cfg)` if the target class
cannot be constructed (e.g. missing required args)."

**Fix:** In practice `instantiate` raises `omegaconf.errors.InstantiationException` or
`hydra._internal.utils.HydraException` (not plain `RuntimeError`) depending on the backend.
If `LazyConfig` uses a different backend, document the actual exception type. At minimum,
note that the exception originates from `LazyConfig.instantiate` and suggest the user check
the `mixer_cfg` target and arguments.

### 5. `__init__` docstring: initialiser signature example is missing a concrete use case

**Quoted text:** "Typically a scaled initialiser (e.g. `1 / sqrt(num_layers)`) to control
residual branch variance."

**Fix:** Add a one-line concrete example showing what `init_method_out` looks like in practice,
so a reader knows what curried form to write:

```python
import math

init_method_out = lambda dim: lambda w: torch.nn.init.normal_(
    w, std=1 / math.sqrt(num_layers)
)
```

This follows the GPT / Megatron pattern and is the primary use case; without an example the
curried signature is hard to guess.

### 6. `flop_count` docstring: trailing whitespace in formula line

**Quoted text:** "`2 · T · D² `" (note the trailing space before the closing backticks).

**Fix:** Remove the trailing space: ``` "``2 · T · D²``" ```. Minor, but looks sloppy in rendered
Sphinx HTML.

### 7. `flop_count` docstring: does not mention bias FLOPs or that biases are ignored

**Quoted text:** "Uses the standard multiply-accumulate convention…"

**Fix:** State explicitly that bias additions are excluded from the count (standard in ML FLOP
counting). If biases are eventually included, callers would get wrong numbers. One line suffices:
"Note: bias additions are excluded, following the standard ML FLOP-counting convention."

### 8. `flop_count` docstring: inner mixer delegation may raise `AttributeError`

**Quoted text:** "Delegated to `self.mixer.flop_count(spatial_dims, inference)`."

**Fix:** Not all inner mixers implement `flop_count`. If `self.mixer` is an `Attention` module
that lacks this method, the call raises `AttributeError` at runtime. Add a `Raises` entry:

```
Raises:
    AttributeError: If the inner mixer does not implement ``flop_count``.
```

### 9. `forward` docstring: the `cp_group=None` type annotation in the signature is wrong

**Quoted text (function signature):**

```python
def forward(self, x: torch.Tensor, cp_group: torch.distributed.ProcessGroup = None, ...
```

**Fix:** The type annotation should be `torch.distributed.ProcessGroup | None = None`, not just
`torch.distributed.ProcessGroup = None`. The current annotation misleads static analysers and
readers into thinking `None` is not a valid value. This is a code change, but fixing it is
better than papering over it in the docstring.

### 10. `forward` docstring: `conditioning` kwarg shape is under-specified

**Quoted text:** "`conditioning` (torch.Tensor, shape `(B, cond_dim)`): FiLM conditioning signal"

**Fix:** The conditioning tensor shape depends on how the `condition_mixer` inside Hyena
processes it — it may be `(B, cond_dim)` before FiLM but the actual contract at the
`QKVSequenceMixer.forward` boundary should be clarified. Also document that passing
`conditioning` to a mixer that does not use it is silently ignored (it falls into
`**mixer_kwargs`), so callers don't need to guard against this.

### 11. Module docstring: no note on how to add a new mixer type

**Fix:** A new external collaborator trying to plug in a new operator (say, RWKV) needs to
know exactly what interface to implement. Add a short paragraph or `Note:` block:

```
Note:
    To register a new mixer type, implement a ``torch.nn.Module`` whose
    ``forward(q, k, v, cp_group, **kwargs)`` method follows the channels-last
    convention ``[B, *spatial, C]`` and, optionally, implement
    ``flop_count(spatial_dims, inference) -> int``.  Then pass its
    ``LazyConfig`` as ``mixer_cfg`` to :class:`QKVSequenceMixer` — no other
    changes are needed.
```

### 12. Class docstring: the ASCII diagram does not show where biases are applied

**Quoted text:**

```
x  ─[Linear(C, 3C)]──► split ──► Q, K, V
```

**Fix:** The diagram is clear, but does not indicate that `qkv_bias` and `out_proj_bias`
optionally add bias terms. Since these default to `False` this is minor, but a parenthetical
`[+ bias?]` on each linear step would make the diagram fully accurate for the non-default
case and prevent confusion when a reader sees biases in a checkpoint.
