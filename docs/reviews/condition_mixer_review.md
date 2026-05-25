# Review: `nvsubquadratic/modules/condition_mixer.py`

Reviewed against the style references in `film.py`, `residual_block.py`, and `sequence_mixer.py`.

______________________________________________________________________

## Issues

### 1. `init_method_in` type annotation is wrong

In `__init__`, the parameter is declared as:

```python
init_method_in: Callable[[torch.Tensor], torch.Tensor] | None = (None,)
```

but the actual call-site is:

```python
init_method_in(hidden_dim)(self.kv_proj.weight.data)
```

This is a **curried** function `fn(dim: int) -> fn(tensor: Tensor) -> None`, exactly the same as in `QKVSequenceMixer`. The annotation should be:

```python
init_method_in: Callable[[int], Callable[[torch.Tensor], None]] | None = (None,)
```

Ditto for `init_method_out`. A new collaborator will write the wrong function and get a runtime error with no diagnostic help.

______________________________________________________________________

### 2. Module docstring does not explain the relationship between `condition` channel dim and `hidden_dim`

The module docstring says "the channel dimension `C` must equal `hidden_dim`" but never explains *why* — the `kv_proj` and `q_proj` are `Linear(hidden_dim, ...)` and operate directly on `condition`, so passing a conditioning tensor whose last dimension is not `hidden_dim` will silently produce wrong-shaped K/V tensors (no shape check is performed). This constraint should be stated as a hard requirement with the consequence of violating it.

______________________________________________________________________

### 3. `forward` docstring does not mention what the inner `mixer` must return

The docstring for `forward` describes the shapes of `q`, `k`, `v` but does not state that the inner `mixer(q, k, v)` must return a tensor with the **same spatial layout and channel dimension as `q`**. Because `mixer` is a black-box `LazyConfig`-instantiated module, a collaborator implementing a custom inner mixer would not know this contract without reading the code.

Add a sentence such as: "The inner `mixer` must return a tensor of shape `(B, *spatial_dims, C)` matching `q`; the output projection `out_proj` will fail with a shape error otherwise."

______________________________________________________________________

### 4. Missing interaction note: `condition_mixer` vs `AdaLNZeroResidualBlock`

`residual_block.py` has two block types: `ResidualBlock` (which *has* a `condition_mixer` branch) and `AdaLNZeroResidualBlock` (which *does not*). A new collaborator reading `condition_mixer.py` has no pointer to this split. The module docstring should note that `QKVConditionMixer` is only used with `ResidualBlock`, not with `AdaLNZeroResidualBlock` (which routes all conditioning through the DiT AdaLN-Zero projection).

______________________________________________________________________

### 5. The `condition_mixer_norm` interaction is not described

`ResidualBlock.forward` applies `condition_mixer_norm` to `x` **before** calling `self.condition_mixer(x, condition)`. The condition mixer itself therefore receives a *normalised* feature map, not the raw residual. Nothing in `condition_mixer.py` documents this; a reader of only this file would assume `x` is the unnormalised residual stream. At minimum, add a note in the `forward` docstring: "In practice `x` has already been passed through `condition_mixer_norm` by the enclosing `ResidualBlock` before this module is called."

______________________________________________________________________

### 6. `ValueError` message for `condition.ndim` mismatch quotes wrong expected value

The error string reads:

```python
f"Got condition.ndim={condition.ndim}, expected {x.ndim}."
```

The word "expected" is ambiguous — it sounds like *only* `x.ndim` is valid, but `2` is also valid (and is handled above this branch). Fix to:

```python
f"Got condition.ndim={condition.ndim}, expected 2 (global vector) or {x.ndim} (matching spatial rank)."
```

______________________________________________________________________

### 7. Class docstring `Attributes` block omits the `hidden_dim` value

Unlike `film.py` (which stores `num_film_layers` and `kernel_hidden_dim` as instance attributes), `QKVConditionMixer` does not persist `hidden_dim`. That is fine, but the `Attributes` block should make clear that `hidden_dim` can be recovered from `self.q_proj.in_features` if needed. This is a minor discoverability issue.

______________________________________________________________________

### 8. No `See Also` cross-reference to `QKVSequenceMixer`

The module docstring compares the condition mixer to FiLM and cross-attention, but does not point to `QKVSequenceMixer` in `sequence_mixer.py`, even though the two modules are structurally parallel (both share the `QKV + inner mixer + out_proj` skeleton). Add a `See Also` block to the class docstring pointing to `QKVSequenceMixer` and `ResidualBlock`.
