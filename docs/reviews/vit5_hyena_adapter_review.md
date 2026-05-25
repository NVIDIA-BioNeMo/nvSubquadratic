# Review: `nvsubquadratic/modules/vit5_hyena_adapter.py`

Reviewed after Phase 1 docstring pass. Issues are numbered and actionable.

______________________________________________________________________

## 1. Module docstring: "drop-in replacement" claim needs qualification

The second line reads:

> "Drop-in replacement interface for `ViT5Attention`."

This is misleading. `ViT5Attention` owns its own QKV and output projections; the adapter delegates all projections to `inner_mixer`. A reader replacing `ViT5Attention` with `ViT5HyenaAdapter` in config needs to know that `inner_mixer` (e.g. `QKVSequenceMixer`) must be configured with matching `hidden_dim` and the correct `num_heads`/`inner_dim` — the adapter itself accepts no `hidden_dim` argument. The docstring should add a sentence clarifying that the projection parameters must be configured **inside `inner_mixer_cfg`**, not on the adapter.

______________________________________________________________________

## 2. Module docstring: `inner_mixer` contract is underspecified for the 2-D output reshape

Step 4 of the "thin, stateless reshape adapter" list says:

> "4. Reshapes back to `[B, T, C]` and returns."

This silently assumes the inner mixer returns **exactly** `[B, H, W, C]` — the same spatial shape it received. If a mixer returns a different shape (e.g. a downsampling mixer), `reshape(B, T, C)` will raise a cryptic error. The docstring should state explicitly: "The inner mixer is assumed to be shape-preserving — its output must have the same `[B, H, W, C]` shape as its input."

______________________________________________________________________

## 3. Class docstring: data-flow diagram uses `inner_mixer (QKVSequenceMixer → Hyena)` as a fixed example

The class-level data-flow diagram labels the inner_mixer step as:

```
▼  inner_mixer  (QKVSequenceMixer → Hyena)
```

This couples the class docstring to a specific implementation. The class is general — it wraps any `[B, H, W, C]`-preserving module. The parenthetical should be replaced with a more general description, e.g. `(any [B, H, W, C]-preserving mixer)`.

______________________________________________________________________

## 4. `__init__` docstring: `inner_mixer_cfg` contract does not mention channels-last convention

The docstring says the instantiated module must "accept `(x: Tensor[B, H, W, C], **kwargs)`" but does not mention that this is a **channels-last** layout. Hyena internally converts to channels-first `[B, C, H, W]` for convolution. A reader wiring a custom mixer needs to know the expected layout convention. Add: "The tensor is in channels-last layout `[B, H, W, C]`; any inner mixer that uses channels-first convolution (like `QKVSequenceMixer`) handles the permutation internally."

______________________________________________________________________

## 5. `forward` docstring: second `reshape` output contiguity caveat is incomplete

The Returns section states: "The reshape is a view (no data copy) when the tensor is contiguous." However, the inner mixer may return a non-contiguous tensor (e.g. after a transpose). In that case `reshape` falls back to a copy. This is fine correctness-wise but users writing CUDA-graph-safe code or tracing with `torch.compile` should know. Add: "If `inner_mixer` returns a non-contiguous tensor, the final `reshape` may trigger a contiguous copy; this does not affect correctness but can affect memory traffic."

______________________________________________________________________

## 6. `flop_count` docstring: silent `AttributeError` is wrong error type in practice

The Raises section says `AttributeError` if `inner_mixer` does not implement `flop_count`. In practice, if `inner_mixer` has no `flop_count`, Python raises `AttributeError` from the attribute lookup, not from a guarded check. This is correct but the docstring should also note that `flop_count` is a **de-facto protocol** not enforced by an interface — callers that want to guard against this should use `hasattr(adapter.inner_mixer, "flop_count")`.

______________________________________________________________________

## 7. Missing: note about `grid_w` and the token-layout contract for the hierarchical case

The module docstring mentions `ViT5ClassificationNet` as the network that pads and arranges tokens, but the current codebase also has `vit5_hierarchical_classification.py` (feat/patch-merging PR). The register-token handling note in the class docstring only mentions a single "register row" convention but does not mention that after patch merging the grid dimensions change. Add a brief note that `grid_w` must be consistent with the spatial width **after any patch-merging stage**, i.e. the calling network must supply the correct `grid_w` at each hierarchical stage.

______________________________________________________________________

## 8. `extra_repr` docstring: wording is slightly off

Current: "Return a concise summary for `repr()` and `print(model)`."

`extra_repr` is called by PyTorch's `__repr__` machinery. The docstring should say what information it returns rather than how it is called: "Return `grid_w=<value>` appended to PyTorch's default module repr." This matches the style in `vit5_attention.py`'s `extra_repr`.
