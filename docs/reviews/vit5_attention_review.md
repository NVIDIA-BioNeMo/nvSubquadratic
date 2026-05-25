# Reviewer Feedback: `nvsubquadratic/modules/vit5_attention.py`

Reviewed after Phase 1 docstring pass.  Issues are numbered for easy tracking by the integrator.

______________________________________________________________________

## Critical — Missing or Incorrect Information

**1. `_build_2d_rope_flat`: `head_dim` divisibility constraint is wrong.**

The docstring states: *"`head_dim` must be divisible by 4."*  However, the code computes `dim_half = head_dim // 2` and uses `torch.arange(0, dim_half, 2)` — which only requires `head_dim` divisible by 2, not 4.  The divisibility-by-4 requirement belongs to `_rotate_half_per_axis` (which needs `d_quarter = d // 4`), not to this function.  Fix: change the Note to say "`head_dim` must be divisible by 2" and move the divisibility-by-4 note to `_rotate_half_per_axis`.

**2. `_rotate_half_per_axis`: missing assertion or note that `head_dim % 4 == 0` is *not* enforced at runtime.**

The docstring says *"`D` must be divisible by 4"* but there is no `assert` or guard in the code.  Passing an odd `head_dim` causes silent integer-truncation in the quarter-splits.  Fix: add "Note: this constraint is not checked at runtime; the caller is responsible for ensuring `head_dim % 4 == 0`."

**3. `ViT5Attention` class docstring: the QK-norm ordering claim is wrong.**

Under **QK normalisation** the docstring says: *"applied to Q and K `after` the per-head reshape but `before` RoPE."*  But in `forward()`, after `qkv.unbind()` the code does `if self.qk_norm: q = self.q_norm(q); k = self.k_norm(k)` and *then* applies RoPE.  So norm is applied **before** RoPE — the docstring is correct there.  However the subsequent `Note:` says *"Norm is applied before RoPE in this module (`q_norm → rope`)"* which agrees.  The class-level prose and the Note are consistent; but the phrase *"after the per-head reshape"* is ambiguous — Q and K come from `qkv.unbind(dim=2)` giving shape `[B, T, H, d_k]`, so the reshape is implicit inside the `unbind`.  Fix: add the explicit intermediate shape `[B, T, H, d_k]` immediately after `unbind` to make "per-head reshape" concrete.

**4. `ViT5Attention.__init__`: no `assert hidden_dim % num_heads == 0` message.**

The code has `assert hidden_dim % num_heads == 0` with no message string.  The class docstring does not document what happens when the constraint is violated.  The generic `Attention.__init__` includes `"hidden_dim must be divisible by num_heads"`.  Fix: add an `AssertionError` entry to the class docstring's Args or add a `Raises` block at the class level, and/or add the message to the `assert`.

**5. `ViT5Attention.__init__`: `num_registers` perfect-square constraint is undocumented in the body.**

The class-level Args block says *"Must be a perfect square when > 0"* but there is no `assert int(num_registers**0.5)**2 == num_registers` in the code.  If a non-square value (e.g. `num_registers=6`) is passed, `reg_rope_h = reg_rope_w = 2` silently and `_build_2d_rope_flat(2, 2, ...)` produces only 4 rows instead of 6, causing a shape mismatch in `torch.cat`.  Fix: either add the assert with an informative message, or document the silent truncation explicitly in the Args block.

______________________________________________________________________

## Moderate — Incomplete or Confusing Documentation

**6. `_build_2d_rope_flat`: Return shape annotation is ambiguous for register tokens.**

The Returns section says shape is `[height * width, head_dim]`.  This is correct but callers (in `__init__`) also pass `reg_rope_h, reg_rope_w` for register tokens.  The docstring does not warn that the *register* call uses a square-root approximation for the grid dimensions, so if `num_registers` is not a perfect square the returned table has fewer rows than `num_registers`.  Fix: add a sentence noting this function is used for both patch and register grids and that the grid dimensions must together equal the intended token count.

**7. `ViT5Attention.forward`: `Raises` section says `RuntimeError` but that's not what PyTorch actually raises.**

The docstring documents: *"`RuntimeError`: If `T` does not match `rope_cos.shape[0]`."*  In practice PyTorch raises a `RuntimeError` on broadcast mismatch, but the message is not user-friendly.  The docstring should additionally note what the actual `T` must equal (`num_patches_h * num_patches_w + int(has_cls) + num_registers`) so users can debug easily.  Fix: expand the `Raises` description to include the expected value formula.

**8. `ViT5Attention.flop_count`: `q_norm.flop_count(T)` signature mismatch with norm modules.**

The docstring says "Delegated to `self.q_norm` / `self.k_norm`" but does not specify whether those norm modules are expected to accept a single `num_tokens` integer or a full shape tuple.  The generic `Attention` FLOP counting differs: it uses global stats.  Fix: add a sentence clarifying the expected signature of the norm's `flop_count` method, e.g. `flop_count(num_tokens: int) -> int`.

**9. `ViT5Attention` class docstring: the `Example` block uses a non-existent import.**

```python
import nvsubquadratic.modules.rms_norm as rms_norm_mod

qk_norm = (LazyConfig(target=rms_norm_mod.RMSNorm, dim=64),)
```

There is no `rms_norm` module at `nvsubquadratic.modules.rms_norm` confirmed in the codebase (the tracker lists it as `[ ]` undocumented/unverified).  The example may fail for readers trying to run it.  Fix: either use an abstract placeholder comment, or use the actual path if confirmed, or replace with a note that any norm accepting `[B, T, H, d_k]` tensors works.

**10. `ViT5Attention` class docstring: register RoPE grid description mixes up base-frequency intuition.**

The docstring says: *"The higher base frequency (lower theta) gives denser angular spacing."*  Standard RoPE convention: higher `rope_base` → lower frequency (slower rotation, sparser angular spacing).  But `reg_rope_base=100` is *lower* than `rope_base=10000`, so `reg_rope_base` gives *higher* frequency rotations (denser spacing).  The parenthetical "(lower theta)" is confusing because `theta_j ∝ rope_base^{-2j/d}` decreases with higher base.  Fix: rewrite as *"A lower base value (`reg_rope_base=100` vs `rope_base=10000`) yields higher rotation frequencies (theta decays more slowly), giving denser angular spacing for register positions."*

______________________________________________________________________

## Minor — Style and Consistency

**11. `ViT5Attention.extra_repr`: docstring does not match `attention.py` style.**

`attention.py`'s `extra_repr` docstring begins with "Return a concise string summary…" and lists the specific keys.  The Phase 1 docstring for `ViT5Attention.extra_repr` says the same but omits `has_cls` and `scale` from the listed parameters even though `extra_repr` does not emit them — which is fine, but `has_cls` is absent from `extra_repr` output yet is a consequential hyperparameter.  Fix (minor): add a note that `has_cls` and `scale` are not printed by `extra_repr`.

**12. Module docstring: cross-reference to `_build_2d_rope_flat` and `_rotate_half_per_axis` uses `:func:` but these are module-private.**

The module docstring cross-references `:func:_build_2d_rope_flat` and `:func:_rotate_half_per_axis`.  These functions are private (leading underscore) and are not exported from the module.  Sphinx will produce broken references unless configured for private member documentation.  Fix: either add `.. autofunction::` directives to the Sphinx config or replace `:func:` with inline monospace references (double-backtick).
