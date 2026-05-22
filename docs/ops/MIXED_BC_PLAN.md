# Mixed Boundary-Condition FFT Convolution — Plan & Tracker

**Status:** In progress (v1 ops + tests)
**Branch:** `dwromero/mixed-bc-fftconv`
**Owner:** dwromero
**Started:** 2026-05-20

This is the working plan and tracker for adding **per-axis boundary
condition** support to the FFT-based convolution operators
(`nvsubquadratic/ops/`) and the modules that consume them
(`nvsubquadratic/modules/ckconv_nd.py`, etc.).

If you pick this up later, read the [survey & decisions](#1-context--decisions)
section, then check [§5 Tracked questions](#5-tracked-questions--revisit-items)
for what still needs to be done.

______________________________________________________________________

## 1. Context & decisions

### Motivation

Several Well PDE datasets have boundaries that are **periodic on some axes
and non-periodic (WALL or OPEN) on others**, e.g.

| Dataset                        | x        | y        | z    |
| ------------------------------ | -------- | -------- | ---- |
| `rayleigh_benard`              | periodic | wall     | —    |
| `viscoelastic_instability`     | periodic | wall     | —    |
| `turbulent_radiative_layer_2D` | periodic | open     | —    |
| `turbulent_radiative_layer_3D` | periodic | periodic | open |
| `rayleigh_taylor_instability`  | periodic | periodic | wall |
| `helmholtz_staircase`          | open     | per-face | —    |
| `acoustic_scattering_maze`     | per-face | per-face | —    |

Today the FFT-conv operators expose only two **global** modes
(`fft_padding="zero"` or `"circular"`) selected per module. There is no way
to say "periodic on x, zero-padded on y" for one and the same conv.

### Decisions (locked in)

1. **API** — extend `fft_padding` to accept either a **single mode
   string** (applies to every axis) or a **list of mode strings** (one
   per spatial axis):

   ```python
   fft_padding: str | Sequence[str] = "zero"
   # "zero"                              -> all axes zero-padded.
   # "circular"                          -> all axes periodic.
   # ["circular", "zero"]                -> 2D, x periodic + y zero-padded.
   # ["zero", "circular", "zero"]        -> 3D, etc.
   # ("circular", "zero")                -> tuple form is equivalent.
   ```

   Internally everything normalises to a tuple `periodic: tuple[bool, ...]`
   of length `data_dim`. Three inputs are deliberately rejected with an
   error that redirects to the canonical form:

   - **Booleans** (`(True, False)`, `True`): the per-axis intent is not
     obvious from the boolean values.
   - **Comma-separated strings** (`"circular, zero"`): redundant with the
     list form and gives two ways to say the same thing; we keep one
     canonical per-axis form.

1. **WALL vs OPEN** — both treated as **zero-padded linear** at the
   conv level. Physical distinctions are handled elsewhere (data
   normalisation, loss, etc.). Per-face BC (different BC on opposite faces
   of the same axis) is **out of scope** for v1.

1. **Kernel size per axis** — auto-derived from the per-axis BC, **not** a
   new knob:

   | axis BC      | grid_lens per axis (in `CKConvND.forward`)   | SIREN kernel size on that axis |
   | ------------ | -------------------------------------------- | ------------------------------ |
   | periodic     | `(s+1)//2` (≡ today's `grid_type="single"`)  | `≈ s`                          |
   | non-periodic | `s`         (≡ today's `grid_type="double"`) | `≈ 2s − 1`                     |

   When `fft_padding` is a tuple, the legacy `grid_type` argument must be
   `None` (or omitted) — raise on conflict, no silent overrides.

1. **First-PR scope** — **ops + tests only.** Module wiring
   (`CKConvND`/`CKConvMultiheadND`), Well config updates, fp16, multihead,
   and the `subq_ops` CUDA path are explicitly deferred (see §4).

1. **`subq_ops` CUDA kernel** — left at zero-only. Any
   `fft_backend="subq_ops"` + mixed BC will raise in the future
   `CKConvND` wiring PR.

______________________________________________________________________

## 2. Algorithm — per-axis recipe

The mixed N-D FFT convolution applies, **independently per spatial axis**:

| axis is      | FFT length `F_d`               | post-IFFT crop range             | phase ramp shift on that axis |
| ------------ | ------------------------------ | -------------------------------- | ----------------------------- |
| periodic     | `N_d` (no padding)             | `0 : N_d` (no crop)              | `−(K_d − 1) // 2`             |
| non-periodic | `min(N_d + (K_d+1)//2, 2·N_d)` | `K_d//2 : K_d//2 + N_d` (center) | `0` (no shift)                |

The whole conv is still **one** `rfftn` / `irfftn` over all spatial dims;
the per-axis recipe just feeds different per-axis `F_d` values to `s=` and
different per-axis slices to the post-IFFT crop. Phase ramps are the
product of per-axis 1-D ramps (length-1 broadcast on non-periodic axes).

Edge behaviour required by the tests:

- All `periodic == False` → bit-identical to existing `fftconv*` linear op.
- All `periodic == True`  → bit-identical to existing `circular_fftconv*` op.

______________________________________________________________________

## 3. v1 deliverables (this PR)

### Code

- [x] `nvsubquadratic/ops/mixed_fftconv.py` (fp32):
  - Self-contained per-axis 1-D phase-ramp LRU cache; N-D ramp built on
    demand by broadcasted multiplication so non-periodic axes contribute
    nothing (skipped, not just length-1).
  - 1D / 2D / 3D BHL variants.
  - BHL `_w_reshape` wrappers (BLH inputs).
  - Channel-chunked variants.
  - Automatic dispatch to the existing linear / circular ops in the
    all-False / all-True cases (no perf cost for legacy usage).
- [x] No `nvsubquadratic/ops/__init__.py` exists — callers import from
  submodules directly (matches existing convention).

> **Note:** an earlier draft of this plan called for extracting
> `_PhaseRampCache1D/2D/3D` from `circular_fftconv.py` into a shared
> `_phase_ramp.py`. We **dropped** that refactor for v1 because the existing
> caches hard-code `FFT_shape == input_shape` per axis, which is only true
> for the all-circular case. There is nothing to share without first
> generalising the API. We may revisit this as a unification refactor
> (see §5 Q4).

### Tests

- [x] `tests/ops/test_mixed_fftconv.py` — 76 tests, all passing on H100:
  - Reference comparison against
    `F.pad(x, mode="circular"|"constant")` + `F.conv{1,2,3}d(padding=0)`
    for every per-axis combo across 1D / 2D / 3D, including the K==N
    edge case and even-K kernels (asymmetric "same" padding).
  - Sanity: all-False matches existing `fftconv*`; all-True matches
    existing `circular_fftconv*`.
  - BHL ↔ BLH wrapper equivalence.
  - Chunked vs non-chunked equivalence.
  - Backward / gradient equivalence vs the spatial reference (1D, 2D, 3D).
  - `use_phase_shift=False` (roll on periodic axes only) matches
    `use_phase_shift=True` for every combo.
  - Shortcut term equivalence and dtype preservation (fp32, bf16).
  - Validation errors: wrong `periodic` length, oversized kernel,
    mismatched shortcut dtype.
- [x] Re-ran existing FFT-conv suites — **101 passed**, no regressions:
  - `tests/ops/test_fftconv.py`
  - `tests/ops/test_circular_fftconv.py`
  - `tests/ops/test_fftconv_chunked.py`

### Docs

- [x] Updated `docs/ops/README.md` "File map" with the new
  `mixed_fftconv.py` row.

______________________________________________________________________

## 4. Deferred — follow-up work

Each item below is intentionally **not** part of this PR. They are
listed so we don't lose track.

### 4.1 Module wiring — `CKConvND` ✅ DONE (2026-05-20)

- [x] `CKConvND` (`nvsubquadratic/modules/ckconv_nd.py`):
  - Accepts `fft_padding: str | Sequence[str]` in two forms: a single
    mode string (`"zero"` / `"circular"`) that applies to every axis, or
    a list of mode strings (e.g. `["circular", "zero"]`) — one per axis.
  - Resolves to a normalised per-axis `periodic` tuple via
    `_resolve_periodic` (length checked against `data_dim`).
  - When `fft_padding` is a per-axis list, **requires** `grid_type=None`
    and raises `ValueError` otherwise. When it's a single mode string,
    `grid_type` is required as before.
  - Boolean inputs (e.g. `(True, False)`) and comma-separated strings
    (`"circular, zero"`) are explicitly rejected with errors that
    redirect to the list form.
  - Per-axis `grid_lens` and per-axis `L_cache` halving auto-derived in
    tuple mode (halve only on periodic axes); helper
    `_grid_is_single_per_axis(grid_type, periodic)` is the single source
    of truth used by both `__init__` (L_cache) and `forward`/`flop_count`.
  - Dispatch: tuple mode routes through `MIXED_FFT_FUNCTIONS[_CHUNKED]`
    (wrapped by `_wrap_mixed_op` to bind `periodic`). All-False / all-True
    tuples internally fall back to the legacy linear / circular ops
    bit-identically (verified by tests). String mode keeps the legacy
    `FFT_FUNCTIONS` tables unchanged.
  - Validation:
    - `is_causal=True` + any periodic axis → `ValueError`.
    - `fft_backend="subq_ops"` + tuple `fft_padding` → `ValueError`.
    - `use_fp16_fft=True` + tuple `fft_padding` → `NotImplementedError`
      (planned for v2; see §4.2).
    - `use_chunked_fftconv` allowed with tuple `fft_padding` for **all**
      per-axis combos (including all-True — new capability vs the legacy
      string-mode where circular + chunked was an error).
  - `flop_count` uses per-axis padded sizes: `s` on periodic axes,
    `min(s + (k+1)//2, 2*s)` on non-periodic axes.
- [x] `mixed_fftconv.py` op: `K <= N` assertion relaxed on non-periodic
  axes to `K <= 2*N` to match the "double-grid" SIREN kernel size
  (`2N - 1`) that `CKConvND` produces on non-periodic axes.
- [x] `mixed_fftconv.py`: added `*_w_reshape_chunked` BLH wrappers so the
  module dispatch table is symmetric with the legacy chunked path.
- [x] Module-level tests in `tests/modules/test_ckconv_nd_mixed_bc.py`:
  resolver / helper unit tests, validation errors, per-axis kernel
  shape, tuple-vs-string bit-identical equivalence, mixed-mode
  forward correctness (matches the underlying op called directly),
  BHL/BLH layout equivalence, chunked-vs-non-chunked, and FLOP
  accounting (mixed sits between all-zero and all-circular).

### 4.2 fp16 variant (deferred)

### 4.2 fp16 variant

- [ ] `nvsubquadratic/ops/mixed_fftconv_fp16.py`:
  - Per-axis cuFFT-fp16 constraints: pad-up to power-of-2 on linear axes;
    require input dim power-of-2 on periodic axes (fallback to fp32 with a
    warning otherwise).
  - Reuse centering / DC-correction logic from `circular_fftconv_fp16`
    on the periodic axes only.
- [ ] `tests/ops/test_mixed_fftconv_fp16.py`.
- [ ] `CKConvND` fp16 dispatch update.

### 4.3 2D multi-head variant

- [ ] `fftconv2d_multihead_mixed_bhl` (and `_bhi`) in
  `nvsubquadratic/ops/fftconv_multihead.py`.
- [ ] `CKConvMultiheadND` wiring + tests (same shape of changes as
  `CKConvND` above; 2D only).

### 4.4 Well experiment configs

- [ ] Add Hyena variants with mixed BC for the datasets that need them.
  Start with the ones we actually run:
  - `rayleigh_benard`              — periodic on x → `(True, False)`
  - `viscoelastic_instability`     — periodic on x → `(True, False)`
  - `turbulent_radiative_layer_2D` — periodic on x → `(True, False)`
  - `turbulent_radiative_layer_3D` — periodic on x,y → `(True, True, False)`
  - `rayleigh_taylor_instability`  — periodic on x,y → `(True, True, False)`
- [ ] Decide whether to fix `examples/well/v1/supernova_explosion_64/cfg_hyena.py`
  which uses `FFT_PADDING="circular"` but the dataset is all-OPEN. v2 is
  already corrected.
- [ ] Longer-term: read Well HDF5 `boundary_conditions` from the
  datamodule and auto-derive `periodic_axes` instead of hard-coding it
  per config.

### 4.5 Per-face BCs

- [ ] Investigate whether `acoustic_scattering_maze` and
  `helmholtz_staircase` benefit from a *per-face* BC treatment
  (different BC on opposite faces of the same axis).
- [ ] If yes: design a follow-up that goes beyond per-axis circular/linear.
  v1 maps any non-periodic axis to symmetric zero-pad.

### 4.6 Custom CUDA path (`subq_ops`)

- [ ] If/when we want mixed-BC to use the
  `subquadratic_ops_torch.fft_conv2d` fast path, the upstream kernel
  must grow per-axis BC support. v1 leaves the kernel zero-only and
  raises in `CKConvND`.

### 4.7 SIREN kernel generator anisotropy

- [ ] Re-verify that the SIREN kernel module
  (`nvsubquadratic/modules/kernels_nd.py`) actually handles
  anisotropic `grid_lens` (e.g. `(64, 128)`) cleanly end-to-end —
  including positional embedding, masks, monitors, and FLOP accounting.
  Used today for the all-isotropic case; we need it for the per-axis
  grid in the mixed path.

______________________________________________________________________

## 5. Tracked questions & "revisit" items

These are not bugs we plan to fix in this PR, just things we noticed
along the way that may want attention later.

### Q1. Silent mutation of user-provided `L_cache` in `CKConvND.__init__`

`ckconv_nd.py` (~L266–289) does `copy.deepcopy(kernel_cfg)` and then
silently halves `L_cache` when `grid_type=="single"` so the SIREN
positional grid spans `[-1, 1]` over the actual kernel size. The
behaviour is *intentional* (per the in-code comment) but the **silent
mutation of a user-provided config** is mildly surprising. Cleaner
pattern would be to compute the effective `L_cache` at construction
time without round-tripping through the config object. Not a v1 bug —
revisit in a separate refactor.

For the mixed path, the same adjustment must become **per-axis** (halve
only on periodic axes); design that in the module-wiring PR (§4.1).

### Q2. `subq_ops` 2D linear kernel — make it BC-aware?

Out of scope for v1, but worth a conversation with the kernel authors
before we commit to a divergent fast path that only supports zero-pad.

### Q3. Auto-wiring from Well HDF5 boundary_conditions

`experiments/datamodules/pde/well.py` can return per-sample BC metadata
from HDF5, but no model code consumes it. Long-term it would be cleaner
to derive the `periodic_axes` tuple from the dataset rather than
hard-coding in each config.

### Q4. Unify the phase-ramp cache across `circular_fftconv` and `mixed_fftconv`

After v1, the codebase will have two parallel phase-ramp caches:

- The original `_PhaseRampCache1D/2D/3D` in `circular_fftconv.py`, which
  hard-codes `FFT_shape == input_shape` per axis (only valid for the
  all-circular case).
- A new general N-D cache local to `mixed_fftconv.py`, which also handles
  per-axis padded `F_d` and zero shifts on linear axes.

These can be unified into a single shared helper once we are happy with
the mixed op's API. Doing so is a pure refactor (no behaviour change for
either op) and should be its own PR.

______________________________________________________________________

## 6. Changelog

- **2026-05-20** — Plan written, feature branch created.
- **2026-05-20** — v1 ops landed: `mixed_fftconv.py` (fp32, 1D/2D/3D,
  BHL + BLH wrappers + channel-chunked), 76-test suite passing,
  no regressions in existing FFT-conv tests.
- **2026-05-20** — Tolerance tightening + analytical-truth tests
  (impulse response, DC response) added; 102 op tests passing.
- **2026-05-20** — `CKConvND` integration landed: per-axis
  `fft_padding: Sequence[bool]` API, auto-derived per-axis grid +
  `L_cache` halving, unified dispatch via `MIXED_FFT_FUNCTIONS*`,
  module-level test suite (`tests/modules/test_ckconv_nd_mixed_bc.py`).
  Full regression sweep on `tests/ops + tests/modules` → 802 passed.
- **2026-05-21** — Public API revised on PR review: `fft_padding` now
  accepts mode-name strings only. Two forms: a single mode string
  (`"zero"` / `"circular"`) that applies to every axis, or a list of mode
  strings (e.g. `["circular", "zero"]`) — one per axis. Comma-separated
  strings and bool tuples are both rejected with redirecting errors.
  Rationale: `(True, False)` did not convey which axis was periodic, and
  having both a comma-string form and a list form was two ways to say
  the same thing. The list form reads identically in Python and OmegaConf
  / YAML overrides. `rayleigh_benard` config updated to the list form.
