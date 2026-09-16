# Mixed Boundary-Condition FFT Convolution

The FFT-convolution operators support **per-axis boundary conditions**: a
single convolution can be periodic on some spatial axes and zero-padded
(linear) on others. This is exposed through
[`mixed_fftconv.py`](../../nvsubquadratic/ops/mixed_fftconv.py) at the op
level and through the `fft_padding` argument of
{class}`~nvsubquadratic.modules.ckconv_nd.CKConvND` at the module level.

______________________________________________________________________

## Motivation

Several PDE datasets have boundaries that are **periodic on some axes and
non-periodic (wall or open) on others**:

| Dataset                        | x        | y        | z    |
| ------------------------------ | -------- | -------- | ---- |
| `rayleigh_benard`              | periodic | wall     | —    |
| `viscoelastic_instability`     | periodic | wall     | —    |
| `turbulent_radiative_layer_2D` | periodic | open     | —    |
| `turbulent_radiative_layer_3D` | periodic | periodic | open |
| `rayleigh_taylor_instability`  | periodic | periodic | wall |

The two global modes (`fft_padding="zero"` or `"circular"`) cannot express
"periodic on x, zero-padded on y" for one and the same convolution. The
mixed path closes that gap.

Wall and open boundaries are both treated as **zero-padded linear** at the
convolution level; physical distinctions (if any) are handled elsewhere
(data normalisation, loss). Per-face boundary conditions, meaning a different
BC on opposite faces of the same axis, are not supported (see
[Limitations](#limitations)).

______________________________________________________________________

## API

`fft_padding` accepts either a **single mode string** (applies to every
axis) or a **list of mode strings** (one per spatial axis):

```python
fft_padding: str | Sequence[str] = "zero"
# "zero"                       -> all axes zero-padded (linear "same" conv).
# "circular"                   -> all axes periodic (wrap-around conv).
# ["circular", "zero"]         -> 2D: x periodic, y zero-padded.
# ["zero", "circular", "zero"] -> 3D mixed.
```

The list form reads identically in Python and in OmegaConf / YAML config
overrides. Internally everything normalises to a per-axis boolean tuple
`periodic: tuple[bool, ...]` of length `data_dim`. Two inputs are
deliberately rejected with an error that redirects to the canonical form:

- **Booleans** (`(True, False)`): the per-axis intent is not obvious from
  the boolean values.
- **Comma-separated strings** (`"circular, zero"`): redundant with the list
  form.

### Kernel size per axis

The kernel grid size is auto-derived from the per-axis boundary condition;
it is **not** a separate knob. When `fft_padding` is a list, the legacy
`grid_type` argument must be `None` (a conflict raises rather than silently
overriding):

| axis BC      | grid length per axis (`CKConvND.forward`) | SIREN kernel size on that axis |
| ------------ | ----------------------------------------- | ------------------------------ |
| periodic     | `(s+1)//2` (≡ `grid_type="single"`)       | `≈ s`                          |
| non-periodic | `s` (≡ `grid_type="double"`)              | `≈ 2s − 1`                     |

______________________________________________________________________

## Algorithm

The mixed N-D FFT convolution is computed in **one** `rfftn` / `irfftn`
call over all spatial dims. The per-axis variation is encoded entirely in
the transform arguments:

| axis is      | FFT length `F_d`               | post-IFFT crop range             | phase ramp shift on that axis |
| ------------ | ------------------------------ | -------------------------------- | ----------------------------- |
| periodic     | `N_d` (no padding)             | `0 : N_d` (no crop)              | `−(K_d − 1) // 2`             |
| non-periodic | `min(N_d + (K_d+1)//2, 2·N_d)` | `K_d//2 : K_d//2 + N_d` (center) | `0` (no shift)                |

Phase ramps are the product of per-axis 1-D ramps (length-1 broadcast on
non-periodic axes, so they contribute nothing there).

Two edge behaviours are guaranteed (and exercised in the tests):

- All axes non-periodic → **bit-identical** to the linear
  [`fftconv`](../../nvsubquadratic/ops/fftconv.py) op.
- All axes periodic → **bit-identical** to the
  [`circular_fftconv`](../../nvsubquadratic/ops/circular_fftconv.py) op.

In those uniform corners the mixed op dispatches internally to the legacy
linear / circular ops, so there is no performance cost for non-mixed usage.

______________________________________________________________________

## What's available

- **Op level** ([`mixed_fftconv.py`](../../nvsubquadratic/ops/mixed_fftconv.py)):
  fp32 1D / 2D / 3D, both BHL and BLH (`_w_reshape`) layouts, and
  channel-chunked variants. Shortcut term and dtype preservation match the
  other ops.
- **Module level** ({class}`~nvsubquadratic.modules.ckconv_nd.CKConvND`):
  pass `fft_padding` as a per-axis list. `use_chunked_fftconv` is supported
  for every per-axis combination (including all-periodic, which the legacy
  string-mode rejected). `flop_count` uses per-axis padded sizes.

______________________________________________________________________

## Limitations

- **Per-face boundary conditions** are out of scope: any non-periodic axis
  maps to symmetric zero-padding. Datasets with different BCs on opposite
  faces of the same axis (e.g. `helmholtz_staircase`,
  `acoustic_scattering_maze`) are approximated by the nearest per-axis BC.
- **Custom CUDA path** (`fft_backend="subq_ops"`) supports zero-padding
  only; combining it with a per-axis `fft_padding` raises. Use
  `fft_backend="torch_fft"` (the default) for mixed boundaries.
- **Auto-wiring**: the periodic axes are specified per config. The Well
  datamodule can read `boundary_conditions` from the HDF5 metadata, but
  model code does not yet consume it to derive `periodic` automatically.
