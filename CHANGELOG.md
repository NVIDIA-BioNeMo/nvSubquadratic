# Changelog

All notable changes to nvSubquadratic are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## \[0.1.1\]

### Changed

- **The accelerated CUDA kernels are now an opt-in `[cuda]` extra**, so
  `pip install nvsubquadratic` no longer requires the CUDA toolkit and succeeds
  in environments without `nvcc` (e.g. a downstream project's CPU CI).
  `subquadratic-ops-torch-cu12` is a source-only sdist whose build needs `nvcc`;
  as a core dependency it made the package impossible to install anywhere without
  a CUDA toolchain. It is now installed via `pip install 'nvsubquadratic[cuda]'`.
  Every other dependency is unchanged and still part of the default install —
  nvSubquadratic targets GPU workflows and the default remains batteries-included.

- The accelerated kernel (`subquadratic_ops_torch`) is imported lazily on every
  code path (no module-load-time import). Selecting `fft_backend="subq_ops"` (or
  calling the direct causal-conv wrappers) without the kernel installed raises a
  clear `ImportError` hinting `pip install 'nvsubquadratic[cuda]'`. The default
  `fft_backend="torch_fft"` path is portable and needs no CUDA kernel.

- **`megatron-core` and `timm` moved out of core into purpose extras**, since
  neither is needed to import or run the operators:

  - `nvsubquadratic[distributed]` — `megatron-core`, used only by
    `nvsubquadratic.parallel.utils.init_parallel_state` for context-parallel /
    distributed training. Calling it without the extra raises a clear
    `ImportError` hinting `pip install 'nvsubquadratic[distributed]'`.
  - `nvsubquadratic[baselines]` — `timm`, used only by the ConvNeXt UNet baseline
    models in `nvsubquadratic.networks.baselines` (stochastic-depth `DropPath`).
    Building those models with `drop_path > 0` without the extra raises a clear
    `ImportError` hinting `pip install 'nvsubquadratic[baselines]'`.

- **Dropped `protobuf` and `huggingface_hub` from the explicit dependency list** —
  they are not imported anywhere in the package and still arrive transitively via
  `wandb` / `datasets`.

- Extras are now `[cuda]`, `[quack]`, `[dali]`, `[distributed]`, `[baselines]`,
  and `[all]` (= the union of all five).

- `Requires-Python` lowered to `>=3.10` (was `>=3.11`); CI now byte-compiles the
  package on 3.10/3.11/3.12. The published wheel is pure-Python (`py3-none-any`),
  so a single wheel serves all supported interpreters.

### Notes

- Behaviour and public API signatures are unchanged. GPU users who relied on the
  fused `fft_backend="subq_ops"` kernel should add the `[cuda]` extra; the
  default `torch_fft` path is unaffected.

## \[0.1.0\]

- Initial public release.
