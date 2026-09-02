# Changelog

All notable changes to nvSubquadratic are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## \[Unreleased\]

### Added

- **`fft_backend="subq_ops_fused"` on `CKConvND`**, backed by
  `subquadratic_ops_torch.fused_fft_conv2d`. It runs the whole
  rfft2 → multiply → irfft2 pipeline in a single cuFFTDx launch and, unlike
  every other FFT path, **natively in fp32/fp16/bf16** instead of upcasting to
  fp32. Roughly 3–4× faster than `torch_fft` and 1.2–2.4× faster than
  `subq_ops` on forward+backward in bf16.

  Restricted to `data_dim=2`, `is_causal=False`, `fft_padding="zero"`, and
  spatial extents of at most **64 per axis** — the kernel's largest FFT tile is
  128 and it requires `max(X, Y) <= fft_size // 2`. The spatial cap is enforced
  on the first forward pass rather than at construction, because the input size
  is not known when the module is built. Per-sample (FiLM) kernels are
  supported. Requires `subquadratic-ops-torch >= 0.3.0`. Note the spatial cap is
  not the only hardware constraint: extents above 32 per axis resolve to the 128
  FFT tile, which needs more shared memory than SM80/SM86 provide, so those
  shapes require compute capability 9.0+ and raise a clear error below it.

  The upstream kernel crops the 'same' window at `fft_size // 2` whereas
  `fftconv.py` crops at `K // 2`; the wrapper pre-pads the filter's top/left by
  the difference so results are interchangeable with the other backends
  (verified to ~3e-7 normwise in fp32). Without that pre-pad the output is
  shifted by `fft_size // 2 - K // 2` pixels.

- **`nvsubquadratic.ops.fftconv_lowering`** — a `torch.compile` pre-grad pass
  that detects `fftconv.py`'s 2D FFT-conv chain and rewrites it onto the fused
  kernel, so a model already on `fft_backend="torch_fft"` picks it up without a
  config change. This matters because inductor cannot generate code for complex
  operators and otherwise falls back to eager cuFFT for the entire chain.

  Enable it per-callable with
  `torch.compile(model, options=fused_fftconv2d_options())`, or globally for a
  scope with the `fused_fftconv2d_lowering()` context manager when a framework
  owns the `torch.compile` call.

  The pass only fires on an exact match of the reference recipe (padding rule,
  crop offset, shape limits, CUDA device, and a compute capability that
  supports the required FFT tile — the 128 tile needs SM90+). `lowering_stats()`
  reports rewrite and per-reason skip counts, since a silent pass is otherwise
  hard to tell apart from one that never fired.

### Fixed

- `tests/conftest.py` only queried the `subquadratic-ops-torch-cu12`
  distribution when resolving the installed kernel version. On a
  `-cu13` install (what `pyproject.toml` pins) it resolved to `(0, 0, 0)`,
  which silently turned `requires_subq_ops_v2` into a blanket `xfail` and hid
  every `subq_ops` test result. It now checks both distributions.

### Changed

- **Bumped the CUDA runtime from 12.9 to 13.0.** The Docker image now builds on
  `nvcr.io/nvidia/cuda:13.0.3-devel-ubuntu22.04` with PyTorch `cu130` wheels, and
  the CUDA extras target CUDA 13: `[cuda]` → `subquadratic-ops-torch-cu13`
  (`>=0.3.0`, the first public release carrying `fused_fft_conv2d` — see Added
  above), `[dali]` → `nvidia-dali-cuda130`. The default `fft_backend="torch_fft"`
  path is unaffected and still needs no CUDA kernel.

- **Raised the PyTorch floor to `>=2.14.0,<2.15.0`** (`torchvision >=0.29.0`).
  This is a hard requirement of the `[cuda]` extra, not a preference: torch pins
  `nvidia-cudnn-cu13` *exactly*, and `subquadratic-ops-torch-cu13 >= 0.3.0`
  requires `nvidia-cudnn-cu13 >= 9.24.0.43`. torch 2.12 and 2.13 both pin
  `==9.20.0.48`, which makes `pip install nvsubquadratic[cuda]` unresolvable;
  torch 2.14 pins `==9.24.0.43` and satisfies both.

  The documented conda/venv installs now pin `torch==2.14.0` /
  `torchvision==0.29.0` to match the Docker image, so extensions built during
  the image build match the torch that ships. `scripts/check_version_pins.py`
  enforces that agreement in pre-commit and CI across all 14 pin sites.

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
