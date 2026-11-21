# Changelog

All notable changes to nvSubquadratic are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## \[Unreleased\]

### Added

- **`TriangleHyena`** (`nvsubquadratic/modules/tri_hyena.py`): TriHyena, the
  subquadratic replacement for triangle attention on pair tensors
  `[B, L, L, C]`. An inner mixer for `QKVSequenceMixer` that runs a Hyena
  operator along every row (starting node) or column (ending node) with
  `CKConvND` (SIREN filter + window) as the long conv and a channel-last causal
  short conv (`short_conv_backend="subq_ops"` for the CUDA kernel).
  `load_reference_state_dict` ports checkpoints of the reference TriHyena
  implementation.
- **`BoxModulationND`** (`masks_nd.py`): hard filter window measured in kernel
  taps (the reference TriHyena `max_offset` crop), alongside the soft Gaussian
  and exponential windows.
- **1D non-causal `fft_backend="subq_ops"` on `CKConvND`**, backed by
  `subquadratic_ops_torch.fft_conv1d` and, for bf16/fp16 inputs on compute
  capability 10.x at its supported lengths, the Toeplitz-GEMM `explicit_conv1d`
  (`ops/fftconv_custom.py`: `fftconv1d_bhl` and variants). Runs in the input
  dtype. `ops/causal_conv1d_custom.causal_conv1d` gains `channel_last`.
- `benchmarks/modules/bench_tri_hyena.py`: TriHyena benchmark (window, long-conv
  backend, eager / compiled, starting / ending node).
- `CKConvND(use_shortcut=False)` drops the learnable `shortcut` parameter and
  the fused `shortcut * x` term; the default and existing state dicts are unchanged.
- `CKConvND(allow_reduced_precision=False)` computes bf16/fp16 activations in
  float32 on every FFT backend: it hands the backend float32 activations and
  kernels, so the convolution and the shortcut term run in float32 and the
  result is rounded once to the input dtype. The default, `True`, keeps each
  backend's own precision: `subq_ops_fused` and 1D non-causal `subq_ops` run in
  the input dtype, and the others convolve in float32 and round before adding
  the shortcut.
- `QKVSequenceMixer` hands an inner mixer that sets `takes_fused_qkv = True` the
  un-split `[B, *spatial, 3C]` projection (`TriangleHyena` opts in); mixers
  without the attribute still receive `(q, k, v, cp_group)`. Under
  `torch.compile` `TriangleHyena` runs its short conv as a fused stencil.

### Fixed

- **Kernels changed at their training length once the grid cache had grown
  twice.** The SIREN and Random Fourier positional embeddings (and the SIREN
  variants built on them) grow their coordinate grid when asked for a length
  beyond `L_cache`. Only the first extension kept the original grid step:
  growing the cache again, or growing another axis afterwards, rescaled the
  axes that had already grown. `SIRENKernelND` and `RandomFourierKernelND`
  then returned a different kernel at every length, including `L_cache`, and
  so did the `CKConvND` layers and Gaussian or exponential windows built on
  them. This affected every release since 0.1.0. Fixed-size training never
  grows the cache twice and was not affected. Every extension now keeps the
  original step, and results that were already correct are bit-identical.
- **The positional embeddings' first layer ran in bf16 under autocast.**
  `SIRENPositionalEmbeddingND` (and `MultiOmegaSIRENPositionalEmbeddingND`),
  `LearnableOmegaSIRENPositionalEmbeddingND` and
  `RandomFourierPositionalEmbeddingND` are meant to compute their first-layer
  projection `W x + b` in float32, but `F.linear` runs in bf16 under
  `torch.autocast` even on float32 operands. With float32 parameters under
  autocast (the usual `bf16-mixed` setup), the coordinates and the phase (of
  order `2π·ω₀` radians or more) were rounded to bf16. On long axes,
  neighbouring grid points merged: up to 32 taps shared one coordinate at
  `L_cache=8192`. At `ω₀=10` the SIREN embeddings were off by 3–6% (relative
  L2) and random Fourier features by 9–18%, and the kernel inherited that
  error. With bf16 parameters, all but the learnable-ω₀ embedding rounded the
  phase to bf16 before the sine. On CUDA, Inductor turns a single-coordinate
  (1D) projection into float32 elementwise math, so compiled 1D was not
  affected; eager 1D and both modes in 2D and 3D were. The first layer is now
  a `torch.nn.Linear` subclass that computes the projection as float32
  multiply-adds (float64 for float64 parameters), which neither autocast nor
  TF32 lowers, and the sine or cosine runs in the same dtype. Eager and
  compiled now agree in every dimension. Parameters and state dicts are
  unchanged, and forward hooks on `positional_embedding.linear` still fire.
  Embeddings, kernels and models pickled whole before this fix
  (`torch.save(model)` rather than a state dict) get the new first layer when
  they are unpickled.
  Kernels computed from existing checkpoints change by the removed rounding
  error under autocast (compiled 1D on CUDA stays the same to float32
  rounding), with bf16 or fp16 first-layer parameters (for example from a
  parameter-only cast), and in float32 runs with TF32 matmuls enabled
  (`torch.set_float32_matmul_precision("high")`, which `experiments/run.py`
  sets). The embedding is now returned in the weight dtype (float32 under
  `bf16-mixed`) instead of bf16, which costs some memory in eager mode;
  `torch.compile` fuses the first layer into one kernel.
- **`CKConvND` with `fft_backend="torch_fft"` rejected bf16/fp16 activations.**
  Its float32 `shortcut` parameter tripped the `torch_fft` ops'
  `shortcut.dtype == x.dtype` assertion unless the parameters were cast as
  well. `CKConvND` now casts the shortcut to the activations' dtype itself (a
  differentiable cast, so the parameter stays float32 and receives the
  gradient), as the `subq_ops` wrappers already did; results are otherwise
  unchanged bit for bit. (Under `python -O`, which strips that assertion,
  such activations used to come back as float32 with the shortcut term
  promoted; they now keep their dtype.) The `hyenand-retrofit` asset no
  longer needs its `CKConvND` subclass, and the ViT-5 benchmark no longer
  patches the FFT functions.
- **`GaussianModulationND` computed its exponent in bf16 under autocast.** Its
  docstring promises a float32 window, but the exponent
  `-0.5 Σ_d (p_d / σ_{d,c})²` came from an `einsum`, which autocast runs as a
  bf16 matmul (and which can run in TF32 without autocast), and the
  exponential was then taken of the rounded result. With float32 parameters
  under bf16 autocast on an H100, the window was off by 0.2–0.3% (relative
  L2), by up to 4% where it is above 1e-3, and the `std_param` gradient by
  0.4–1.4%. The exponent is now a sum of float32 multiply-adds over the axes
  and the exponential runs in float32, so autocast and TF32 no longer change
  the window. In eager mode each axis is one pass over a float32 tensor of
  the kernel's size; `torch.compile` fuses them into one kernel.

### Changed

- The `[cuda]` extra now requires `subquadratic-ops-torch-cu13>=0.3.1`; only the
  latest subquadratic-ops-torch release is supported.

## \[0.2.0\] - 2026-09-16

### Added

- **`fft_backend="subq_ops_fused"` on `CKConvND`**, backed by
  `subquadratic_ops_torch.fused_fft_conv2d`. It runs the whole
  rfft2 → multiply → irfft2 pipeline in a single cuFFTDx launch and, unlike
  every other FFT path, **natively in fp32/fp16/bf16** instead of upcasting to
  fp32. Measured on an H100 at batch 8, hidden 768, forward+backward, bf16:

  | spatial | FFT tile | vs `torch_fft` | vs `subq_ops` |
  | ------- | -------- | -------------- | ------------- |
  | 16      | 32       | 1.9×           | 1.4×          |
  | 32      | 64       | 4.4×           | 2.5×          |
  | 64      | 128      | 4.9×           | 1.3×          |

  The margin over `torch_fft` grows with spatial extent; the margin over
  `subq_ops` does not vary monotonically. Speedups are shape- and
  hardware-dependent — reproduce with `benchmarks/ops/bench_fused_fftconv2d.py`
  rather than assuming these figures transfer to another GPU or shape.

  Restricted to `data_dim=2`, `is_causal=False`, `fft_padding="zero"`, and
  spatial extents of at most **64 per axis** — the kernel's largest FFT tile is
  128 and it requires `max(X, Y) <= fft_size // 2`. The spatial cap is enforced
  on the first forward pass rather than at construction, because the input size
  is not known when the module is built. Per-sample (FiLM) kernels are
  supported. Requires `subquadratic-ops-torch >= 0.3.0`. Note the spatial cap is
  not the only hardware constraint: extents above 32 per axis resolve to the 128
  FFT tile, which needs more shared memory than SM80/SM86 provide, so those
  shapes require compute capability 9.0+ and raise a clear error below it.
  CI validates extents up to 32 per axis (the 64 FFT tile) on SM86 hardware; the
  64-per-axis path (128 tile, SM90+) is covered only by its arch guard and has
  not yet been exercised on Hopper/Blackwell.

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
  supports the required FFT tile — the 128 tile needs SM90+). It additionally
  declines whenever rewriting would change what the graph observes: when a
  spectrum or the multiply result is consumed outside the chain, when the
  kernel's dtype or device differs from the input's, when the chain's output
  dtype or device would not match, on a non-default FFT `norm`, on an in-place
  `mul_` ordered after the `irfft2`, and on symbolic slice bounds under dynamic
  shapes. Because every intermediate is verified private to the chain before
  the rewrite, the replaced nodes are erased individually rather than by
  whole-graph dead-code elimination, which cannot distinguish an unrelated
  in-place op from a dead one. `lowering_stats()` reports rewrite and
  per-reason skip counts, since a silent pass is otherwise hard to tell apart
  from one that never fired.

- **Hierarchical patch merging** (#141). `nvsubquadratic.modules.patch_merging`
  adds `PatchMerging` (1D token sequences with an optional register row),
  `PatchMerging2D` and `PatchEmbedHierarchical`; `ViT5HierarchicalNet` (with
  `StageSpec`) and `ViT5HierarchicalClassificationNet` build multi-stage,
  progressively-downsampled ViT5 variants on top of them. Ships with experiment
  recipes under `examples/patch_merging/cifar10/` and
  `examples/vit5_imagenet/{v5_patchmerge,v6_hierarchical,local_comparison}/`,
  plus two CIFAR-10 datamodules — `experiments/datamodules/cifar10.py`
  (torchvision) and `cifar10_hf.py` (HuggingFace, with `MixupConfig` /
  `AugmentConfig`).

- **3D motion spatial-recall dataset** (#140).
  `SpatialRecall3DMotionDataset` and `SpatialRecall3DMotionDataModule` in
  `experiments/datamodules/spatial_recall_dataset.py` sweep a digit stamp
  through a 3D volume along a bounded, monotone path, with optional quarter-turn
  spin per depth slice and a reserved readout cube. Example configs under
  `examples/spatial_recall_v2/motion_3d/`.

### Fixed

- **`SubqOpsCausalConv1d` could not run under `torch.autocast`.** The fused
  kernel picks its specialisation from the *input* dtype and then requires every
  tensor to match it exactly, so an autocast region — where activations arrive
  as bf16/fp16 while parameters stay fp32 — raised
  `ValueError: in_w expected dtype (code=4, bits=16) but got (code=2, bits=32)`.
  The weight and bias are now narrowed to the input dtype at call time, mirroring
  what autocast does for the built-in conv ops. The fp32 master parameters are
  untouched; only the values handed to the kernel are cast.

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

- The `[quack]` extra now installs `quack-kernels[cu13]`, selecting
  `nvidia-cutlass-dsl`'s CUDA 13 runtime libraries to match this release. Install
  it with `pip`, not `uv` — quack's README documents a `uv` ordering race on its
  cu13 path ([NVIDIA/cutlass#3259](https://github.com/NVIDIA/cutlass/issues/3259)).

### Notes

- **Upgrading an existing GPU environment from 0.1.1 requires uninstalling the
  CUDA 12 packages first.** `subquadratic-ops-torch-cu12` and
  `subquadratic-ops-torch-cu13` both install a top-level `subquadratic_ops_torch/`
  package, and `nvidia-dali-cuda120` / `nvidia-dali-cuda130` both install
  `nvidia/`. A plain `pip install -U` therefore overwrites files without removing
  the old distribution, and later uninstalling either one leaves the other
  broken. Run:

  ```bash
  pip uninstall -y subquadratic-ops-torch-cu12 nvidia-dali-cuda120
  pip install -U "nvsubquadratic[cuda]"
  ```

  A fresh environment needs none of this. Note also that CUDA 13 requires an
  NVIDIA driver >= 580 on the host (or the CUDA forward-compatibility package).

- A plain `pip install nvsubquadratic[cuda]` resolves entirely from public PyPI:
  torch 2.14.0 on PyPI is already a CUDA 13.0 build. The
  `--index-url https://download.pytorch.org/whl/cu130` route is only needed for
  environments that compile Apex/mamba against a specific torch build (the
  Dockerfile and the SLURM/enroot scripts).

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
