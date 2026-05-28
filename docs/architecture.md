# Architecture

nvSubquadratic is one layer in a three-library stack.  Each layer has a
narrow responsibility; they compose via stable interfaces so that a
research team can swap any one of them without touching the others.

```
                    ┌───────────────────────────┐
                    │      experiments/         │     (training driver)
                    │   PyTorch Lightning       │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │      nvsubquadratic        │   ← this library
                    │  (PyTorch API: mixers,    │     (API & ergonomics)
                    │   networks, datamodules)  │
                    └─────┬─────────────┬───────┘
                          │             │
            ┌─────────────┘             └───────────────┐
            ▼                                           ▼
 ┌───────────────────────┐                 ┌──────────────────────────┐
 │  subquadratic-ops     │                 │      megatron-core       │
 │  (fused CUDA kernels) │                 │  (TP/PP/CP parallelism)  │
 │  Causal Conv1D /      │                 │  initialised by          │
 │  FFT Conv 1D/2D /     │                 │  parallel/utils.py       │
 │  B2B Causal Conv1D    │                 │                          │
 └───────────────────────┘                 └──────────────────────────┘
```

## What each layer owns

- **nvsubquadratic** (this library) — the PyTorch-native API.
  Sequence/spatial mixers (Hyena, Mamba, attention variants), learned
  kernels, residual blocks, networks, and the datamodule/wrapper
  scaffolding consumed by `experiments`.  All public surface lives in
  the {doc}`api_reference/index`.
- **subquadratic-ops** (separate repo) — the fused CUDA kernels.  Causal
  Conv1D for short kernels (2–256), FFT-based Causal Conv1D for long
  kernels (up to 8K–16M), B2B Causal Conv1D for striped Hyena
  architectures, plus the 1D/2D FFT primitives.  nvSubquadratic
  delegates here via {mod}`subquadratic_ops_torch` and the published
  docs are at
  <https://nvidia-bionemo.github.io/subquadraticOps-docs/>.
- **megatron-core** — Megatron's distributed-training primitives
  (tensor / pipeline / context parallelism).  nvSubquadratic uses it via
  {mod}`nvsubquadratic.parallel.utils`'s
  `init_parallel_state` and the context-parallel
  `DistributedDepthwiseConvNd` wrappers.

This layering keeps API ergonomics in nvSubquadratic, kernel
optimisation in subquadratic-ops, and distributed bookkeeping in
megatron-core.  Practically: if a kernel is slow, fix it in
subquadratic-ops; if an interface is awkward, fix it here.

## Naming conventions

Two conventions show up everywhere in the ops and module code.  Both
are documented in detail in `docs/ops/README.md`; the short version:

- **`BHL` vs `BLH`** — memory layout.  `BHL` is channels-first
  (`[B, H, *spatial]`, matches `torch.nn.ConvNd`); `BLH` is
  channels-last (`[B, *spatial, H]`, common in transformer code).  The
  FFT runs faster on contiguous spatial axes, so BHL is the fast path.
- **`_w_reshape`** — wrappers that accept BLH input, internally reshape
  to BHL, run the fast path, and reshape back.  Recommended entry point
  for channels-last callers.
- **`_chunked`** — processes channels in groups to cap peak FFT memory.
- **`fp32` vs `fp16`** — internal compute precision.  fp16 ops require
  power-of-2 spatial dims (cuFFT constraint) and use dual mean-centering
  for numerical stability — see
  [FP16 Circular FFT Convolution: Derivation](ops/FP16_FFTCONV_DERIVATION.md).

So `causal_fftconv1d_fp32_bhl_w_reshape` is a causal 1D FFT conv that
accepts channels-last input, runs the fp32 channels-first kernel under
the hood, and returns channels-last output.

## Operator-agnostic dispatch

Hyena, attention, CKConv, and Mamba all expose the same
`(query, key, value)` mixer signature.  The dispatch lives in
{class}`nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`:
configure it with a `LazyConfig` over any of the mixers and the rest of
the model code is unchanged.  Switching architectures is a one-line
config diff.

## The lazy-instantiation system

Every config file under `examples/` is a tree of
{class}`nvsubquadratic.lazy_config.LazyConfig` specs.  A spec is a
deferred constructor call: `LazyConfig(SomeClass)(arg=...)` returns a
proxy that holds the target and the arguments, but doesn't instantiate
yet.  {func}`nvsubquadratic.lazy_config.instantiate` walks the tree,
resolves nested specs depth-first, evaluates arithmetic strings (for
`L_cache = "max(H, W)"`-style expressions), and constructs the live
object graph.

This is why most module constructors take `*_cfg: LazyConfig` rather
than concrete instances — the user-facing config file decides what to
instantiate; the module never imports a specific norm/conv/scheduler
class directly.

## Further reading

- {doc}`api_reference/index` — the curated API.
- `docs/ops/README.md` — math primer for the FFT-based convolution ops.
- `docs-tracker.md` (repo root) — current docstring coverage status.
