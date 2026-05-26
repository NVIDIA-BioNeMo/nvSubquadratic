# Package overview

`nvsubquadratic` is organised bottom-up: function-only convolution
primitives in `ops/`, then the `nn.Module`-shaped mixers and blocks
that compose them in `modules/`, then full architectures in
`networks/`.  The sibling `experiments/` package is the training
driver — it consumes a `networks/` config via
{class}`~nvsubquadratic.lazy_config.LazyConfig` and runs it through
PyTorch Lightning.

## Layout

```text
nvsubquadratic/
├── lazy_config.py                deferred-instantiation system (LazyConfig, instantiate)
├── ops/                          function-only convolution primitives
│   ├── fftconv.py                fp32 reference FFT conv (1D / 2D / 3D, linear)
│   ├── fftconv_fp16.py           half-precision linear-conv variants
│   ├── circular_fftconv.py       fp32 periodic-boundary FFT conv
│   ├── circular_fftconv_fp16.py  half-precision periodic variants
│   ├── fftconv_multihead.py      multi-head + low-rank factorisations
│   ├── fftconv_chunked.py        peak-FFT-memory chunking helpers
│   ├── fftconv_custom.py         subquadratic_ops_torch CUDA wrappers
│   ├── causal_conv1d_custom.py   direct (non-FFT) CUDA causal conv1d wrappers
│   └── mixed_fftconv.py          per-axis mixed boundary-condition FFT conv
├── modules/                      nn.Module building blocks
│   ├── hyena_nd.py               Hyena ND mixer (two-gate sandwich, CP)
│   ├── mamba_nd.py               Mamba SSM (ND, selective, raster scan)
│   ├── attention.py              multi-head attention (RoPE, ND)
│   ├── vit5_attention.py         ViT-5 register-aware attention
│   ├── vit5_hyena_adapter.py     Hyena drop-in for ViT-5
│   ├── sequence_mixer.py         operator-agnostic QKV dispatch
│   ├── condition_mixer.py        cross-attention conditioning mixer
│   ├── kernels_nd.py             SIREN / RFF kernels (multi-ω₀, block-diag)
│   ├── ckconv_nd.py              CKConv ND (implicit k_θ(p))
│   ├── ckconv_multihead_nd.py    multi-head CKConv (low-rank)
│   ├── distributed_depthwise_conv_nd.py   CP-aware depthwise convs
│   ├── causal_conv1d.py          left-only-padded Conv1d wrapper
│   ├── subq_ops_causal_conv1d.py nn.Conv1d-compatible CUDA depthwise
│   ├── residual_block.py         pre-norm + mixer + MLP (+ AdaLN-Zero)
│   ├── vit5_residual_block.py    ViT-5 residual block (LayerScale, registers)
│   ├── patchify.py               strided patch embedding / unpatchify
│   ├── position_encoding.py      axis-factorised learned PE
│   ├── masks_nd.py               exponential / Gaussian / block-aligned masks
│   ├── mlp.py                    GELU / SwiGLU / GLU MLP
│   ├── film.py                   FiLM kernel generator + register pooling
│   ├── grn.py                    GlobalResponseNorm (ConvNeXt V2)
│   ├── layer_scale.py            LayerScale γ·F(x)
│   ├── drop_path.py              stochastic depth
│   ├── rms_norm.py               RMSNorm + PerHeadRMSNorm
│   ├── rms_norm_channel_first.py channel-first RMSNorm
│   └── schedulers.py             ResumableSequentialLR
├── networks/                     end-to-end architectures
│   ├── general_purpose_resnet.py ResidualNetwork (LazyConfig stack)
│   ├── classification_resnet.py  GAP-readout classification head
│   ├── vit5_classification.py    ViT-5 hybrid Hyena/attention backbone
│   ├── jit.py                    JiT diffusion backbone
│   ├── jit_utils.py              JiT helpers (RoPE, RMSNorm, sin-cos PE)
│   ├── huggingface_diffusers.py  HF DiT / UVit adapters
│   └── baselines/
│       ├── unet_convnext.py      Well UNet-ConvNeXt baseline
│       └── unet_convnext_v2.py   …with fixed finest-skip
├── parallel/                     context-parallel primitives
│   ├── a2a_comms.py              AllToAllSingle (autograd-aware)
│   └── utils.py                  init_parallel_state + zigzag split/gather
├── utils/                        cross-area utilities
│   ├── init.py                   weight-init factories
│   ├── qk_norm.py                QK normalization (apply + L2Norm)
│   ├── rope.py                   rotary position embedding (1D / 2D / 3D)
│   └── quack_utils.py            QuACK capability probe
├── metrics/
│   └── cleanfid.py               FID via cleanfid
└── testing/
    └── utils.py                  compute_relative_error
```

## What each area does

**`ops/`** — Function-only convolution primitives.  Linear and circular
FFT convolutions in 1D / 2D / 3D, in both pure-PyTorch reference
(`fp32`) and CUDA-accelerated paths via :mod:`subquadratic_ops_torch`.
The `fp16` variants use dual mean-centering for numerical stability
under cuFFT's 65 504 limit (see :doc:`ops/FP16_FFTCONV_DERIVATION`).
Multi-head and low-rank factorisations, chunked variants that bound
peak FFT memory, and a per-axis mixed-boundary-condition family round
out the surface.  Naming encodes the contract:
`<op>_<precision>_<layout>[_w_reshape][_chunked]` where `bhl` is
channels-first and `blh` is channels-last.

**`modules/`** — `nn.Module`-shaped building blocks.  Sequence/spatial
mixers (Hyena, Mamba, attention, ViT-5 variants) share a common QKV
signature that
{class}`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`
dispatches over.  Learned kernels
({class}`~nvsubquadratic.modules.kernels_nd.SIRENKernelND` and its
multi-ω₀ / learnable-ω₀ / block-diagonal variants, random-Fourier
parametrisations) feed the global convs.  Residual blocks (pre-norm +
mixer + MLP, optional AdaLN-Zero), norms (RMSNorm, GRN, LayerScale),
gating/conditioning (FiLM, DropPath, condition mixer), embeddings
(Patchify, position encoding, masks), and the MLP block round it out.

**`networks/`** — End-to-end architectures.
{class}`~nvsubquadratic.networks.general_purpose_resnet.ResidualNetwork`
composes `modules/` blocks under a `LazyConfig` stack;
{class}`~nvsubquadratic.networks.classification_resnet.ClassificationResNet`
adds a GAP-readout head.  The ViT-5 family is the main hybrid
Hyena/attention vision backbone.  The JiT backbone and the Hugging
Face diffusers adapters
({class}`~nvsubquadratic.networks.huggingface_diffusers.DiffusersDiTWrapper`,
{class}`~nvsubquadratic.networks.huggingface_diffusers.DiffusersUVitWrapper`)
cover diffusion.  The Well UNet-ConvNeXt baselines provide PDE-task
reference points.

**`parallel/`** — Context-parallel primitives the Hyena global
convolution and context-parallel attention rely on:
{class}`~nvsubquadratic.parallel.a2a_comms.AllToAllSingleFunction`
(autograd-aware all-to-all that swaps sequence and channel sharding)
and zigzag split / gather in
{func}`~nvsubquadratic.parallel.utils.zigzag_split_across_group_ranks`
that balance causal load across CP ranks.  `init_parallel_state`
wires up Megatron's parallel state on top of NCCL.

**`utils/`, `metrics/`, `testing/`, `lazy_config.py`** — Supporting
machinery.  {class}`~nvsubquadratic.lazy_config.LazyConfig` builds an
unevaluated constructor-call spec;
{func}`~nvsubquadratic.lazy_config.instantiate` walks the tree and
constructs the live object graph (this is what every example config
ultimately calls).  `utils/` houses weight-init factories
({func}`~nvsubquadratic.utils.init.trunc_normal_init`,
`small_init`, `wang_init`), QK-norm
({func}`~nvsubquadratic.utils.qk_norm.apply_qk_norm`), rotary
embeddings, and the QuACK capability probe.  `metrics/` wraps
`cleanfid` for FID; `testing/` provides numerical-comparison
helpers used by the test suite.

## Where to go next

- {doc}`api_reference/index` — the curated API for every area above.
- {doc}`examples/index` — per-dataset training recipes that compose
  these networks.
- {doc}`ops/README` — math primer for the FFT convolution primitives.
