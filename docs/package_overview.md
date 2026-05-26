# Package overview

`nvsubquadratic` is organised bottom-up: low-level convolution
primitives in `ops/`, then the `nn.Module`-shaped mixers and blocks
that compose them in `modules/`, then full architectures in
`networks/`.  The sibling `experiments/` package is the training
driver — it consumes a `networks/` config via
:class:`~nvsubquadratic.lazy_config.LazyConfig` and runs it through
PyTorch Lightning.

## `ops/`

Function-only convolution primitives.  Linear and circular FFT
convolutions in 1D / 2D / 3D, in both pure-PyTorch reference (`fp32`)
and CUDA-accelerated paths via :mod:`subquadratic_ops_torch`.  The
`fp16` variants use dual mean-centering for numerical stability under
cuFFT's 65 504 limit (see :doc:`ops/FP16_FFTCONV_DERIVATION`).
Multi-head and low-rank factorisations, chunked variants that bound
peak FFT memory, and a per-axis mixed-boundary-condition family for
mixed periodic / non-periodic data round out the surface.  Naming
encodes the contract: `<op>_<precision>_<layout>[_w_reshape][_chunked]`
where `bhl` is channels-first and `blh` is channels-last.

## `modules/`

`nn.Module`-shaped building blocks.  Sequence/spatial mixers
(:class:`~nvsubquadratic.modules.hyena_nd.Hyena`,
:class:`~nvsubquadratic.modules.mamba_nd.Mamba`,
:class:`~nvsubquadratic.modules.attention.Attention`, the ViT-5
attention/Hyena adapters) with a common QKV signature that
:class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`
dispatches over.  Learned kernels
(:class:`~nvsubquadratic.modules.kernels_nd.SIRENKernelND` and its
multi-ω₀ / learnable-ω₀ / block-diagonal variants, random-Fourier
parametrisations) feed the global convs.  Convolutions
(:class:`~nvsubquadratic.modules.causal_conv1d.CausalConv1D`, the
distributed depthwise variants, CKConv N-D) sit alongside.  Plus
residual blocks (pre-norm + mixer + MLP, optional AdaLN-Zero), norms
(:class:`~nvsubquadratic.modules.rms_norm.RMSNorm`, channel-first
variants, :class:`~nvsubquadratic.modules.grn.GlobalResponseNorm`,
:class:`~nvsubquadratic.modules.layer_scale.LayerScale`),
gating/conditioning (:class:`~nvsubquadratic.modules.film.KernelFiLMGenerator`,
:class:`~nvsubquadratic.modules.drop_path.DropPath`,
:class:`~nvsubquadratic.modules.condition_mixer.QKVConditionMixer`),
embeddings (:class:`~nvsubquadratic.modules.patchify.Patchify`,
:class:`~nvsubquadratic.modules.position_encoding.PositionEmbeddingND`,
:class:`~nvsubquadratic.modules.masks_nd.GaussianModulationND`), and
:class:`~nvsubquadratic.modules.mlp.MLP` (GELU / SwiGLU / GLU).

## `networks/`

End-to-end architectures.
:class:`~nvsubquadratic.networks.general_purpose_resnet.ResidualNetwork`
and :class:`~nvsubquadratic.networks.classification_resnet.ClassificationResNet`
compose `modules/` blocks under a `LazyConfig` stack.  The ViT-5
family (:class:`~nvsubquadratic.networks.vit5_classification.ViT5ClassificationNet`)
is the main hybrid Hyena/attention vision backbone; the JiT diffusion
backbone (:class:`~nvsubquadratic.networks.jit.JiT` plus its
factory functions) and the Hugging Face diffusers adapters
(:class:`~nvsubquadratic.networks.huggingface_diffusers.DiffusersDiTWrapper`,
:class:`~nvsubquadratic.networks.huggingface_diffusers.DiffusersUVitWrapper`)
cover the diffusion path.  The Well UNet-ConvNeXt baselines provide
the PDE-task reference points.

## `parallel/`

Context-parallel primitives that the Hyena global convolution and
context-parallel attention rely on:
:class:`~nvsubquadratic.parallel.a2a_comms.AllToAllSingleFunction`
(autograd-aware all-to-all that swaps sequence and channel sharding)
and the zigzag split / gather helpers in
:func:`~nvsubquadratic.parallel.utils.zigzag_split_across_group_ranks`
that balance causal load across CP ranks.  `init_parallel_state` wires
up Megatron's parallel state on top of NCCL.

## `utils/`, `metrics/`, `testing/`, `lazy_config.py`

Supporting machinery.  :mod:`nvsubquadratic.lazy_config` is the
deferred-instantiation system every example config uses
(:class:`~nvsubquadratic.lazy_config.LazyConfig` builds an unevaluated
spec; :func:`~nvsubquadratic.lazy_config.instantiate` walks it).
:mod:`nvsubquadratic.utils` houses weight-init factories
(`trunc_normal_init`, `small_init`, `wang_init`), QK-norm and rotary
position embeddings, and the QuACK capability probe.
:mod:`nvsubquadratic.metrics` wraps `cleanfid` for FID computation.
:mod:`nvsubquadratic.testing` provides numerical-comparison helpers
consumed by the test suite.

## Where to go next

- {doc}`api_reference/index` — the curated API for every area above.
- {doc}`examples/index` — per-dataset training recipes that compose
  these networks.
- :doc:`ops/README` — math primer for the FFT convolution primitives.
