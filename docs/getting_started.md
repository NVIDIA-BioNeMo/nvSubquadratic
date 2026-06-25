# Getting Started

This page walks a new user from a fresh checkout to a working Hyena
forward pass.  For the full installation matrix (dev container, Docker,
Apptainer, conda, venv) see the project [README](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/README.md).

## Requirements

- CUDA-compatible NVIDIA GPU
- CUDA Toolkit 12.0 or higher
- Python 3.11 or higher

The optional fused RMSNorm kernel (`quack-kernels`) requires Hopper or
Blackwell (H100, B200, B300); on Ampere the library falls back to a
pure-PyTorch path automatically.

## Install

For users who just want to use the library:

```bash
pip install nvsubquadratic
```

For contributors or developer setup, the recommended path is conda:

```bash
bash setup_conda_env.sh
conda activate nvsubquadratic
```

This creates an environment with Python 3.12 and PyTorch 2.10 (CUDA
12.9), installs the dev dependencies, builds NVIDIA Apex from source,
and installs `quack-kernels`.

For an alternative venv-based install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements-dev.txt
pip install --no-build-isolation -e .
```

Docker, Apptainer, enroot/SLURM, and dev-container instructions live in
the [project README](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/README.md#installation).

## Hello, Hyena

A minimal forward pass through a real 2D Hyena mixer.  Everything is
wired with {doc}`LazyConfig <lazy_config>`: each `LazyConfig(Cls)(...)`
records the class and its arguments without constructing anything, and a
single `instantiate(...)` call at the end builds the whole tree.  This is
exactly how the {doc}`experiments <api_reference/experiments>` configs
assemble their networks, except there the scalar fields are filled by
`"${net.hidden_dim}"`-style interpolation instead of the concrete
integers used below.

```python
import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.rms_norm_channel_first import RMSNormChannelFirst
from nvsubquadratic.utils.qk_norm import L2Norm

device = torch.device("cuda")

B, H, X, Y = 2, 64, 32, 32

hyena_cfg = LazyConfig(Hyena)(
    # The long-range global convolution.  CKConvND owns a SIREN-parameterised
    # kernel and applies it as an FFT conv — the kernel is *generated* by the
    # SIREN MLP, never random.
    global_conv_cfg=LazyConfig(CKConvND)(
        data_dim=2,
        hidden_dim=H,
        kernel_cfg=LazyConfig(SIRENKernelND)(
            data_dim=2,
            out_dim=H,
            mlp_hidden_dim=32,
            num_layers=3,
            embedding_dim=32,
            omega_0=10.0,
            hidden_omega_0=1.0,
            L_cache=max(X, Y),
            use_bias=True,
        ),
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type="double",  # linear (non-circular) convolution
        fft_padding="zero",
        fft_backend="torch_fft",  # portable; "subq_ops" uses the fused 2D CUDA kernel
        is_causal=False,
    ),
    # Depthwise short conv on the concatenated [Q; K; V] (3 * H channels).
    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
        in_channels=3 * H,
        out_channels=3 * H,
        kernel_size=3,
        groups=3 * H,
        padding=1,
        bias=False,
    ),
    gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),  # first gate σ
    gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),  # second gate σ₂
    pixelhyena_norm_cfg=LazyConfig(RMSNormChannelFirst)(
        dim=H, eps=1e-6, use_quack=False
    ),
    output_norm_cfg=LazyConfig(RMSNormChannelFirst)(dim=H, eps=1e-6, use_quack=False),
    qk_norm_cfg=LazyConfig(L2Norm)(dim=1),  # L2 QK-norm on the channel axis
)

# Build the whole module tree in one call.
hyena = instantiate(hyena_cfg).to(device)

# Hyena consumes channels-last Q, K, V tensors [B, *spatial, C].  In a full
# model these come from a linear projection W·x (see QKVSequenceMixer); here
# we feed random activations to exercise the forward.
q = torch.randn(B, X, Y, H, device=device)
k = torch.randn(B, X, Y, H, device=device)
v = torch.randn(B, X, Y, H, device=device)

y = hyena(q, k, v)
print(y.shape)  # torch.Size([2, 32, 32, 64])  -> [B, X, Y, C]
```

In a real network you rarely hold `Q`, `K`, `V` yourself: the
`QKVSequenceMixer` (see
[`mixer_defaults.py`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/spatial_recall_v2/mixer_defaults.py))
projects a single activation `x` into the three tensors and forwards them
to this `Hyena`.  That same factory is what the spatial-recall experiments
instantiate.

## Going lower: the FFT conv op directly

The Hyena above ultimately routes its long-range mixing through one of the
FFT-convolution ops.  When you only need the convolution itself, with no
gating, kernel generation, or `nn.Module`, you can call the op
directly.  Here `kernel` is supplied explicitly (any 2D filter; a SIREN
kernel would normally produce it):

```python
import torch

from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl

device = torch.device("cuda")

B, H, X, Y = 2, 64, 32, 32
x = torch.randn(B, H, X, Y, device=device)  # channels-first [B, C, X, Y]
kernel = torch.randn(1, H, X, Y, device=device)  # per-channel filter [1, C, K_x, K_y]

y = fftconv2d_fp32_bhl(x, kernel)  # "same"-size circular-free conv
print(y.shape)  # torch.Size([2, 64, 32, 32])
```

The op casts to fp32 internally for numerical stability and returns the
result in `x`'s original dtype.  Note the layout difference: the ops work
**channels-first** `[B, C, *spatial]` (the `_bhl` suffix), whereas the
`Hyena` module's public interface is **channels-last** `[B, *spatial, C]`.

The lower-level FFT ops in
{doc}`nvsubquadratic.ops <api_reference/ops>` are deliberately
function-only so higher-level mixers can compose them freely.  The
{doc}`nvsubquadratic.modules <api_reference/modules>` package wraps them
in `nn.Module`-shaped mixers (Hyena, Mamba, Attention, CKConv), and
{doc}`experiments <api_reference/experiments>` wires those mixers into
Lightning-driven training pipelines.

## Next steps

- {doc}`architecture`: the three layers (nvSubquadratic, subquadratic-ops,
  megatron-core) and the naming conventions used throughout the
  library.
- [`examples/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples):
  end-to-end training recipes per dataset.
- {doc}`api_reference/index`: the full curated API surface.
