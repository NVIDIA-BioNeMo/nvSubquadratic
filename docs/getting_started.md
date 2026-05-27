# Getting Started

This page walks a new user from a fresh checkout to a working Hyena
forward pass.  For the full installation matrix (dev container, Docker,
Apptainer, conda, venv) see the project [README](https://github.com/NVIDIA-Digital-Bio/nvSubquadratic-private/blob/main/README.md).

## Requirements

- CUDA-compatible NVIDIA GPU (Ampere or Hopper architecture)
- CUDA Toolkit 12.0 or higher
- Python 3.11 or higher

The optional fused RMSNorm kernel (`quack-kernels`) requires Hopper or
Blackwell (H100, B200, B300); on Ampere the library falls back to a
pure-PyTorch path automatically.

## Install

The recommended developer setup is conda:

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
the [project README](https://github.com/NVIDIA-Digital-Bio/nvSubquadratic-private/blob/main/README.md#installation).

## Hello, Hyena

A minimal forward pass through a 2D Hyena mixer:

```python
import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import (
    SIRENKernelND,
    SIRENPositionalEmbeddingND,
)
from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl

device = torch.device("cuda")

B, H, X, Y = 2, 64, 32, 32
x = torch.randn(B, H, X, Y, device=device)

# A SIREN-parameterised long-range 2D kernel.
kernel_cfg = LazyConfig(SIRENKernelND)(
    out_dim=H,
    data_dim=2,
    mlp_hidden_dim=64,
    num_layers=3,
    embedding_dim=32,
    omega_0=10.0,
    L_cache=max(X, Y),
    use_bias=True,
)

# Wire a Hyena mixer that consumes the kernel via a global FFT conv.
mixer_cfg = LazyConfig(Hyena)(
    global_conv_cfg=LazyConfig(lambda: None)(),  # replaced below
    short_conv_cfg=LazyConfig(torch.nn.Identity)(),
    gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
    pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),
    qk_norm_cfg=None,
)

# For a self-contained example, skip the LazyConfig dance and call the
# op directly:
kernel = torch.randn(1, H, X, Y, device=device)
y = fftconv2d_fp32_bhl(x, kernel)
print(y.shape)  # torch.Size([2, 64, 32, 32])
```

The lower-level FFT ops in
{doc}`nvsubquadratic.ops <api_reference/ops>` are deliberately
function-only so higher-level mixers can compose them freely.  The
{doc}`nvsubquadratic.modules <api_reference/modules>` package wraps them
in `nn.Module`-shaped mixers (Hyena, Mamba, Attention, CKConv), and
{doc}`experiments <api_reference/experiments>` wires those mixers into
Lightning-driven training pipelines.

## Next steps

- {doc}`architecture` — the three-layer nvSubquadratic / subquadratic-ops
  / megatron-core story and the naming conventions used throughout the
  library.
- {doc}`examples/index` — end-to-end training recipes per dataset.
- {doc}`api_reference/index` — the full curated API surface.
