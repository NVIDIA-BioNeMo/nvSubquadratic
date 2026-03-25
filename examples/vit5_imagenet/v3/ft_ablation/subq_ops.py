"""FiLM finetuning — custom subquadratic_ops CUDA FFT backend.

Same hyperparameters as baseline but uses the custom fftconv2d kernels
from subquadratic_ops_torch instead of torch.fft.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(fft_backend="subq_ops")
