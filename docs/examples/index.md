# Examples

Each subdirectory of [`examples/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples)
is a self-contained training recipe.  Recipes are
{class}`nvsubquadratic.lazy_config.LazyConfig` trees describing the
network, datamodule, Lightning wrapper, and trainer; running them is
`python -m experiments.run --config <path>`.

The active experimental roadmap (priorities, owners, status) lives at
[`examples/overview_tracker.md`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/overview_tracker.md).

## Classification

### MNIST / SMNIST

[`examples/mnist_classification/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/mnist_classification)
covers MNIST with both attention and Hyena baselines, plus a small CCNN
backbone.  [`examples/smnist_classification/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/smnist_classification)
covers sequential MNIST (1D input).

### ImageNet

[`examples/imagenet_classification/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/imagenet_classification)
ships seven CCNN configs (Hyena / Hyena-circular / attention, with and
without augmentation, plus tiny variants for laptop sanity checks).
Representative entry points: `ccnn_7_512_hyena.py`,
`ccnn_7_512_attention.py`.

### TinyImageNet — ViT-5

[`examples/vit5_imagenet/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/vit5_imagenet)
is the ViT-5 baseline suite (v1–v5) with its own
[`TRACKER.md`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/vit5_imagenet/TRACKER.md).

### UCF101

[`examples/ucf101_classification/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/ucf101_classification)
covers video classification with both sequence- and clip-mode datamodules.

## Diffusion

### MNIST

[`examples/mnist_diffusion/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/mnist_diffusion)
is a small DDPM/JiT diffusion sanity-check.

### ImageNet

[`examples/imagenet_diffusion/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/imagenet_diffusion)
is the full ImageNet diffusion setup.  See its
[README](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/imagenet_diffusion/README.md)
for the JiT vs Hyena-vs-attention comparison.

## Spatial recall

[`examples/spatial_recall_1d/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/spatial_recall_1d),
[`spatial_recall_2d/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/spatial_recall_2d),
[`spatial_recall_3d/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/spatial_recall_3d),
and the newer
[`spatial_recall_v2/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/spatial_recall_v2)
are synthetic recall benchmarks measuring how well each mixer (Hyena,
attention, Mamba, CKConv) routes information across long-range
spatial/sequence positions.  See
[`spatial_recall_v2/TRACKER.md`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/spatial_recall_v2/TRACKER.md)
for the v2 task suite.

## Benchmarks

[`examples/vit_b_benchmark/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/vit_b_benchmark)
holds the throughput-comparison configs used to produce the numbers in
{doc}`../benchmarks`.

## Scientific

### The Well

[`examples/well/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/well)
covers The Well PDE benchmark — see its
[README](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/well/README.md)
for sub-datasets and baselines.
