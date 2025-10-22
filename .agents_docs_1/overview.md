# Overview
- **Goal**: Understand the `nvSubquadratic` codebase and design a plan to port the simple diffusion pipeline from `../ccnn_v2`.
- **Stakeholders**: davidknigge (requester), codex-agent (executor).
- **Constraints**: Operate per Autonomous Agent Playbook, maintain documentation artefacts, avoid seeking approvals (`approval_policy=never`), keep work reproducible.

# Environment
- Conda environment `nvsq` (Python 3.11) created locally at `/home/davidknigge/anaconda3/envs/nvsq`. Activate via `conda activate nvsq` before installing project dependencies.

# Key Artifacts to Inspect
- `README.md`, `pyproject.toml`, `nvsubquadratic/` package sources, `examples/`, and `tests/`.
- Compare with diffusion implementation in `../ccnn_v2`.

# Architecture Notes
- Core modules live under `nvsubquadratic/modules`: residual blocks wire `QKVSequenceMixer` (self-attn/Hyena backends) with configurable MLPs, norms, and dropouts via `LazyConfig`.
- `nvsubquadratic/lazy_config.py` supplies Hydra-like lazy instantiation utilities used throughout configs and examples.
- FFT and depthwise convolution helpers live under `nvsubquadratic/ops` and `nvsubquadratic/modules/distributed_depthwise_conv_nd.py`, leveraging `subquadratic-ops` CUDA kernels.
- Example workflows (`examples/`) rely on Lightning wrappers and dataclass configs to assemble datasets, networks, and optimizers.
- Tests validate package importability, self-attention behavior, and distributed conv ops; `pyproject.toml` sets Python >=3.11 with PyTorch/Lightning, Megatron-Core, and subquadratic dependencies.
- New diffusion pathway introduces `nvsubquadratic/networks/diffusion_resnet.DiffusionResNet`, `examples/lightning_wrappers.DiffusionWrapper`, and Hugging Face-backed ImageNet datamodule/config under `examples/imagenet_diffusion/`, preserving shared cache usage.

# Reference Implementation (ccnn_v2)
- Diffusion entrypoint: `experiments/imagenet/simple_diffusion2.py` configures ImageNet datamodule, `DiffusionResNet`, and Lightning `DiffusionWrapper`.
- Network: `modern_ccnn.networks.diffusion_resnet.DiffusionResNet` builds CKConv residual blocks with timestep conditioning and positional encodings.
- Training loop: `modern_ccnn.lightning_wrappers.DiffusionWrapper` implements DDPM training/validation/test, schedule buffers, sampling, and WandB logging.

# Open Questions
- Preferred runtime environment and dependencies for this project.
- Integration points where a diffusion pipeline would fit within existing abstractions.
- Which dataset wrappers/examples should host diffusion runs (ImageNet vs other) and what adjustments are needed for WandB/logging parity.
