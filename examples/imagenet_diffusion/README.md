# ImageNet Diffusion Experiments

Class-conditional ImageNet generation using CCNN (Hyena + SIREN kernels) with the JiT flow-matching framework.

## Experiment matrix

| Config | Resolution | Model size | hidden | blocks | patch | omega_0 | Gated | FiLM | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `ccnn_jit_baseline` | 64x64 | JiT-B | 768 | 12 | 4 | 10.0 | no | no | Existing baseline |
| `jit_baseline` | 64x64 | JiT-B | 768 | 12 | 4 | — | no | no | Reference JiT transformer |
| `ccnn_jit_128` | 128x128 | JiT-B | 768 | 12 | 8 | 10.0 | no | no | Higher res, same token count |
| `ccnn_jit_256` | 256x256 | JiT-B | 768 | 12 | 16 | 10.0 | no | no | Higher res, same token count |
| `ccnn_jit_128_low_omega` | 128x128 | JiT-B | 768 | 12 | 8 | **3.0** | no | no | Reduced omega_0 |
| `ccnn_jit_256_low_omega` | 256x256 | JiT-B | 768 | 12 | 16 | **3.0** | no | no | Reduced omega_0 |
| `ccnn_jit_L_256` | 256x256 | JiT-L | 1024 | 24 | 16 | 10.0 | no | no | Larger model |
| `ccnn_jit_128_gated_film_ema` | 128x128 | JiT-B | 768 | 12 | 8 | 10.0 | yes | yes | Gated Hyena + FiLM (untested) |

## Design decisions

### Resolution scaling

All configs keep the token count at 16x16 = 256 tokens to match the 64px baseline:

- 64px / patch 4 = 16x16
- 128px / patch 8 = 16x16
- 256px / patch 16 = 16x16

This means the patch embedding carries more information at higher resolutions (larger patches) while the sequence model operates on the same length. Batch sizes and gradient accumulation are adjusted to maintain ~1024 effective batch size.

### Low omega_0 experiments

The `_low_omega` variants reduce `omega_0` from 10.0 to 3.0 on the first SIREN layer. This controls the initial frequency range of the implicit convolution kernel — a lower value biases the kernel toward lower spatial frequencies, which should help suppress the high-frequency artifacts observed in prior runs. `hidden_omega_0` remains at 1.0.

### JiT-L model size

`ccnn_jit_L_256` matches the JiT-L transformer architecture:
- 1024 hidden dim (vs 768 for JiT-B)
- 24 blocks (vs 12)
- Kernel MLP scaled to 64 hidden / 64 embedding (vs 32/32)
- Lower learning rate (1e-4 vs 2e-4)
- Higher EMA decay (0.9999 vs 0.9998)

### Gated FiLM experiment

`ccnn_jit_128_gated_film_ema` adapts the classification architecture from `vit5_small_pretrain_hyena_cls_row_gated_film_ema` to diffusion:

- **Gated Hyena**: dual nonlinearity (SiLU + Sigmoid gates)
- **FiLM-conditioned SIREN**: the timestep condition vector (collapsed to `[B, hidden_dim]` by AdaLN) is forwarded through the mixer chain to modulate SIREN hidden layers via learned (gamma, beta) pairs
- **EMA**: decay 0.9998

In the classification setup, FiLM conditioning comes from register tokens pooled via `RegisterPooling`. In diffusion there are no registers — instead the AdaLN timestep condition serves as the FiLM input. A one-line change in `AdaLNZeroResidualBlock.forward` (passing `conditioning=cond` to the sequence mixer) wires this up. This change is safe for all non-FiLM configs since the kwarg flows through `**mixer_kwargs` and is only consumed when a `film_generator` exists on the SIREN kernel. **This path is untested end-to-end.**

## Shared settings (all configs)

| Parameter | Value |
|---|---|
| Training timesteps | 1000 |
| Inference steps (Heun) | 50 |
| CFG guidance scale | 2.9 |
| CFG interval | [0.1, 1.0] |
| Condition dropout | 0.1 |
| Flow-matching p_mean / p_std | -0.8 / 0.8 |
| Optimizer | Adam (betas 0.9, 0.95) |
| Scheduler | Constant with 2% warmup |
| Gradient clip | 1.0 |
| Classes | 1000 (ImageNet) |
| Training iterations | 250k |

## Running

```bash
# Snellius (SLURM)
sbatch slurm/diffusion/ccnn_jit_128_snellius.sh

# Local / interactive (single node, 4 GPUs)
PYTHONPATH=. python experiments/run.py \
    --config examples/imagenet_diffusion/ccnn_jit_128.py \
    experiment_dir=runs/ccnn_jit_128

# Override omega_0 at launch time (any config)
PYTHONPATH=. python experiments/run.py \
    --config examples/imagenet_diffusion/ccnn_jit_128.py \
    experiment_dir=runs/ccnn_jit_128_omega5 \
    net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0=5.0
```

## SLURM scripts

- `slurm/diffusion/ccnn_jit_128_snellius.sh` — template for 128px on Snellius H100 nodes. Copy and edit `CONFIG_FILE` / `EXPERIMENT_NAME` for other configs.
- `slurm/diffusion/ccnn_jit_diff_baseline_snellius.sh` — existing 64px CCNN baseline.
- `slurm/diffusion/jit_diff_baseline_snellius.sh` — existing 64px JiT transformer baseline.
