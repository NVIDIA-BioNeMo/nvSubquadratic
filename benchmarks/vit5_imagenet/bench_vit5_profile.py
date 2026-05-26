"""Per-phase profiling of a single ViT-5-Small forward+backward step.

Instruments forward, attention/mixer, MLP, backward, and optimizer
phases with CUDA-synced timers and reports their share of the step.
Use this to diagnose a regression: if total step time goes up, this
script shows which phase moved.

Targets: H100 SXM 80GB, BF16, batch size 256.

Usage:
    PYTHONPATH=. conda run -n nv-subq python \\
        benchmarks/vit5_imagenet/bench_vit5_profile.py

Output: stdout phase-breakdown table.
"""

import sys

import torch
import torch.nn.functional as F


sys.path.insert(0, ".")

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


HIDDEN_DIM = 384
NUM_BLOCKS = 12
NUM_HEADS = 6
PATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_REGISTERS = 4
NUM_PATCHES_H = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES_W = IMAGE_SIZE // PATCH_SIZE
BATCH_SIZE = 256


def build_model():
    """Build ViT-5-Small classification model on CUDA in bfloat16."""
    net = ViT5ClassificationNet(
        in_channels=3,
        num_classes=1000,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_patches_h=NUM_PATCHES_H,
                num_patches_w=NUM_PATCHES_W,
                num_registers=NUM_REGISTERS,
                qk_norm=LazyConfig(RMSNorm)(dim=64, eps=1e-6),
                rope_base=10000.0,
                reg_rope_base=100.0,
                attn_dropout=0.0,
                proj_dropout=0.0,
                qkv_bias=False,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=4.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=1e-4,
            drop_path_rate=0.05,
        ),
    )
    return net.cuda().to(torch.bfloat16)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, device="cuda", dtype=torch.bfloat16)
    inp = {"input": x, "condition": None}
    target = torch.randint(0, 1000, (BATCH_SIZE,), device="cuda")

    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(inp)
            loss = F.cross_entropy(out["logits"], target)
            loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(inp)
            loss = F.cross_entropy(out["logits"], target)
            loss.backward()
        torch.cuda.synchronize()

    print("=== TOP 30 CUDA kernels by total CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n=== TOP 30 by CPU time (shows Python overhead) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
