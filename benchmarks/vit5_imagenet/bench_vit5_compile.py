"""Profile + test optimizations for ViT-5-Small."""

import sys
import torch
import torch.nn.functional as F
import time

sys.path.insert(0, ".")

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.modules.mlp import MLP
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
    net = ViT5ClassificationNet(
        in_channels=3,
        num_classes=1000,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
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


def benchmark(model, batch_size, label, num_warmup=10, num_iters=50):
    x = torch.randn(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3, device="cuda", dtype=torch.bfloat16)
    inp = {"input": x, "condition": None}
    target = torch.randint(0, 1000, (batch_size,), device="cuda")

    for _ in range(num_warmup):
        out = model(inp)
        loss = F.cross_entropy(out["logits"], target)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        out = model(inp)
        loss = F.cross_entropy(out["logits"], target)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_step = elapsed / num_iters * 1000
    samples_per_sec = batch_size * num_iters / elapsed
    print(f"  [{label}] Time/step: {ms_per_step:.1f} ms | Throughput: {samples_per_sec:.0f} samples/sec")
    return ms_per_step, samples_per_sec


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Baseline (current code, eager)")
    print("=" * 60)
    model = build_model()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        benchmark(model, BATCH_SIZE, "baseline-eager")
    del model
    torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("Test 2: torch.compile (default)")
    print("=" * 60)
    model = build_model()
    model = torch.compile(model)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        benchmark(model, BATCH_SIZE, "torch.compile", num_warmup=20, num_iters=50)
    del model
    torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("Test 3: torch.compile (max-autotune)")
    print("=" * 60)
    model = build_model()
    model = torch.compile(model, mode="max-autotune")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        benchmark(model, BATCH_SIZE, "compile-max-autotune", num_warmup=20, num_iters=50)
    del model
    torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("Test 4: Profiling one step with torch.profiler")
    print("=" * 60)
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

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
