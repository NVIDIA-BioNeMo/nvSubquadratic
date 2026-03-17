"""Benchmark optimized ViT-5-Small: correctness check + throughput measurement."""

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
    # --- Correctness checks ---
    print("=" * 60)
    print("Correctness checks")
    print("=" * 60)

    # 1. Basic forward pass shape
    model = build_model()
    x = torch.randn(2, IMAGE_SIZE, IMAGE_SIZE, 3, device="cuda", dtype=torch.bfloat16)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model({"input": x, "condition": None})
    assert out["logits"].shape == (2, 1000), f"Expected (2, 1000), got {out['logits'].shape}"
    print("  [PASS] Forward pass shape correct")

    # 2. Gradient flow
    x = torch.randn(2, IMAGE_SIZE, IMAGE_SIZE, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model({"input": x, "condition": None})
        out["logits"].sum().backward()
    assert x.grad is not None, "No gradient on input"
    print("  [PASS] Gradient flow works")

    # 3. Deterministic eval
    model.eval()
    x = torch.randn(2, IMAGE_SIZE, IMAGE_SIZE, 3, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out1 = model({"input": x.clone(), "condition": None})
        out2 = model({"input": x.clone(), "condition": None})
    assert torch.allclose(out1["logits"], out2["logits"]), "Non-deterministic in eval mode"
    print("  [PASS] Deterministic in eval mode")

    # 4. RoPE buffers are registered
    attn = model.blocks[0].sequence_mixer
    assert hasattr(attn, "rope_cos"), "Missing rope_cos buffer"
    assert hasattr(attn, "rope_sin"), "Missing rope_sin buffer"
    T = 1 + NUM_PATCHES_H * NUM_PATCHES_W + NUM_REGISTERS
    assert attn.rope_cos.shape == (T, attn.head_dim), f"Bad rope_cos shape: {attn.rope_cos.shape}"
    print(f"  [PASS] RoPE buffers shape: ({T}, {attn.head_dim})")

    # 5. CLS token has identity RoPE (cos=1, sin=0)
    ones = torch.ones(attn.head_dim, device=attn.rope_cos.device, dtype=attn.rope_cos.dtype)
    zeros = torch.zeros(attn.head_dim, device=attn.rope_sin.device, dtype=attn.rope_sin.dtype)
    assert torch.allclose(attn.rope_cos[0], ones)
    assert torch.allclose(attn.rope_sin[0], zeros)
    print("  [PASS] CLS token has identity RoPE")

    # 6. Zero registers
    attn_no_reg = (
        ViT5Attention(
            hidden_dim=384,
            num_heads=6,
            num_patches_h=14,
            num_patches_w=14,
            num_registers=0,
            qk_norm=LazyConfig(RMSNorm)(dim=64, eps=1e-6),
        )
        .cuda()
        .to(torch.bfloat16)
    )
    x_no_reg = torch.randn(2, 1 + 196, 384, device="cuda", dtype=torch.bfloat16)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_no_reg = attn_no_reg(x_no_reg)
    assert out_no_reg.shape == (2, 197, 384)
    print("  [PASS] Zero-register variant works")

    del model
    torch.cuda.empty_cache()

    # --- Throughput benchmarks ---
    print()
    print("=" * 60)
    print(f"Throughput benchmarks (batch_size={BATCH_SIZE})")
    print("=" * 60)

    print("\n--- Eager mode ---")
    model = build_model()
    model.train()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        benchmark(model, BATCH_SIZE, "eager")
    del model
    torch.cuda.empty_cache()

    print("\n--- torch.compile (default) ---")
    model = build_model()
    model.train()
    model = torch.compile(model)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        benchmark(model, BATCH_SIZE, "compile-default", num_warmup=20)
    del model
    torch.cuda.empty_cache()

    print("\n--- torch.compile (max-autotune) ---")
    model = build_model()
    model.train()
    model = torch.compile(model, mode="max-autotune")
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            benchmark(model, BATCH_SIZE, "compile-max-autotune", num_warmup=20)
    except Exception as e:
        print(f"  [FAILED] max-autotune: {e}")
    del model
    torch.cuda.empty_cache()
