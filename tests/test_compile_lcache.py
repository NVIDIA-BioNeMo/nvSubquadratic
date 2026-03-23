"""Test: does torch.compile work when L_cache=14 and input requires 15 rows?

Simulates the real pipeline:
1. Build model with L_cache=14 and num_registers=14
2. Compile with max-autotune-no-cudagraphs
3. Run forward on a 224x224 image (which gives 15x14 tokens after register prepend)
"""

import os
import sys

import torch


sys.path.insert(0, ".")

for env_path in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
    "/home/dwromero/projects/nvSubquadratic-private/.env",
]:
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")
        break

from nvsubquadratic.lazy_config import instantiate  # noqa: E402


def build_net():
    from examples.vit5_imagenet.v3.gap_film_regs._base import get_config

    cfg = get_config(
        num_registers=14,
        num_film_layers=3,
        film_after_pos_embed=True,
        reg_init="zeros",
        train_do=False,
    )
    return instantiate(cfg.net), cfg


print("Building model (L_cache=14, num_registers=14)...")
net, cfg = build_net()
net.eval()

# Check L_cache before compile
for i, block in enumerate(net.blocks):
    siren = block.sequence_mixer.inner_mixer.mixer.global_conv.kernel
    lc = siren.positional_embedding.L_cache
    print(f"  block {i}: L_cache = {lc}")

# Enable compile-compatible FFT
import nvsubquadratic.ops.fftconv as _fftconv  # noqa: E402


_fftconv.COMPILE_COMPATIBLE = True

print("\nCompiling with max-autotune-no-cudagraphs...")
net_compiled = torch.compile(net, mode="max-autotune-no-cudagraphs")

print("Running compiled forward (this triggers L_cache auto-extension)...")
dummy = {"input": torch.randn(1, 224, 224, 3)}
try:
    with torch.no_grad():
        out = net_compiled(dummy)
    print(f"  SUCCESS! logits shape: {out['logits'].shape}")
    print(f"  logits[:5]: {out['logits'][0, :5]}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Check L_cache after forward
print("\nL_cache after forward:")
for i, block in enumerate(net.blocks):
    siren = block.sequence_mixer.inner_mixer.mixer.global_conv.kernel
    lc = siren.positional_embedding.L_cache
    print(f"  block {i}: L_cache = {lc}")

print("\nRunning second compiled forward (should be stable now)...")
try:
    with torch.no_grad():
        out2 = net_compiled(dummy)
    print(f"  SUCCESS! logits shape: {out2['logits'].shape}")
    diff = (out2["logits"] - out["logits"]).abs().max().item()
    print(f"  logit diff vs first pass: {diff:.4e}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
