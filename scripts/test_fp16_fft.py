#!/usr/bin/env python
"""Compare f32 vs fp16 FFT convolution on a trained Hyena-GAP checkpoint.

Loads a trained ViT-5 Hyena-GAP model, runs inference twice (once with f32 FFT,
once with fp16 FFT via CKConvND.use_fp16_fft), and compares logits, accuracy,
and throughput.

Usage (inside nv-subq env):
    PYTHONPATH=. python scripts/test_fp16_fft.py \
        --checkpoint <path/to/checkpoint.ckpt> \
        --num-batches 50 --batch-size 64
"""

import argparse
import importlib
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from nvsubquadratic.modules.ckconv_nd import CKConvND


def load_config():
    """Load the v2 Hyena GAP config."""
    spec = importlib.util.spec_from_file_location(
        "config",
        "examples/vit5_imagenet/v2/vit5_small_pretrain_hyena_gap_apex.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def build_network(cfg):
    """Instantiate the network from the config."""
    from nvsubquadratic.lazy_config import instantiate
    return instantiate(cfg.net)


def load_checkpoint(net, ckpt_path):
    """Load a Lightning checkpoint into the raw network."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        key = k
        for prefix in ("network._orig_mod.", "network.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        new_sd[key] = v
    net.load_state_dict(new_sd, strict=True)
    return net


def set_fp16_fft(net, enabled: bool):
    """Toggle use_fp16_fft on all CKConvND modules."""
    from nvsubquadratic.ops.fftconv import fftconv2d_bhl, fftconv2d_bhl_w_reshape
    from nvsubquadratic.ops.fftconv_fp16 import fftconv2d_fp16_bhl, fftconv2d_fp16_bhl_w_reshape

    count = 0
    for module in net.modules():
        if isinstance(module, CKConvND):
            module.use_fp16_fft = enabled
            if enabled:
                module.fftconv_fn = fftconv2d_fp16_bhl_w_reshape
                module.fftconv_fn_bhl_input = fftconv2d_fp16_bhl
            else:
                module.fftconv_fn = fftconv2d_bhl_w_reshape
                module.fftconv_fn_bhl_input = fftconv2d_bhl
            count += 1
    return count


@torch.no_grad()
def run_inference(net, batches, device="cuda"):
    """Run inference on a list of batches, return logits and timing."""
    net.eval()
    all_logits = []

    # Warmup
    _ = net(batches[0])
    torch.cuda.synchronize()

    start = time.perf_counter()
    for batch in batches:
        out = net(batch)
        all_logits.append(out["logits"].float().cpu())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return torch.cat(all_logits, dim=0), elapsed


def main():
    parser = argparse.ArgumentParser(description="Compare f32 vs fp16 FFT convolution")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batches: {args.num_batches} x {args.batch_size}")
    print()

    # Build and load model
    print("Loading config and building network...")
    cfg = load_config()
    net = build_network(cfg)
    print("Loading checkpoint...")
    net = load_checkpoint(net, args.checkpoint)
    net = net.to(device).eval()

    # Generate reproducible dummy batches
    torch.manual_seed(42)
    batches = [
        {"input": torch.randn(args.batch_size, 224, 224, 3, device=device),
         "condition": torch.zeros(args.batch_size, dtype=torch.long, device=device)}
        for _ in range(args.num_batches)
    ]

    # ─── Run with f32 FFT (baseline) ─────────────────────────────────────
    print("\n=== F32 FFT (baseline) ===")
    set_fp16_fft(net, enabled=False)
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_f32, time_f32 = run_inference(net, batches, device)
    mem_f32 = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Time: {time_f32:.3f}s ({args.num_batches / time_f32:.1f} batches/s)")
    print(f"  Peak memory: {mem_f32:.0f} MB")
    print(f"  Logits range: [{logits_f32.min():.4f}, {logits_f32.max():.4f}]")

    # ─── Run with fp16 FFT ───────────────────────────────────────────────
    print("\n=== FP16 FFT ===")
    n = set_fp16_fft(net, enabled=True)
    print(f"  Enabled fp16 FFT on {n} CKConvND modules")
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_fp16, time_fp16 = run_inference(net, batches, device)
    mem_fp16 = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Time: {time_fp16:.3f}s ({args.num_batches / time_fp16:.1f} batches/s)")
    print(f"  Peak memory: {mem_fp16:.0f} MB")
    print(f"  Logits range: [{logits_fp16.min():.4f}, {logits_fp16.max():.4f}]")

    # ─── Compare ─────────────────────────────────────────────────────────
    print("\n=== Comparison ===")
    abs_diff = (logits_f32 - logits_fp16).abs()
    print(f"  Max absolute diff:  {abs_diff.max():.6f}")
    print(f"  Mean absolute diff: {abs_diff.mean():.6f}")

    pred_f32 = logits_f32.argmax(dim=-1)
    pred_fp16 = logits_fp16.argmax(dim=-1)
    match_rate = (pred_f32 == pred_fp16).float().mean()
    print(f"  Top-1 prediction match: {match_rate:.4f} ({(match_rate * 100):.1f}%)")

    cos_sim = F.cosine_similarity(logits_f32, logits_fp16, dim=-1).mean()
    print(f"  Cosine similarity: {cos_sim:.6f}")

    speedup = time_f32 / time_fp16
    print(f"\n  Speed: {speedup:.3f}x ({'faster' if speedup > 1 else 'slower'})")
    print(f"  Memory: {mem_fp16:.0f} vs {mem_f32:.0f} MB ({(1-mem_fp16/mem_f32)*100:+.1f}%)")


if __name__ == "__main__":
    main()
