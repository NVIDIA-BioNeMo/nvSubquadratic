"""Smoke test: instantiate channel-first RMSNorm configs, build nets, run forward + backward on GPU."""

import importlib
import sys
import time

import torch

from nvsubquadratic.lazy_config import instantiate


CONFIGS = [
    "examples.vit5_imagenet.v5.hyena_gap_pretrain_cf_norm",
]

BATCH_SIZE = 2
IMAGE_SIZE = 224
DEVICE = "cuda"


def smoke_test_config(module_path: str) -> None:
    name = module_path.rsplit(".", 1)[-1]
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    mod = importlib.import_module(module_path)
    config = mod.get_config()
    net = instantiate(config.net).to(DEVICE)

    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  params: {num_params:.1f}M")

    # torch.compile like the training pipeline
    compile_mode = getattr(config, "compile_mode", "max-autotune")
    print(f"  compiling with mode={compile_mode} ...")
    net = torch.compile(net, mode=compile_mode)

    x = {
        "input": torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, device=DEVICE),
        "condition": None,
    }

    # Warm-up compiled forward (triggers JIT)
    net.train()
    print("  warm-up forward (JIT compile) ...")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        out = net(x)
    out["logits"].sum().backward()
    net.zero_grad()
    torch.cuda.synchronize()
    print("  warm-up done")

    # Timed forward + backward
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        t0 = time.perf_counter()
        out = net(x)
        torch.cuda.synchronize()
        dt_fwd = time.perf_counter() - t0

    logits = out["logits"]
    print(f"  logits shape: {logits.shape}")
    print(f"  forward time: {dt_fwd:.3f}s")

    loss = logits.sum()
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    dt_bwd = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  backward time: {dt_bwd:.3f}s")
    print(f"  peak GPU mem: {peak_mb:.0f} MB")

    grad_ok = all(p.grad is not None for p in net._orig_mod.parameters() if p.requires_grad)
    print(f"  all grads present: {grad_ok}")
    print("  PASS" if grad_ok else "  FAIL (missing grads)")

    del net, x, out, loss
    torch.cuda.empty_cache()
    return grad_ok


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    passed, failed = [], []

    for cfg in CONFIGS:
        try:
            ok = smoke_test_config(cfg)
            (passed if ok else failed).append(cfg.rsplit(".", 1)[-1])
        except Exception as e:
            name = cfg.rsplit(".", 1)[-1]
            print(f"\n  FAIL: {name} — {e}")
            import traceback

            traceback.print_exc()
            failed.append(name)
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
