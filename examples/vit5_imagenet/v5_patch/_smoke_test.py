"""Smoke test: instantiate every v5_patch config, build the net, run one forward pass on GPU."""

import importlib
import sys
import time

import torch

from nvsubquadratic.lazy_config import instantiate


CONFIGS = [
    "examples.vit5_imagenet.v5_patch.hyena_patch16",
    "examples.vit5_imagenet.v5_patch.hyena_patch8",
    "examples.vit5_imagenet.v5_patch.hyena_patch4",
    "examples.vit5_imagenet.v5_patch.hyena_patch2",
    "examples.vit5_imagenet.v5_patch.hyena_patch1",
    "examples.vit5_imagenet.v5_patch.attention_patch16",
    "examples.vit5_imagenet.v5_patch.attention_patch8",
    "examples.vit5_imagenet.v5_patch.attention_patch4",
    "examples.vit5_imagenet.v5_patch.attention_patch2",
    "examples.vit5_imagenet.v5_patch.attention_patch1",
]

BATCH_SIZE = 1  # minimal batch for smoke test
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
    print(f"  num_registers: {net.num_registers}")
    print(f"  prepend_registers: {net.prepend_registers}")
    has_pad = hasattr(net, "reg_zero_pad") and net.reg_zero_pad is not None
    if has_pad:
        print(f"  reg_zero_pad shape: {net.reg_zero_pad.shape}")
    else:
        print("  reg_zero_pad: None")

    x = {
        "input": torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, device=DEVICE),
        "condition": None,
    }

    net.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        t0 = time.perf_counter()
        out = net(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    logits = out["logits"]
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  logits shape: {logits.shape}")
    print(f"  forward time: {dt:.3f}s")
    print(f"  peak GPU mem: {peak_mb:.0f} MB")
    print("  PASS")

    del net, x, out
    torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    passed, failed = [], []

    for cfg in CONFIGS:
        try:
            smoke_test_config(cfg)
            passed.append(cfg.rsplit(".", 1)[-1])
        except Exception as e:
            name = cfg.rsplit(".", 1)[-1]
            print(f"\n  FAIL: {name} — {e}")
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
