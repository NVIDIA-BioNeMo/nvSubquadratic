"""Profile the training pipeline to identify the wall-clock bottleneck.

Measures data loading, forward, backward, and optimizer step independently
to show where time is actually spent in the training loop.

Usage:
    PYTHONPATH=. python scripts/profile_training_bottleneck.py --eager --num-workers 14
    PYTHONPATH=. python scripts/profile_training_bottleneck.py --eager --dali --num-workers 14
    PYTHONPATH=. python scripts/profile_training_bottleneck.py --num-workers 14          # compiled
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from experiments.datamodules.imagenet import AugmentConfig, ImageNetDataModule, MixupConfig
from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet

os.environ.setdefault("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
os.environ.setdefault("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

try:
    from apex.optimizers import FusedLAMB as Lamb
    OPTIMIZER_NAME = "Apex FusedLAMB"
except ImportError:
    from torch_optimizer import Lamb
    OPTIMIZER_NAME = "torch_optimizer.Lamb"

# ── Constants ────────────────────────────────────────────────────────────────

BATCH_SIZE = 256
NUM_STEPS = 50
HIDDEN_DIM = 384
NUM_HEADS = 6
NUM_BLOCKS = 12
NUM_REGISTERS = 4
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = IMAGE_SIZE // PATCH_SIZE

AUGMENT_CFG = AugmentConfig(use_three_augment=True, color_jitter=0.3)
MIXUP_CFG = MixupConfig(mixup=0.8, cutmix=1.0, mixup_prob=1.0,
                         mixup_switch_prob=0.5, smoothing=0.0)

# ── Builders ─────────────────────────────────────────────────────────────────


def build_model():
    net_cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=3, num_classes=1000, hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE,
        num_registers=NUM_REGISTERS, dropout_rate=0.0,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                num_patches_h=NUM_PATCHES, num_patches_w=NUM_PATCHES,
                num_registers=NUM_REGISTERS,
                qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // NUM_HEADS, eps=1e-6),
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM, activation="gelu", expansion_factor=4.0,
                dropout_cfg=LazyConfig(nn.Dropout)(p=0.0),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM, layer_scale_init=1e-4, drop_path_rate=0.05,
        ),
    )
    return instantiate(net_cfg)


def build_cpu_dataloader(prefetch_factor: int = 2, num_workers: int = 14):
    """Standard torchvision CPU dataloader."""
    dm = ImageNetDataModule(
        data_dir=os.environ["IMAGENET_PATH"],
        imagefolder_dir=os.environ.get("IMAGENET_FOLDER_PATH"),
        prefetch_factor=prefetch_factor,
        batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True, seed=42,
        image_size=IMAGE_SIZE, final_image_size=IMAGE_SIZE,
        center_crop=True, num_classes=1000, drop_labels=False,
        hf_dataset_name="ILSVRC/imagenet-1k", hf_dataset_config=None,
        hf_auth_token=os.environ.get("HF_TOKEN"), task="classification",
        mixup_cfg=MIXUP_CFG, augment_cfg=AUGMENT_CFG,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm.train_dataloader(), dm


def build_dali_dataloader(prefetch_factor: int = 2, num_workers: int = 14, optimized: str = ""):
    """NVIDIA DALI GPU-pipelined dataloader.

    Args:
        optimized: "" for original DALI, "v2" for optimised, "v3" for v3.
    """
    common = dict(
        data_dir=os.environ["IMAGENET_PATH"],
        imagefolder_dir=os.environ.get("IMAGENET_FOLDER_PATH"),
        prefetch_factor=prefetch_factor,
        batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True, seed=42,
        image_size=IMAGE_SIZE, final_image_size=IMAGE_SIZE,
        num_classes=1000, drop_labels=False, task="classification",
        augment_cfg=AUGMENT_CFG, device_id=0,
    )
    if optimized == "v3":
        from experiments.datamodules.dali_imagenet_optimized_v3 import DALIImageNetOptimizedV3DataModule
        dm = DALIImageNetOptimizedV3DataModule(**common)
    elif optimized == "v2":
        from experiments.datamodules.dali_imagenet_optimized import DALIImageNetOptimizedDataModule
        dm = DALIImageNetOptimizedDataModule(**common)
    else:
        from experiments.datamodules.dali_imagenet import DALIImageNetDataModule
        dm = DALIImageNetDataModule(**common)
    dm.setup("fit")
    return dm.train_dataloader(), dm


def _build_loader(use_dali: bool, optimized: str = "", **kwargs):
    if use_dali:
        return build_dali_dataloader(optimized=optimized, **kwargs)
    return build_cpu_dataloader(**kwargs)


# ── Benchmarking helpers ─────────────────────────────────────────────────────


def bench_dataloader(prefetch_factor, num_workers, use_dali=False, optimized=False, warmup=5):
    """Benchmark pure data loading for a single (prefetch, workers) combo."""
    loader, dm = _build_loader(use_dali, optimized=optimized, prefetch_factor=prefetch_factor, num_workers=num_workers)
    it = iter(loader)
    for _ in range(warmup):
        next(it)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_STEPS):
        next(it)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del loader, it
    ms = elapsed / NUM_STEPS * 1000
    tput = BATCH_SIZE * NUM_STEPS / elapsed
    return ms, tput


def prepare_batch(raw_batch, dm, device, use_dali):
    """Convert a raw dataloader batch into the dict the model expects."""
    if use_dali:
        return dm.on_before_batch_transfer(raw_batch, 0)
    images, labels = raw_batch
    images = images.permute(0, 2, 3, 1).contiguous().to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return {"input": images, "label": labels, "condition": None}


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args):
    device = torch.device("cuda")
    mode_str = "eager" if args.eager else "compiled"
    nw = args.num_workers
    use_dali = args.dali or args.dali_optimized or args.dali_v3
    if args.dali_v3:
        dali_opt_str = "v3"
    elif args.dali_optimized:
        dali_opt_str = "v2"
    else:
        dali_opt_str = ""
    if dali_opt_str == "v3":
        decode_str = "DALI-v3 (GPU pipelined, bf16)"
    elif dali_opt_str == "v2":
        decode_str = "DALI-optimized (GPU pipelined)"
    elif use_dali:
        decode_str = "DALI (GPU pipelined)"
    else:
        decode_str = "CPU (PIL)"
    gpu_name = torch.cuda.get_device_name()
    node = os.environ.get("SLURMD_NODENAME", "unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Device: {gpu_name} (node={node})")
    print(f"Mode: {mode_str.upper()}")
    print(f"Decode: {decode_str}")
    print(f"Optimizer: {OPTIMIZER_NAME}")
    print(f"Batch size: {BATCH_SIZE}, Workers: {nw}")
    print(f"Timestamp: {timestamp}")
    print()

    # ── 0. Prefetch factor sweep ─────────────────────────────────────
    print("=" * 60)
    print("PHASE 0: Prefetch factor sweep (data loading only)")
    print("=" * 60)
    prefetch_factors = [2, 4, 8, 16]
    pf_results = {}
    best_pf, best_ms = 2, float("inf")
    for pf in prefetch_factors:
        ms, tput = bench_dataloader(pf, nw, use_dali=use_dali, optimized=dali_opt_str)
        pf_results[pf] = {"ms": round(ms, 1), "tput": round(tput)}
        print(f"  prefetch_factor={pf:>2d}: {ms:6.1f}ms/batch, {tput:>5.0f} samples/sec")
        if ms < best_ms:
            best_ms, best_pf = ms, pf
    print(f"  >> Best: prefetch_factor={best_pf} ({best_ms:.1f}ms)")
    print()

    # ── 1. Data loading speed ────────────────────────────────────────
    print("=" * 60)
    print(f"PHASE 1: Data loading speed (prefetch_factor={best_pf})")
    print("=" * 60)
    data_ms, data_tput = bench_dataloader(best_pf, nw, use_dali=use_dali, optimized=dali_opt_str, warmup=5)
    print(f"  Per batch: {data_ms:.1f}ms")
    print(f"  Throughput: {data_tput:.0f} samples/sec")
    print()

    # ── 2. Forward + backward (synthetic) ────────────────────────────
    print("=" * 60)
    print(f"PHASE 2: Forward + backward (synthetic, mode={mode_str})")
    print("=" * 60)
    model = build_model().to(device)
    if not args.eager:
        print("  Compiling with max-autotune...")
        model = torch.compile(model, mode="max-autotune")
    loss_fn = nn.CrossEntropyLoss()

    fake_img = torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, device=device)
    fake_lbl = torch.randint(0, 1000, (BATCH_SIZE,), device=device)
    fake_batch = {"input": fake_img, "label": fake_lbl, "condition": None}

    warmup_iters = 5 if args.eager else 20
    print(f"  Warming up ({warmup_iters} iters)...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup_iters):
            out = model(fake_batch)["logits"]
            loss = loss_fn(out, fake_lbl)
            loss.backward()
            model.zero_grad()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(NUM_STEPS):
            out = model(fake_batch)["logits"]
            loss = loss_fn(out, fake_lbl)
            loss.backward()
            model.zero_grad()
    end_ev.record()
    torch.cuda.synchronize()

    compute_ms_total = start_ev.elapsed_time(end_ev)
    compute_ms = compute_ms_total / NUM_STEPS
    compute_tput = BATCH_SIZE * NUM_STEPS / (compute_ms_total / 1000)
    print(f"  Per step: {compute_ms:.1f}ms")
    print(f"  Throughput: {compute_tput:.0f} samples/sec")
    print()

    # ── 3. Optimizer step ────────────────────────────────────────────
    print("=" * 60)
    print(f"PHASE 3: Optimizer step ({OPTIMIZER_NAME})")
    print("=" * 60)
    optimizer = Lamb(model.parameters(), lr=4e-3, weight_decay=0.05)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(fake_batch)["logits"]
        loss = loss_fn(out, fake_lbl)
        loss.backward()
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(NUM_STEPS):
        optimizer.step()
    end_ev.record()
    torch.cuda.synchronize()

    optim_ms_total = start_ev.elapsed_time(end_ev)
    optim_ms = optim_ms_total / NUM_STEPS
    print(f"  Per step: {optim_ms:.1f}ms")
    print()

    # ── 4. Full training step ────────────────────────────────────────
    print("=" * 60)
    print(f"PHASE 4: Full training step (data + compute + optimizer, decode={decode_str})")
    print("=" * 60)
    loader, dm = _build_loader(use_dali, optimized=dali_opt_str, prefetch_factor=best_pf, num_workers=nw)
    dm.trainer = type("_Mock", (), {"training": True})()

    it = iter(loader)
    # Warmup: run enough iterations to let torch.compile finish compiling
    # the augmentation pipeline AND the model with real data
    full_warmup = 20
    print(f"  Warming up full pipeline ({full_warmup} iters)...")
    for _ in range(full_warmup):
        batch = prepare_batch(next(it), dm, device, use_dali)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)["logits"]
            loss = loss_fn(out, batch["label"])
            loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_STEPS):
        batch = prepare_batch(next(it), dm, device, use_dali)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)["logits"]
            loss = loss_fn(out, batch["label"])
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    full_time = time.perf_counter() - t0
    full_ms = full_time / NUM_STEPS * 1000
    full_tput = BATCH_SIZE * NUM_STEPS / full_time
    print(f"  Per step: {full_ms:.1f}ms")
    print(f"  Throughput: {full_tput:.0f} samples/sec")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    overhead_ms = full_ms - data_ms - compute_ms - optim_ms
    print("=" * 60)
    print("SUMMARY (per step)")
    print("=" * 60)
    print(f"  Data loading:    {data_ms:7.1f}ms  ({data_ms / full_ms * 100:5.1f}%)")
    print(f"  Forward+backward:{compute_ms:7.1f}ms  ({compute_ms / full_ms * 100:5.1f}%)")
    print(f"  Optimizer step:  {optim_ms:7.1f}ms  ({optim_ms / full_ms * 100:5.1f}%)")
    print(f"  Other overhead:  {overhead_ms:7.1f}ms  ({overhead_ms / full_ms * 100:5.1f}%)")
    print(f"  ─────────────────────────")
    print(f"  Full step:       {full_ms:7.1f}ms")
    print()

    if data_ms > compute_ms:
        ratio = data_ms / compute_ms
        print(f"  ** DATA LOADING is {ratio:.1f}x slower than compute → DATA-BOUND **")
    else:
        ratio = compute_ms / data_ms
        print(f"  ** COMPUTE is {ratio:.1f}x slower than data loading → COMPUTE-BOUND **")

    # ── Write JSON results ───────────────────────────────────────────
    results = {
        "timestamp": timestamp,
        "node": node,
        "gpu": gpu_name,
        "mode": mode_str,
        "decode": f"dali-{dali_opt_str}" if dali_opt_str else ("dali" if use_dali else "cpu"),
        "optimizer": OPTIMIZER_NAME,
        "batch_size": BATCH_SIZE,
        "num_workers": nw,
        "num_steps": NUM_STEPS,
        "prefetch_sweep": pf_results,
        "best_prefetch_factor": best_pf,
        "data_ms": round(data_ms, 1),
        "data_tput": round(data_tput),
        "compute_ms": round(compute_ms, 1),
        "compute_tput": round(compute_tput),
        "optim_ms": round(optim_ms, 1),
        "full_ms": round(full_ms, 1),
        "full_tput": round(full_tput),
        "overhead_ms": round(overhead_ms, 1),
        "data_pct": round(data_ms / full_ms * 100, 1),
        "compute_pct": round(compute_ms / full_ms * 100, 1),
    }

    tracker_dir = Path("benchmarks")
    tracker_dir.mkdir(exist_ok=True)
    jsonl_path = tracker_dir / f"dataloader_profile_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(results) + "\n")
    print(f"\n  Results appended to {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile training pipeline bottleneck")
    parser.add_argument("--eager", action="store_true", help="Eager mode (no torch.compile)")
    parser.add_argument("--dali", action="store_true", help="Use NVIDIA DALI pipeline")
    parser.add_argument("--dali-optimized", action="store_true", help="Use optimised DALI pipeline (v2)")
    parser.add_argument("--dali-v3", action="store_true", help="Use optimised DALI pipeline v3 (bf16, CHW)")
    parser.add_argument("--num-workers", type=int, default=14)
    main(parser.parse_args())
