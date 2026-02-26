"""Profile the training pipeline to identify the wall-clock bottleneck.

Measures data loading, forward, backward, and optimizer step independently
to show where time is actually spent in the training loop.

Usage (single-GPU):
    PYTHONPATH=. python scripts/profile_training_bottleneck.py --dali-optimized --num-workers 12
Usage (multi-GPU with DDP):
    PYTHONPATH=. torchrun --nproc_per_node=8 scripts/profile_training_bottleneck.py --ddp --dali-optimized --num-workers 12
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = IMAGE_SIZE // PATCH_SIZE

MODEL_PRESETS = {
    "small": {"hidden_dim": 384, "num_heads": 6, "num_blocks": 12, "num_registers": 4},
    "base":  {"hidden_dim": 768, "num_heads": 12, "num_blocks": 12, "num_registers": 4},
}

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


def build_dali_dataloader(prefetch_factor: int = 2, num_workers: int = 14, optimized: str = "", device_id: int = 0):
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
        augment_cfg=AUGMENT_CFG, device_id=device_id,
    )
    if optimized == "fused":
        from experiments.datamodules.dali_imagenet_fused import DALIImageNetFusedDataModule
        dm = DALIImageNetFusedDataModule(**common)
    elif optimized == "v3":
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


def _build_loader(use_dali: bool, optimized: str = "", device_id: int = 0, **kwargs):
    if use_dali:
        return build_dali_dataloader(optimized=optimized, device_id=device_id, **kwargs)
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
    global HIDDEN_DIM, NUM_HEADS, NUM_BLOCKS, NUM_REGISTERS
    preset = MODEL_PRESETS[args.model_size]
    HIDDEN_DIM = preset["hidden_dim"]
    NUM_HEADS = preset["num_heads"]
    NUM_BLOCKS = preset["num_blocks"]
    NUM_REGISTERS = preset["num_registers"]

    use_ddp = args.ddp
    local_rank = 0
    world_size = 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    rank0 = local_rank == 0
    mode_str = "eager" if args.eager else "compiled"
    nw = args.num_workers
    use_dali = args.dali or args.dali_optimized or args.dali_v3 or args.dali_fused
    if args.dali_fused:
        dali_opt_str = "fused"
    elif args.dali_v3:
        dali_opt_str = "v3"
    elif args.dali_optimized:
        dali_opt_str = "v2"
    else:
        dali_opt_str = ""
    if dali_opt_str == "fused":
        decode_str = "DALI-fused (augmentations in DALI pipeline)"
    elif dali_opt_str == "v3":
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
    ddp_str = f"DDP x{world_size}" if use_ddp else "single-GPU"

    def log(msg=""):
        if rank0:
            print(msg, flush=True)

    model_size = args.model_size.upper()
    log(f"Device: {gpu_name} (node={node})")
    log(f"Model: ViT-5-{model_size} (dim={HIDDEN_DIM}, heads={NUM_HEADS}, blocks={NUM_BLOCKS})")
    log(f"Mode: {mode_str.upper()}, {ddp_str}")
    log(f"Decode: {decode_str}")
    log(f"Optimizer: {OPTIMIZER_NAME}")
    log(f"Batch size: {BATCH_SIZE} per GPU, Workers: {nw}")
    log(f"Timestamp: {timestamp}")
    log()

    # ── 0. Prefetch factor sweep (rank 0 only for data benchmarks) ──
    log("=" * 60)
    log("PHASE 0: Prefetch factor sweep (data loading only)")
    log("=" * 60)
    prefetch_factors = [2, 4, 8, 16]
    pf_results = {}
    best_pf, best_ms = 2, float("inf")
    for pf in prefetch_factors:
        ms, tput = bench_dataloader(pf, nw, use_dali=use_dali, optimized=dali_opt_str)
        pf_results[pf] = {"ms": round(ms, 1), "tput": round(tput)}
        log(f"  prefetch_factor={pf:>2d}: {ms:6.1f}ms/batch, {tput:>5.0f} samples/sec")
        if ms < best_ms:
            best_ms, best_pf = ms, pf
    log(f"  >> Best: prefetch_factor={best_pf} ({best_ms:.1f}ms)")
    log()

    # ── 1. Data loading speed ────────────────────────────────────────
    log("=" * 60)
    log(f"PHASE 1: Data loading speed (prefetch_factor={best_pf})")
    log("=" * 60)
    data_ms, data_tput = bench_dataloader(best_pf, nw, use_dali=use_dali, optimized=dali_opt_str, warmup=5)
    log(f"  Per batch: {data_ms:.1f}ms")
    log(f"  Throughput: {data_tput:.0f} samples/sec")
    log()

    # ── 2. Forward + backward (no DDP — pure compute baseline) ───────
    log("=" * 60)
    log(f"PHASE 2: Forward + backward — NO DDP (synthetic, mode={mode_str})")
    log("=" * 60)
    raw_model = build_model().to(device)
    if not args.eager:
        log("  Compiling with max-autotune...")
        raw_model = torch.compile(raw_model, mode="max-autotune")
    loss_fn = nn.CrossEntropyLoss()

    fake_img = torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, device=device)
    fake_lbl = torch.randint(0, 1000, (BATCH_SIZE,), device=device)
    fake_batch = {"input": fake_img, "label": fake_lbl, "condition": None}

    warmup_iters = 5 if args.eager else 20
    log(f"  Warming up ({warmup_iters} iters)...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup_iters):
            out = raw_model(fake_batch)["logits"]
            loss = loss_fn(out, fake_lbl)
            loss.backward()
            raw_model.zero_grad()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(NUM_STEPS):
            out = raw_model(fake_batch)["logits"]
            loss = loss_fn(out, fake_lbl)
            loss.backward()
            raw_model.zero_grad()
    end_ev.record()
    torch.cuda.synchronize()

    compute_ms_total = start_ev.elapsed_time(end_ev)
    compute_ms = compute_ms_total / NUM_STEPS
    compute_tput = BATCH_SIZE * NUM_STEPS / (compute_ms_total / 1000)
    log(f"  Per step: {compute_ms:.1f}ms")
    log(f"  Throughput: {compute_tput:.0f} samples/sec")
    log()

    # ── 2b. Forward + backward WITH DDP (backward overlaps allreduce) ─
    ddp_compute_ms = compute_ms
    allreduce_ms = 0.0
    if use_ddp:
        log("=" * 60)
        log(f"PHASE 2b: Forward + backward — WITH DDP (synthetic, mode={mode_str})")
        log("=" * 60)
        ddp_model = DDP(raw_model, device_ids=[local_rank])
        log(f"  Warming up DDP ({warmup_iters} iters)...")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(warmup_iters):
                out = ddp_model(fake_batch)["logits"]
                loss = loss_fn(out, fake_lbl)
                loss.backward()
                ddp_model.zero_grad()
        torch.cuda.synchronize()

        start_ev.record()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(NUM_STEPS):
                out = ddp_model(fake_batch)["logits"]
                loss = loss_fn(out, fake_lbl)
                loss.backward()
                ddp_model.zero_grad()
        end_ev.record()
        torch.cuda.synchronize()

        ddp_compute_total = start_ev.elapsed_time(end_ev)
        ddp_compute_ms = ddp_compute_total / NUM_STEPS
        ddp_compute_tput = BATCH_SIZE * world_size * NUM_STEPS / (ddp_compute_total / 1000)
        log(f"  Per step (fwd+bwd+allreduce): {ddp_compute_ms:.1f}ms")
        log(f"  Aggregate throughput: {ddp_compute_tput:.0f} samples/sec")
        log()

        # ── 2c. Allreduce only (measure raw communication cost) ──────
        log("=" * 60)
        log("PHASE 2c: Allreduce only (raw NCCL cost)")
        log("=" * 60)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = ddp_model(fake_batch)["logits"]
            loss = loss_fn(out, fake_lbl)
            loss.backward()
        grads = [p.grad.clone() for p in ddp_model.parameters() if p.grad is not None]
        ddp_model.zero_grad()
        torch.cuda.synchronize()

        start_ev.record()
        for _ in range(NUM_STEPS):
            for g in grads:
                dist.all_reduce(g, op=dist.ReduceOp.AVG)
        end_ev.record()
        torch.cuda.synchronize()

        ar_total = start_ev.elapsed_time(end_ev)
        allreduce_ms = ar_total / NUM_STEPS
        overlap_ms = compute_ms + allreduce_ms - ddp_compute_ms
        log(f"  Per step (all params): {allreduce_ms:.1f}ms")
        log(f"  Overlap with backward: {overlap_ms:.1f}ms "
            f"({overlap_ms / allreduce_ms * 100:.0f}% of allreduce hidden)" if allreduce_ms > 0 else "")
        log()

        model = ddp_model
    else:
        model = raw_model

    # ── 3. Optimizer step ────────────────────────────────────────────
    log("=" * 60)
    log(f"PHASE 3: Optimizer step ({OPTIMIZER_NAME})")
    log("=" * 60)
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
    log(f"  Per step: {optim_ms:.1f}ms")
    log()

    # ── 4. Full training step ────────────────────────────────────────
    log("=" * 60)
    log(f"PHASE 4: Full training step (data + compute + optimizer, decode={decode_str})")
    log("=" * 60)
    loader, dm = _build_loader(
        use_dali, optimized=dali_opt_str, device_id=local_rank,
        prefetch_factor=best_pf, num_workers=nw,
    )
    dm.trainer = type("_Mock", (), {"training": True})()

    it = iter(loader)
    full_warmup = 20
    log(f"  Warming up full pipeline ({full_warmup} iters)...")
    for _ in range(full_warmup):
        batch = prepare_batch(next(it), dm, device, use_dali)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)["logits"]
            loss = loss_fn(out, batch["label"])
            loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    if use_ddp:
        dist.barrier()
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
    agg_tput = full_tput * world_size
    log(f"  Per step: {full_ms:.1f}ms")
    log(f"  Per-GPU throughput: {full_tput:.0f} samples/sec")
    if use_ddp:
        log(f"  Aggregate throughput: {agg_tput:.0f} samples/sec ({world_size} GPUs)")
    log()

    # ── Summary ──────────────────────────────────────────────────────
    effective_compute = ddp_compute_ms if use_ddp else compute_ms
    overhead_ms = full_ms - data_ms - effective_compute - optim_ms
    log("=" * 60)
    log("SUMMARY (per step)")
    log("=" * 60)
    log(f"  Data loading:       {data_ms:7.1f}ms  ({data_ms / full_ms * 100:5.1f}%)")
    log(f"  Fwd+bwd (no DDP):   {compute_ms:7.1f}ms  ({compute_ms / full_ms * 100:5.1f}%)")
    if use_ddp:
        log(f"  Allreduce (raw):    {allreduce_ms:7.1f}ms  ({allreduce_ms / full_ms * 100:5.1f}%)")
        log(f"  Fwd+bwd+AR (DDP):  {ddp_compute_ms:7.1f}ms  ({ddp_compute_ms / full_ms * 100:5.1f}%)")
        overlap_ms = compute_ms + allreduce_ms - ddp_compute_ms
        log(f"  AR overlap w/ bwd: {overlap_ms:7.1f}ms  ({overlap_ms / allreduce_ms * 100:.0f}% hidden)" if allreduce_ms > 0 else "")
    log(f"  Optimizer step:     {optim_ms:7.1f}ms  ({optim_ms / full_ms * 100:5.1f}%)")
    log(f"  Other overhead:     {overhead_ms:7.1f}ms  ({overhead_ms / full_ms * 100:5.1f}%)")
    log(f"  ─────────────────────────")
    log(f"  Full step:          {full_ms:7.1f}ms")
    log()

    if data_ms > effective_compute:
        ratio = data_ms / effective_compute
        log(f"  ** DATA LOADING is {ratio:.1f}x slower than compute -> DATA-BOUND **")
    else:
        ratio = effective_compute / data_ms
        log(f"  ** COMPUTE is {ratio:.1f}x slower than data loading -> COMPUTE-BOUND **")

    # ── Write JSON results ───────────────────────────────────────────
    if rank0:
        results = {
            "timestamp": timestamp,
            "node": node,
            "gpu": gpu_name,
            "model_size": args.model_size,
            "hidden_dim": HIDDEN_DIM,
            "mode": mode_str,
            "ddp": use_ddp,
            "world_size": world_size,
            "decode": f"dali-{dali_opt_str}" if dali_opt_str else ("dali" if use_dali else "cpu"),
            "optimizer": OPTIMIZER_NAME,
            "batch_size": BATCH_SIZE,
            "num_workers": nw,
            "num_steps": NUM_STEPS,
            "prefetch_sweep": pf_results,
            "best_prefetch_factor": best_pf,
            "data_ms": round(data_ms, 1),
            "data_tput": round(data_tput),
            "compute_no_ddp_ms": round(compute_ms, 1),
            "compute_ddp_ms": round(ddp_compute_ms, 1),
            "allreduce_ms": round(allreduce_ms, 1),
            "optim_ms": round(optim_ms, 1),
            "full_ms": round(full_ms, 1),
            "full_tput": round(full_tput),
            "agg_tput": round(agg_tput),
            "overhead_ms": round(overhead_ms, 1),
            "data_pct": round(data_ms / full_ms * 100, 1),
            "compute_pct": round(effective_compute / full_ms * 100, 1),
        }

        tracker_dir = Path("benchmarks")
        tracker_dir.mkdir(exist_ok=True)
        jsonl_path = tracker_dir / f"dataloader_profile_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(results) + "\n")
        log(f"\n  Results appended to {jsonl_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile training pipeline bottleneck")
    parser.add_argument("--eager", action="store_true", help="Eager mode (no torch.compile)")
    parser.add_argument("--dali", action="store_true", help="Use NVIDIA DALI pipeline")
    parser.add_argument("--dali-optimized", action="store_true", help="Use optimised DALI pipeline (v2)")
    parser.add_argument("--dali-v3", action="store_true", help="Use optimised DALI pipeline v3 (bf16, CHW)")
    parser.add_argument("--dali-fused", action="store_true", help="Use fully-fused DALI pipeline (augmentations in DALI)")
    parser.add_argument("--ddp", action="store_true", help="Run with DDP across all visible GPUs")
    parser.add_argument("--num-workers", type=int, default=14)
    parser.add_argument("--model-size", choices=["small", "base"], default="small",
                        help="Model size: small (384d/6h) or base (768d/12h)")
    main(parser.parse_args())
