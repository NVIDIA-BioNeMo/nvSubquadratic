"""Profile the training pipeline to identify the wall-clock bottleneck.

Measures data loading, forward, backward, and optimizer step independently
to show where time is actually spent in the training loop.

Usage (interactive SLURM session with 1 GPU):
    PYTHONPATH=. python scripts/profile_training_bottleneck.py
"""

import os
import time

import torch
import torch.nn as nn

os.environ.setdefault("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
os.environ.setdefault("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

from experiments.datamodules.imagenet import ImageNetDataModule, MixupConfig, AugmentConfig
from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet

BATCH_SIZE = 256
NUM_STEPS = 50
HIDDEN_DIM = 384
NUM_HEADS = 6
NUM_BLOCKS = 12
NUM_REGISTERS = 4
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE)


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


def build_dataloader(prefetch_factor: int = 4):
    folder_path = os.environ.get("IMAGENET_FOLDER_PATH")
    dm = ImageNetDataModule(
        data_dir=os.environ["IMAGENET_PATH"],
        imagefolder_dir=folder_path,
        prefetch_factor=prefetch_factor,
        batch_size=BATCH_SIZE, num_workers=16, pin_memory=True, seed=42,
        image_size=IMAGE_SIZE, final_image_size=IMAGE_SIZE,
        center_crop=True, num_classes=1000, drop_labels=False,
        hf_dataset_name="ILSVRC/imagenet-1k", hf_dataset_config=None,
        hf_auth_token=os.environ.get("HF_TOKEN"), task="classification",
        mixup_cfg=MixupConfig(mixup=0.8, cutmix=1.0, mixup_prob=1.0,
                              mixup_switch_prob=0.5, smoothing=0.0),
        augment_cfg=AugmentConfig(use_three_augment=True, color_jitter=0.3),
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm.train_dataloader()


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Batch size: {BATCH_SIZE}, Workers: 16")
    print()

    # ── 0. Prefetch factor sweep ───────────────────────────────────────
    print("=" * 60)
    print("PHASE 0: Prefetch factor sweep (data loading only)")
    print("=" * 60)
    best_pf, best_ms = 2, float("inf")
    for pf in [2, 4, 8]:
        loader = build_dataloader(prefetch_factor=pf)
        it = iter(loader)
        for _ in range(3):
            next(it)
        t0 = time.perf_counter()
        for _ in range(NUM_STEPS):
            next(it)
        elapsed = time.perf_counter() - t0
        ms = elapsed / NUM_STEPS * 1000
        tput = BATCH_SIZE * NUM_STEPS / elapsed
        print(f"  prefetch_factor={pf}: {ms:.1f}ms/batch, {tput:.0f} samples/sec")
        if ms < best_ms:
            best_ms, best_pf = ms, pf
        del loader, it
    print(f"  >> Best: prefetch_factor={best_pf} ({best_ms:.1f}ms)")
    print()

    # ── 1. Data loading speed (using best prefetch_factor) ────────────
    print("=" * 60)
    print(f"PHASE 1: Data loading speed (prefetch_factor={best_pf})")
    print("=" * 60)
    loader = build_dataloader(prefetch_factor=best_pf)
    it = iter(loader)

    for _ in range(3):
        next(it)

    t0 = time.perf_counter()
    for i in range(NUM_STEPS):
        batch = next(it)
    data_time = time.perf_counter() - t0
    data_per_step = data_time / NUM_STEPS
    data_throughput = BATCH_SIZE * NUM_STEPS / data_time
    print(f"  {NUM_STEPS} batches in {data_time:.2f}s")
    print(f"  Per batch: {data_per_step * 1000:.1f}ms")
    print(f"  Throughput: {data_throughput:.0f} samples/sec")
    print()

    # ── 2. Model forward + backward (synthetic data) ──────────────────
    print("=" * 60)
    print("PHASE 2: Forward + backward (synthetic data, no data loading)")
    print("=" * 60)
    model = build_model().to(device)
    model = torch.compile(model, mode="max-autotune")
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic input matching what the model expects (channels-last dict)
    fake_img = torch.randn(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, device=device)
    fake_lbl = torch.randint(0, 1000, (BATCH_SIZE,), device=device)
    fake_batch = {"input": fake_img, "label": fake_lbl, "condition": None}

    # Compile warmup
    print("  Warming up torch.compile...")
    for _ in range(5):
        out = model(fake_batch)["logits"]
        loss = loss_fn(out, fake_lbl)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_STEPS):
        out = model(fake_batch)["logits"]
        loss = loss_fn(out, fake_lbl)
        loss.backward()
        model.zero_grad()
    end.record()
    torch.cuda.synchronize()

    compute_ms = start.elapsed_time(end)
    compute_per_step = compute_ms / NUM_STEPS
    compute_throughput = BATCH_SIZE * NUM_STEPS / (compute_ms / 1000)
    print(f"  {NUM_STEPS} steps in {compute_ms:.0f}ms")
    print(f"  Per step: {compute_per_step:.1f}ms")
    print(f"  Throughput: {compute_throughput:.0f} samples/sec")
    print()

    # ── 3. Optimizer step ─────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 3: Optimizer step timing")
    print("=" * 60)
    from torch_optimizer import Lamb
    optimizer = Lamb(model.parameters(), lr=4e-3, weight_decay=0.05)

    # Run one forward-backward to have gradients
    out = model(fake_batch)["logits"]
    loss = loss_fn(out, fake_lbl)
    loss.backward()
    torch.cuda.synchronize()

    start.record()
    for _ in range(NUM_STEPS):
        optimizer.step()
    end.record()
    torch.cuda.synchronize()

    optim_ms = start.elapsed_time(end)
    optim_per_step = optim_ms / NUM_STEPS
    print(f"  {NUM_STEPS} steps in {optim_ms:.0f}ms")
    print(f"  Per step: {optim_per_step:.1f}ms")
    print()

    # ── 4. Full training step (data loading + compute + optimizer) ────
    print("=" * 60)
    print("PHASE 4: Full training step (data + compute + optimizer)")
    print("=" * 60)
    it = iter(loader)
    for _ in range(3):
        next(it)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(NUM_STEPS):
        images, labels = next(it)
        images = images.permute(0, 2, 3, 1).contiguous().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch = {"input": images, "label": labels, "condition": None}
        optimizer.zero_grad()
        out = model(batch)["logits"]
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    full_time = time.perf_counter() - t0
    full_per_step = full_time / NUM_STEPS
    full_throughput = BATCH_SIZE * NUM_STEPS / full_time
    print(f"  {NUM_STEPS} steps in {full_time:.2f}s")
    print(f"  Per step: {full_per_step * 1000:.1f}ms")
    print(f"  Throughput: {full_throughput:.0f} samples/sec")
    print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY (per step)")
    print("=" * 60)
    print(f"  Data loading:    {data_per_step * 1000:7.1f}ms  ({data_per_step / full_per_step * 100:5.1f}%)")
    print(f"  Forward+backward:{compute_per_step:7.1f}ms  ({compute_per_step / 1000 / full_per_step * 100:5.1f}%)")
    print(f"  Optimizer step:  {optim_per_step:7.1f}ms  ({optim_per_step / 1000 / full_per_step * 100:5.1f}%)")
    overhead = full_per_step * 1000 - data_per_step * 1000 - compute_per_step - optim_per_step
    print(f"  Other overhead:  {overhead:7.1f}ms  ({overhead / (full_per_step * 1000) * 100:5.1f}%)")
    print(f"  ─────────────────────────")
    print(f"  Full step:       {full_per_step * 1000:7.1f}ms")
    print()

    if data_per_step * 1000 > compute_per_step:
        ratio = data_per_step * 1000 / compute_per_step
        print(f"  ** DATA LOADING is {ratio:.1f}x slower than compute **")
        print(f"  ** The training loop is DATA-BOUND, not compute-bound **")
        print(f"  ** Recommended: switch to torchvision ImageFolder or NVIDIA DALI **")
    else:
        print(f"  Training loop is COMPUTE-BOUND — optimizations are effective")


if __name__ == "__main__":
    main()
