"""Test script for CKConv Patchify/Unpatchify reconstruction.

This script compares reconstruction performance of:
1. CKConv with kernel_size = stride (non-overlapping) - should reach zero loss
2. CKConv with kernel_size > stride (overlapping) - limited by SIREN capacity
3. Normal Patchify/Unpatchify with overlapping patches - baseline comparison

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_ckconv_patchify_reconstruction.py
"""

import torch
import torch.nn.functional as F

from nvsubquadratic.modules.ckconv_patchify import CKConvPatchify, CKConvUnpatchify
from nvsubquadratic.modules.patchify import Patchify, Unpatchify


class NetCKConv(torch.nn.Module):
    """Wrapper for CKConv patchify/unpatchify."""

    def __init__(self, patchify, unpatchify):
        super().__init__()
        self.patchify = patchify
        self.unpatchify = unpatchify

    def forward(self, x):
        return self.unpatchify(self.patchify(x), target_size=x.shape[2:])


class NetNormal(torch.nn.Module):
    """Wrapper for normal Patchify/Unpatchify."""

    def __init__(self, patchify, unpatchify):
        super().__init__()
        self.patchify = patchify
        self.unpatchify = unpatchify

    def forward(self, x):
        return self.unpatchify(self.patchify(x))


def train_and_evaluate(model, target, lr=0.0005, steps=5000, log_steps=None):
    """Train a model for reconstruction and return the minimum loss."""
    if log_steps is None:
        log_steps = [1, 1000, 2000, 3000, 4000, 5000]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    min_loss = float("inf")

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        output = model(target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        min_loss = min(min_loss, loss.item())

        if step in log_steps:
            print(f"Step {step:4d}: Loss={loss.item():.12f} (min={min_loss:.12f})")

    return min_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Fixed random seed and input for consistency
    torch.manual_seed(42)
    target_bchw = torch.randn(2, 3, 64, 64, device=device)
    target_bhwc = target_bchw.permute(0, 2, 3, 1)

    LR = 0.0005
    STEPS = 5000

    # Test 1: CKConv with kernel_size = stride = 4 (non-overlapping)
    print("=" * 70)
    print("TEST 1: CKConv with kernel_size=4, stride=4 (non-overlapping)")
    print("=" * 70)

    torch.manual_seed(123)
    p1 = CKConvPatchify(
        in_features=3,
        out_features=256,
        init_stride=4,
        max_stride=16,
        kernel_size=4,
        kernel_hidden_dim=32,
        kernel_num_layers=3,
    ).to(device)
    u1 = CKConvUnpatchify(
        in_features=256,
        out_features=3,
        init_stride=4,
        max_stride=16,
        kernel_size=4,
        kernel_hidden_dim=32,
        kernel_num_layers=3,
        target_size=64,
    ).to(device)
    model1 = NetCKConv(p1, u1).to(device)
    min1 = train_and_evaluate(model1, target_bchw, lr=LR, steps=STEPS)

    # Test 2: CKConv with kernel_size = 16, stride = 4 (overlapping)
    print()
    print("=" * 70)
    print("TEST 2: CKConv with kernel_size=16, stride=4 (overlapping)")
    print("=" * 70)

    torch.manual_seed(123)
    p2 = CKConvPatchify(
        in_features=3,
        out_features=256,
        init_stride=4,
        max_stride=16,
        kernel_size=16,
        kernel_hidden_dim=32,
        kernel_num_layers=3,
    ).to(device)
    u2 = CKConvUnpatchify(
        in_features=256,
        out_features=3,
        init_stride=4,
        max_stride=16,
        kernel_size=16,
        kernel_hidden_dim=32,
        kernel_num_layers=3,
        target_size=64,
    ).to(device)
    model2 = NetCKConv(p2, u2).to(device)
    min2 = train_and_evaluate(model2, target_bchw, lr=LR, steps=STEPS)

    # Test 3: Normal Patchify with kernel_size = 16, stride = 4 (overlapping)
    print()
    print("=" * 70)
    print("TEST 3: Normal Patchify with kernel_size=16, stride=4 (overlapping)")
    print("=" * 70)

    torch.manual_seed(123)
    p3 = Patchify(in_features=3, out_features=256, data_dim=2, patch_size=16, stride=4).to(device)
    u3 = Unpatchify(in_features=256, out_features=3, data_dim=2, patch_size=16, stride=4).to(device)
    model3 = NetNormal(p3, u3).to(device)
    min3 = train_and_evaluate(model3, target_bhwc, lr=LR, steps=STEPS)

    # Final comparison
    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    params3 = sum(p.numel() for p in model3.parameters())

    print(f"Test 1 - CKConv (ks=4, s=4):   min_loss={min1:.15f}, params={params1:,}")
    print(f"Test 2 - CKConv (ks=16, s=4):  min_loss={min2:.15f}, params={params2:,}")
    print(f"Test 3 - Normal (ks=16, s=4):  min_loss={min3:.15f}, params={params3:,}")
    print()
    print("Zero reached (< 1e-10):")
    print(f"  Test 1 (CKConv non-overlap): {min1 < 1e-10}")
    print(f"  Test 2 (CKConv overlap):     {min2 < 1e-10}")
    print(f"  Test 3 (Normal overlap):     {min3 < 1e-10}")


if __name__ == "__main__":
    main()
