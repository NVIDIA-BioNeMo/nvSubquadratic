"""Test timm PatchEmbed patchify/unpatchify reconstruction capability.

This test uses timm's PatchEmbed for patchification and the standard MAE-style
unpatchification (linear projection + einops rearrange) for comparison.

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_timm_patchify_reconstruction.py
    PYTHONPATH=. python nvsubquadratic/modules/test_timm_patchify_reconstruction.py --num_steps 2000
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import PatchEmbed

from nvsubquadratic.modules.test_spectral_patchify_reconstruction import (
    TEST_CONFIG,
    get_test_target,
)


class TimmPatchifyReconstructionNet(nn.Module):
    """Network using timm's PatchEmbed for patchify and MAE-style unpatchify.

    Standard MAE unpatchify:
    1. Linear projection from embed_dim to patch_size^2 * in_channels
    2. Reshape using einops to reconstruct spatial dimensions
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        img_size: int = 64,
        patch_size: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size

        # timm's PatchEmbed: Conv2d with stride=patch_size
        # flatten=False keeps spatial dimensions (B, embed_dim, H/patch, W/patch)
        self.patchify = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_channels,
            flatten=False,  # Keep spatial dims
            strict_img_size=False,  # Allow different sizes
        )

        # MAE-style unpatchify: Linear projection to pixel space
        # Project from hidden_channels to patch_size^2 * in_channels
        # This is equivalent to predicting all pixels in each patch
        self.unpatchify_proj = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=patch_size * patch_size * in_channels,
            kernel_size=1,  # 1x1 conv = per-position linear projection
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass."""
        intermediates = {"input": x}
        B, C, H, W = x.shape

        # Patchify: (B, C, H, W) -> (B, hidden, H/p, W/p)
        x_patched = self.patchify(x)
        intermediates["patched"] = x_patched

        # Project to pixel space: (B, hidden, H/p, W/p) -> (B, p*p*C, H/p, W/p)
        x_proj = self.unpatchify_proj(x_patched)
        intermediates["projected"] = x_proj

        # Reshape to image using einops:
        # (B, p*p*C, H/p, W/p) -> (B, C, H, W)
        p = self.patch_size
        x_out = rearrange(x_proj, "b (ph pw c) h w -> b c (h ph) (w pw)", ph=p, pw=p, c=self.in_channels)

        intermediates["output"] = x_out

        return x_out, intermediates


def main(num_steps: int | None = None):
    print("=" * 70)
    print("Timm PatchEmbed Reconstruction Test")
    print("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Use shared configuration
    batch_size = TEST_CONFIG["batch_size"]
    in_channels = TEST_CONFIG["in_channels"]
    # Use enough hidden channels to avoid bottleneck: patch_size^2 * in_channels
    patch_size = TEST_CONFIG["stride"]
    hidden_channels = patch_size * patch_size * in_channels  # 4*4*3 = 48
    H, W = TEST_CONFIG["H"], TEST_CONFIG["W"]
    stride = patch_size

    if num_steps is None:
        num_steps = TEST_CONFIG["num_steps"]

    print("\nConfiguration (shared with other tests):")
    print(f"  Input shape: [{batch_size}, {in_channels}, {H}, {W}]")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Patch size (stride): {stride}")
    print(f"  Patched shape: [{batch_size}, {hidden_channels}, {H // stride}, {W // stride}]")
    print(f"  Training steps: {num_steps}")

    # Get shared test target
    target = get_test_target(device)
    print(f"\nTarget tensor: shape={target.shape}, mean={target.mean():.4f}, std={target.std():.4f}")

    # Create model
    model = TimmPatchifyReconstructionNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        img_size=H,
        patch_size=stride,
    ).to(device)

    # Print model info
    print("\nModel:")
    print(f"  Patchify (timm): {model.patchify}")
    print(f"  Unpatchify proj: {model.unpatchify_proj}")
    print("  Unpatchify method: einops rearrange (MAE-style)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {trainable_params:,} / {total_params:,}")

    # Initial forward pass
    print("\n" + "-" * 70)
    print("Initial forward pass:")
    print("-" * 70)
    with torch.no_grad():
        output, intermediates = model(target)
        for name, tensor in intermediates.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  {name:24s}: {tuple(tensor.shape)}")
        initial_loss = F.mse_loss(output, target).item()
    print(f"\nInitial MSE loss: {initial_loss:.6f}")

    # Training
    print("\n" + "-" * 70)
    print("Training:")
    print("-" * 70)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        output, _ = model(target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0 or step == num_steps - 1:
            print(f"Step {step:4d}: Loss = {loss.item():.6f}")

    # Final evaluation
    print("\n" + "-" * 70)
    print("Final evaluation:")
    print("-" * 70)
    with torch.no_grad():
        output, intermediates = model(target)
        final_loss = F.mse_loss(output, target).item()

        # Per-pixel metrics
        abs_error = (output - target).abs()
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()

        # Correlation
        target_flat = target.flatten()
        output_flat = output.flatten()
        correlation = torch.corrcoef(torch.stack([target_flat, output_flat]))[0, 1].item()

        print(f"Final MSE loss: {final_loss:.6f}")
        print(f"Mean absolute error: {mean_error:.6f}")
        print(f"Max absolute error: {max_error:.6f}")
        print(f"Correlation: {correlation:.6f}")
        print(f"Loss reduction: {initial_loss / final_loss:.1f}x")

    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Show one sample from batch
        sample_idx = 0

        # Target
        ax = axes[0, 0]
        img = target[sample_idx].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title("Target")
        ax.axis("off")

        # Output
        ax = axes[0, 1]
        img = output[sample_idx].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(f"Output (Loss={final_loss:.6f})")
        ax.axis("off")

        # Difference
        ax = axes[0, 2]
        diff = (output[sample_idx] - target[sample_idx]).abs().mean(dim=0).cpu().numpy()
        im = ax.imshow(diff, cmap="hot")
        ax.set_title(f"Abs Diff (max={max_error:.4f})")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

        # Patched representation
        ax = axes[1, 0]
        patched = intermediates["patched"][sample_idx]
        patched_vis = patched[:3].permute(1, 2, 0).cpu().numpy()
        patched_vis = (patched_vis - patched_vis.min()) / (patched_vis.max() - patched_vis.min() + 1e-8)
        ax.imshow(patched_vis)
        ax.set_title("Patched (first 3 ch)")
        ax.axis("off")

        # Training loss curve
        ax = axes[1, 1]
        ax.plot(losses)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training Loss")
        ax.set_yscale("log")
        ax.grid(True)

        # Text summary
        ax = axes[1, 2]
        ax.axis("off")
        summary = f"""Timm PatchEmbed Reconstruction

Input: {list(target.shape)}
Patch size: {stride}
Hidden channels: {hidden_channels}

Final Loss: {final_loss:.6f}
Correlation: {correlation:.4f}
Max Error: {max_error:.4f}

Trainable params: {trainable_params:,}
"""
        ax.text(0.1, 0.5, summary, fontsize=10, family="monospace", verticalalignment="center", transform=ax.transAxes)

        plt.tight_layout()
        save_path = "nvsubquadratic/modules/timm_patchify_reconstruction_test.png"
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved visualization to: {save_path}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available, skipping visualization")

    # Summary
    print("\n" + "=" * 70)
    if final_loss < 0.01 and correlation > 0.99:
        print("✅ SUCCESS: Timm PatchEmbed can reconstruct input!")
    elif final_loss < 0.1 and correlation > 0.95:
        print("⚠️ PARTIAL SUCCESS: Reconstruction is reasonable but not perfect")
    else:
        print("❌ FAILED: Timm PatchEmbed pipeline loses too much information")
    print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    print("=" * 70)

    return final_loss, correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=None, help="Number of training steps")
    args = parser.parse_args()

    main(num_steps=args.num_steps)
