"""Diffusion model initialization diagnostic script.

Traces activation magnitudes through the network at init with a fake normalized
image to identify where values blow up.

Usage:
    PYTHONPATH=. python benchmarks/vit5_imagenet/debug_diffusion_init.py
"""

import math

import torch
import torch.nn.functional as F

# ---- build the model from config ----------------------------------------
from examples.imagenet_diffusion.ccnn_jit_baseline import get_config
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper
from experiments.utils.cli import apply_config_overrides
from nvsubquadratic.lazy_config import instantiate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4
IMAGE_SIZE = 64  # must match config


def make_fake_batch(batch_size, image_size, device):
    """Return a channels-last batch in [-1, 1] with random class labels."""
    # Simulate a normalised imagenet image: roughly unit-normal clipped to [-1,1]
    images = torch.randn(batch_size, image_size, image_size, 3, device=device).clamp(-1, 1)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    return {"input": images, "label": labels}


@torch.no_grad()
def run_diagnostics():
    print("=" * 70)
    print("Diffusion init diagnostics")
    print("=" * 70)

    cfg = apply_config_overrides(get_config(), [])  # resolves ${...} interpolations
    network = instantiate(cfg.net).to(DEVICE).eval()
    wrapper = DiffusionWrapper(network=network, cfg=cfg).to(DEVICE).eval()

    total_params = sum(p.numel() for p in network.parameters())
    print(f"\nNetwork params: {total_params / 1e6:.1f}M")

    batch = make_fake_batch(BATCH, IMAGE_SIZE, DEVICE)
    images = batch["input"]  # (B, H, W, 3), channels-last, in [-1, 1]
    labels = batch["label"]

    print(f"\nInput images: shape={tuple(images.shape)}, "
          f"mean={images.mean():.3f}, std={images.std():.3f}, "
          f"min={images.min():.3f}, max={images.max():.3f}")

    # --- timestep embedding ------------------------------------------------
    t_logit = torch.randn(BATCH, device=DEVICE) * wrapper.p_std + wrapper.p_mean
    timesteps = torch.sigmoid(t_logit)
    print(f"\nTimesteps: mean={timesteps.mean():.3f}, std={timesteps.std():.3f}, "
          f"min={timesteps.min():.3f}, max={timesteps.max():.3f}")

    condition, class_emb = wrapper._condition_from_timesteps(timesteps, labels=labels)
    print(f"Condition emb: shape={tuple(condition.shape)}, "
          f"mean={condition.mean():.3f}, std={condition.std():.3f}, "
          f"abs_max={condition.abs().max():.3f}")

    # --- noisy image -------------------------------------------------------
    images_bchw = torch.moveaxis(images, -1, 1).contiguous()
    eps_bchw = torch.randn_like(images_bchw) * wrapper.noise_scale
    t_b = timesteps.view(BATCH, 1, 1, 1)
    z_bchw = t_b * images_bchw + (1.0 - t_b) * eps_bchw
    target_v = images_bchw - eps_bchw
    print(f"\nNoisy image z: mean={z_bchw.mean():.3f}, std={z_bchw.std():.3f}, "
          f"abs_max={z_bchw.abs().max():.3f}")
    print(f"Target v:      mean={target_v.mean():.3f}, std={target_v.std():.3f}, "
          f"abs_max={target_v.abs().max():.3f}")

    # --- trace through network layer by layer ------------------------------
    print("\n--- Network activation trace ---")
    z_cl = torch.moveaxis(z_bchw, 1, -1).contiguous()

    # 1. dropout_in (no-op at eval)
    x = network.dropout_in(z_cl)

    # 2. in_proj (Patchify)
    x = network.in_proj(x)
    print(f"After Patchify:      shape={tuple(x.shape)}, "
          f"mean={x.mean():.3f}, std={x.std():.3f}, abs_max={x.abs().max():.3f}")

    # 3. condition_in_proj
    cond = network.condition_in_proj(condition)
    print(f"After cond_in_proj:  shape={tuple(cond.shape)}, "
          f"mean={cond.mean():.3f}, std={cond.std():.3f}, abs_max={cond.abs().max():.3f}")

    # 4. blocks (trace first, middle, last)
    for i, block in enumerate(network.blocks):
        x = block(x, cond)
        if i in (0, len(network.blocks) // 2, len(network.blocks) - 1):
            print(f"After block {i:2d}:       "
                  f"mean={x.mean():.3f}, std={x.std():.3f}, abs_max={x.abs().max():.3f}")

    # 5. out_norm
    x = network.out_norm(x)
    print(f"After out_norm:      "
          f"mean={x.mean():.3f}, std={x.std():.3f}, abs_max={x.abs().max():.3f}")

    # 6. out_proj (Unpatchify)
    x = network.out_proj(x)
    print(f"After Unpatchify:    shape={tuple(x.shape)}, "
          f"mean={x.mean():.3f}, std={x.std():.3f}, abs_max={x.abs().max():.3f}")

    # --- loss --------------------------------------------------------------
    prediction = x  # channels-last (B, H, W, 3)
    prediction_bchw = torch.moveaxis(prediction, -1, 1).contiguous()

    denominator = torch.clamp(1.0 - t_b, min=0.05)
    predicted_v = (prediction_bchw - z_bchw) / denominator
    loss = F.mse_loss(predicted_v, target_v)

    print(f"\nDenominator (1-t):   mean={denominator.mean():.3f}, "
          f"min={denominator.min():.3f}  (clamped at 0.05)")
    print(f"Prediction bchw:     mean={prediction_bchw.mean():.3f}, "
          f"std={prediction_bchw.std():.3f}, abs_max={prediction_bchw.abs().max():.3f}")
    print(f"Predicted v:         mean={predicted_v.mean():.3f}, "
          f"std={predicted_v.std():.3f}, abs_max={predicted_v.abs().max():.3f}")
    print(f"Target v:            mean={target_v.mean():.3f}, "
          f"std={target_v.std():.3f}, abs_max={target_v.abs().max():.3f}")
    print(f"\n>>> Initial loss: {loss.item():.4f}")

    # --- what loss would be with perfect zero-init prediction (prediction = z) ---
    ideal_predicted_v = (z_bchw - z_bchw) / denominator  # = 0
    ideal_loss = F.mse_loss(ideal_predicted_v, target_v)
    print(f">>> Loss if prediction=z (ideal zero-out): {ideal_loss.item():.4f}")

    # --- what loss would be with prediction = 0 ---
    zero_predicted_v = (torch.zeros_like(z_bchw) - z_bchw) / denominator
    zero_loss = F.mse_loss(zero_predicted_v, target_v)
    print(f">>> Loss if prediction=0: {zero_loss.item():.4f}")

    # --- Unpatchify weight scale check ------------------------------------
    deconv = network.out_proj.deconv
    w = deconv.weight
    print(f"\n--- Unpatchify (ConvTranspose2d) weight stats ---")
    print(f"  Shape: {tuple(w.shape)}  (in={w.shape[0]}, out={w.shape[1]}, k={w.shape[2]}x{w.shape[3]})")
    print(f"  std={w.std():.4f}, abs_max={w.abs().max():.4f}")
    print(f"  Expected output std from 768 channels of N(0,1): "
          f"{math.sqrt(w.shape[0]) * w.std().item():.4f}")

    # --- Patchify weight scale check -------------------------------------
    conv = network.in_proj.conv
    w2 = conv.weight
    print(f"\n--- Patchify (Conv2d) weight stats ---")
    print(f"  Shape: {tuple(w2.shape)}  (out={w2.shape[0]}, in={w2.shape[1]}, k={w2.shape[2]}x{w2.shape[3]})")
    print(f"  std={w2.std():.4f}, abs_max={w2.abs().max():.4f}")


if __name__ == "__main__":
    run_diagnostics()
