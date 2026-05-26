"""Find the max batch size that fits in 80 GB for each supernova_explosion_64 model.

Runs a forward + backward pass on synthetic ``[B, C, *spatial]`` inputs
at increasing batch sizes and reports peak GPU memory.  Stops when peak
exceeds the configured budget (default 80 GB).

Targets: H100 SXM 80GB or any Ampere+ 80 GB GPU.

Usage:
    PYTHONPATH=. conda run -n nv-subq python benchmarks/well/profile_batch_size.py

Output: stdout summary table (peak GiB / batch size for each model).
"""

import gc
import sys

import torch


MEMORY_BUDGET_GB = 80.0
BATCH_SIZES = [2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]

IN_CHANNELS = 24
OUT_CHANNELS = 6
SPATIAL = (64, 64, 64)


def _resolve_patch_interpolations(net_cfg):
    """Replace OmegaConf interpolation strings that reference the parent 'net' key."""
    ps = net_cfg.in_proj_cfg.patch_size
    net_cfg.in_proj_cfg.stride = ps
    net_cfg.out_proj_cfg.patch_size = ps
    net_cfg.out_proj_cfg.stride = ps
    if hasattr(net_cfg, "block_cfg"):
        seq = net_cfg.block_cfg.sequence_mixer_cfg
        if hasattr(seq, "mixer_cfg") and hasattr(seq.mixer_cfg, "global_conv_cfg"):
            kernel = seq.mixer_cfg.global_conv_cfg.kernel_cfg
            kernel.L_cache = 64 // ps


def profile_model(model, name, device, dtype, compile_mode="default"):
    print(f"\n{'=' * 60}")
    print(f"Model: {name}  |  params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'=' * 60}")

    model = model.to(device=device)
    model = torch.compile(model, mode=compile_mode)

    print("  compiling (warmup)...", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    x_warm = torch.randn(2, *SPATIAL, IN_CHANNELS, device=device, dtype=torch.float32)
    c_warm = torch.zeros(2, *SPATIAL, 0, device=device, dtype=torch.float32)
    with torch.autocast("cuda", dtype=dtype):
        y_warm = model({"input": x_warm, "condition": c_warm})
    y_warm["logits"].float().sum().backward()
    model.zero_grad(set_to_none=True)
    del x_warm, c_warm, y_warm
    torch.cuda.empty_cache()
    gc.collect()
    print("  compilation done", flush=True)

    best_bs = 0

    for bs in BATCH_SIZES:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        gc.collect()

        x = torch.randn(bs, *SPATIAL, IN_CHANNELS, device=device, dtype=torch.float32)
        c = torch.zeros(bs, *SPATIAL, 0, device=device, dtype=torch.float32)

        try:
            with torch.autocast("cuda", dtype=dtype):
                out = model({"input": x, "condition": c})
            loss = out["logits"].float().sum()
            loss.backward()
            model.zero_grad(set_to_none=True)

            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            status = "OK" if peak_gb <= MEMORY_BUDGET_GB else "OVER"
            print(f"  bs={bs:>3d}  peak={peak_gb:6.2f} GB  [{status}]", flush=True)

            del x, c, out, loss
            torch.cuda.empty_cache()
            gc.collect()

            if peak_gb <= MEMORY_BUDGET_GB:
                best_bs = bs
            else:
                break
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:>3d}  OOM", flush=True)
            del x, c  # noqa: F821
            torch.cuda.empty_cache()
            gc.collect()
            break

    print(f"  >>> max batch size within {MEMORY_BUDGET_GB} GB: {best_bs}")
    return best_bs


def main():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    sys.path.insert(0, ".")

    # --- CNextU-net ---
    from nvsubquadratic.networks.baselines.unet_convnext import WellUNetConvNext

    unet = WellUNetConvNext(
        dim_in=IN_CHANNELS,
        dim_out=OUT_CHANNELS,
        n_spatial_dims=3,
        spatial_resolution=SPATIAL,
        stages=4,
        blocks_per_stage=2,
        blocks_at_neck=1,
        init_features=42,
        gradient_checkpointing=False,
    )
    profile_model(unet, "CNextU-net", device, dtype)
    del unet
    torch.cuda.empty_cache()
    gc.collect()

    # --- Hyena ---
    from examples.well.v2.supernova_explosion_64.hyena import get_config as get_hyena_cfg
    from nvsubquadratic.lazy_config import instantiate

    hyena_cfg = get_hyena_cfg()
    _resolve_patch_interpolations(hyena_cfg.net)
    hyena_net = instantiate(hyena_cfg.net)
    profile_model(hyena_net, "Hyena (zero-pad)", device, dtype)
    del hyena_net
    torch.cuda.empty_cache()
    gc.collect()

    # --- Attention ---
    from examples.well.v2.supernova_explosion_64.attention import get_config as get_attn_cfg

    attn_cfg = get_attn_cfg()
    _resolve_patch_interpolations(attn_cfg.net)
    attn_net = instantiate(attn_cfg.net)
    profile_model(attn_net, "Attention", device, dtype)
    del attn_net
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
