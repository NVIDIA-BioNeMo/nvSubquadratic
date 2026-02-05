from examples.imagenet_diffusion.ccnn_12_768_hyena_qknorm import get_config as get_base_config


def get_config():
    """Large model with 128x128 resolution.

    Est. Param count: ~548M
    Est. Memory: High. Peak ~71 GB at BS=4.
    Nodes (GBS=512): 16
    Nodes (GBS=1024): 32
    """
    config = get_base_config()

    # Large Model Specs
    config.net.hidden_dim = 1024
    config.net.num_blocks = 16
    config.diffusion.time_embed_dim = 1024
    config.net.norm_cfg.normalized_shape = 1024

    # Batch Size (Optimized for 80GB: Fits 4)
    config.dataset.batch_size = 4

    # Resolution
    config.dataset.final_image_size = 128
    config.diffusion.cosine_schedule_image_resolution = 128
    config.diffusion.cosine_schedule_noise_res_high = 128
    config.diffusion.cosine_schedule_noise_res_low = 64

    # Tags
    config.wandb.job_group = "sweep_large_128"

    return config
