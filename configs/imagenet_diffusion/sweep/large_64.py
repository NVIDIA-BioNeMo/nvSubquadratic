from configs.imagenet_diffusion.ccnn_12_768_hyena_qknorm import get_config as get_base_config


def get_config():
    """Large model with 64x64 resolution.

    Est. Param count: ~548M
    Est. Memory: Medium. Peak ~61 GB at BS=16.
    Nodes (GBS=512): 4
    Nodes (GBS=1024): 8
    """
    config = get_base_config()

    # Large Model Specs
    config.net.hidden_dim = 1024
    config.net.num_blocks = 16
    config.diffusion.time_embed_dim = 1024
    config.net.norm_cfg.normalized_shape = 1024

    # Batch Size (Conservative, large footprint)
    config.dataset.batch_size = 16

    # Resolution
    config.dataset.final_image_size = 64
    config.diffusion.cosine_schedule_image_resolution = 64
    config.diffusion.cosine_schedule_noise_res_high = 64
    config.diffusion.cosine_schedule_noise_res_low = 32

    # Tags
    config.wandb.job_group = "sweep_large_64"

    return config
