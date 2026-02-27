from configs.imagenet_diffusion.ccnn_12_768_hyena_qknorm import get_config as get_base_config


def get_config():
    """XL Model: 1536 dim, 24 layers. 128x128 Resolution.

    Est. Param count: ~2B.
    Est. Memory: High. Peak ~54 GB at BS=1.
    Nodes (GBS=512): 64
    Nodes (GBS=1024): 128
    """
    config = get_base_config()

    # XL Model Specs
    config.net.hidden_dim = 1536
    config.net.num_blocks = 24
    config.diffusion.time_embed_dim = 1536
    config.net.norm_cfg.normalized_shape = 1536

    # Batch Size
    # Large-128 fits BS=2. XL is ~3.5x heavier. BS=1 is the only hope.
    config.dataset.batch_size = 1

    # Resolution
    config.dataset.final_image_size = 128
    config.diffusion.cosine_schedule_image_resolution = 128
    config.diffusion.cosine_schedule_noise_res_high = 128
    config.diffusion.cosine_schedule_noise_res_low = 64

    # Tags
    config.wandb.job_group = "sweep_xl_128"

    return config
