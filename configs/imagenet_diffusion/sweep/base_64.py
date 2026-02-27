from configs.imagenet_diffusion.ccnn_12_768_hyena_qknorm import get_config as get_base_config


def get_config():
    """Base model with 64x64 resolution.

    Est. Param count: ~233M
    Est. Memory: Low. Peak ~64 GB at BS=32.
    Nodes (GBS=512): 2
    Nodes (GBS=1024): 4
    """
    config = get_base_config()

    # Base Model Specs (Defaults)
    config.net.hidden_dim = 768
    config.net.num_blocks = 12

    # Batch Size (Fits 32 on 80GB)
    config.dataset.batch_size = 32

    # Resolution
    config.dataset.final_image_size = 64
    config.diffusion.cosine_schedule_image_resolution = 64
    config.diffusion.cosine_schedule_noise_res_high = 64
    config.diffusion.cosine_schedule_noise_res_low = 32

    # Tags
    config.wandb.job_group = "sweep_base_64"

    return config
