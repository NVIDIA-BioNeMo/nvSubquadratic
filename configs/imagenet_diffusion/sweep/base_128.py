from configs.imagenet_diffusion.ccnn_12_768_hyena_qknorm import get_config as get_base_config


def get_config():
    """Base model with 128x128 resolution.

    Est. Param count: ~233M
    Est. Memory: Medium. Peak ~70 GB at BS=8.
    Nodes (GBS=512): 8
    Nodes (GBS=1024): 16
    """
    config = get_base_config()

    # Base Model Specs
    config.net.hidden_dim = 768
    config.net.num_blocks = 12

    # Batch Size (Optimized for 80GB: Fits 8)
    config.dataset.batch_size = 8

    # Resolution
    config.dataset.final_image_size = 128
    config.diffusion.cosine_schedule_image_resolution = 128
    config.diffusion.cosine_schedule_noise_res_high = 128
    config.diffusion.cosine_schedule_noise_res_low = 64

    # Tags
    config.wandb.job_group = "sweep_base_128"

    return config
