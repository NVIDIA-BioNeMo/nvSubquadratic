from examples.imagenet_diffusion.ccnn_12_768_hyena_qknorm import get_config as get_base_config


def get_config():
    """Base Model (256x256).

    Est. Param count: ~233M.
    Est. Memory: High. Peak ~41 GB at BS=1.
    Nodes (GBS=512): 64
    Nodes (GBS=1024): 128
    """
    config = get_base_config()

    # Base Model Specs
    config.net.hidden_dim = 768
    config.net.num_blocks = 12
    config.diffusion.time_embed_dim = 768

    # Batch Size
    config.dataset.batch_size = 1

    # Resolution
    config.dataset.final_image_size = 256
    config.diffusion.cosine_schedule_image_resolution = 256
    config.diffusion.cosine_schedule_noise_res_high = 256
    config.diffusion.cosine_schedule_noise_res_low = 128

    # Tags
    config.wandb.job_group = "sweep_base_256"

    return config
