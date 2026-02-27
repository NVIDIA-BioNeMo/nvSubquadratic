from configs.imagenet_diffusion.ccnn_12_768_hyena_qknorm import get_config as get_base_config


def get_config():
    """Base Model, 64x64, Epsilon Prediction Objective.

    Est. Param count: ~233M
    Est. Memory: Low.
    Nodes (GBS=512): 4
    Nodes (GBS=1024): 8
    """
    config = get_base_config()

    # Base Model Specs (Same as base_64)
    config.net.hidden_dim = 768
    config.net.num_blocks = 12

    # Prediction Type
    config.diffusion.prediction_type = "epsilon"

    # Batch Size
    config.dataset.batch_size = 32

    # Resolution
    config.dataset.final_image_size = 64
    config.diffusion.cosine_schedule_image_resolution = 64
    config.diffusion.cosine_schedule_noise_res_high = 64
    config.diffusion.cosine_schedule_noise_res_low = 32

    # Tags
    config.wandb.job_group = "sweep_base_64_epsilon"

    return config
