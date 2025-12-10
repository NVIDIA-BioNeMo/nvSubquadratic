"""Verify config instantiation."""

from examples.text_pretraining.zyda_1d_hyena import get_config
from nvsubquadratic.lazy_config import instantiate


def test_config():
    """Test config instantiation."""
    cfg = get_config()

    print("Instantiating dataset...")
    _ = instantiate(cfg.dataset)
    print("Dataset instantiated.")

    print("Instantiating network...")
    # We need to resolve placeholders if any, but LazyConfig handles it if passed correctly.
    # However, LazyConfig usually resolves ${...} references when instantiated if they are within the same config structure?
    # No, OmegaConf resolves them. We need to convert to OmegaConf first if we rely on interpolation.
    # But here we are just instantiating individual components.
    # The config uses ${net.hidden_dim} which refers to itself.
    # We might need to manually resolve or just check if the structure is valid.

    # Let's try to instantiate the network directly.
    # Note: ${net.hidden_dim} won't be resolved if we just instantiate cfg.net in isolation unless we use OmegaConf.
    # But let's see if we can just check the config structure first.

    # Actually, let's try to use the `instantiate` function as intended.
    # If we can't fully instantiate due to interpolation, we can at least check imports.

    # To properly test interpolation, we should convert to DictConfig/ListConfig.
    from omegaconf import OmegaConf

    cfg_omega = OmegaConf.structured(cfg)
    OmegaConf.resolve(cfg_omega)

    # Now instantiate net
    print("Instantiating network (via OmegaConf)...")
    net = instantiate(cfg_omega.net)
    print("Network instantiated.")

    print("Instantiating wrapper...")
    _ = instantiate(cfg_omega.lightning_wrapper_class, network=net, cfg=cfg_omega)
    print("Wrapper instantiated.")


if __name__ == "__main__":
    test_config()
