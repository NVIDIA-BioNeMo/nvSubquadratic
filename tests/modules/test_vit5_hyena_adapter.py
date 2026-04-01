import pytest
import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapterND
from nvsubquadratic.networks.vit5_general_purpose import ViT5GeneralPurposeNet


class DummyMixer(nn.Module):
    def forward(self, x, **kwargs):
        return x


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def test_vit5_hyena_adapter_nd_shapes():
    B, C = 2, 16

    # 1D Case
    grid_shape_1d = (10,)
    cfg_1d = LazyConfig(DummyMixer)
    adapter_1d = ViT5HyenaAdapterND(cfg_1d, grid_shape_1d)
    x_1d = torch.randn(B, 10, C)  # 10 tokens
    out_1d = adapter_1d(x_1d)
    assert out_1d.shape == (B, 10, C)

    # 2D Case
    grid_shape_2d = (8, 8)
    cfg_2d = LazyConfig(DummyMixer)
    adapter_2d = ViT5HyenaAdapterND(cfg_2d, grid_shape_2d)

    # 64 patches + 8 padded prefix tokens (e.g. 4 registers padded to 8) = 72 tokens
    T_2d = 8 + 64
    x_2d = torch.randn(B, T_2d, C)
    out_2d = adapter_2d(x_2d)
    assert out_2d.shape == (B, T_2d, C)

    # 3D Case
    grid_shape_3d = (4, 8, 8)
    cfg_3d = LazyConfig(DummyMixer)
    adapter_3d = ViT5HyenaAdapterND(cfg_3d, grid_shape_3d)

    # 256 patches + 64 padded prefix tokens (e.g. 16 registers padded to 64) = 320 tokens
    T_3d = 64 + 256
    x_3d = torch.randn(B, T_3d, C)
    out_3d = adapter_3d(x_3d)
    assert out_3d.shape == (B, T_3d, C)


@pytest.mark.parametrize("data_dim", [1, 2, 3])
@pytest.mark.parametrize("use_cls_token", [True, False])
@pytest.mark.parametrize("num_registers", [0, 2, 3])
def test_vit5_general_purpose_zero_padding_integration(data_dim, use_cls_token, num_registers):
    # Test whether vit5_general_purpose properly handles spatial alignment padding
    # when passed a non-divisible amount of registers, 0 registers, or standard amounts.
    B = 2
    in_channels = 3
    out_channels = 5
    hidden_dim = 16
    patch_size = 1
    input_size = 8  # 8 // 1 = 8 patches per dimension

    in_proj_cfg = LazyConfig(nn.Linear)
    out_proj_cfg = LazyConfig(nn.Linear)
    block_cfg = LazyConfig(DummyBlock)
    norm_cfg = LazyConfig(nn.LayerNorm)(normalized_shape=hidden_dim)

    net = ViT5GeneralPurposeNet(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=hidden_dim,
        num_blocks=1,
        data_dim=data_dim,
        patch_size=patch_size,
        input_size=input_size,
        num_registers=num_registers,
        in_proj_cfg=in_proj_cfg,
        out_proj_cfg=out_proj_cfg,
        block_cfg=block_cfg,
        norm_cfg=norm_cfg,
        use_cls_token=use_cls_token,
        prepend_registers=True,
    )

    spatial_shape = [input_size] * data_dim
    x = torch.randn(B, *spatial_shape, in_channels)

    out = net({"input": x})

    assert "logits" in out
    # Output should exactly match expected prediction shape
    expected_out_shape = (B, *(input_size // patch_size for _ in range(data_dim)), out_channels)
    assert out["logits"].shape == expected_out_shape


def test_vit5_general_purpose_too_many_registers():
    # Test that passing more register tokens than the plane size throws an error
    in_proj_cfg = LazyConfig(nn.Linear)(in_features=3, out_features=16)
    out_proj_cfg = LazyConfig(nn.Linear)(in_features=16, out_features=5)
    block_cfg = LazyConfig(DummyBlock)
    norm_cfg = LazyConfig(nn.LayerNorm)(normalized_shape=16)

    # For data_dim=2, input_size=8, patch_size=2 -> grid_shape is (4, 4)
    # The slice size (plane width) becomes 4.
    # Therefore, providing 5 registers should trigger an AssertionError.
    with pytest.raises(AssertionError, match="prefix tokens must fit in a single slice"):
        ViT5GeneralPurposeNet(
            in_channels=3,
            out_channels=5,
            hidden_dim=16,
            num_blocks=1,
            data_dim=2,
            patch_size=2,
            input_size=8,
            num_registers=5,  # 5 > 4 (slice size)
            in_proj_cfg=in_proj_cfg,
            out_proj_cfg=out_proj_cfg,
            block_cfg=block_cfg,
            norm_cfg=norm_cfg,
            use_cls_token=False,
        )
