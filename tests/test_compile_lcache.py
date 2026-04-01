# TODO: Add license header here

"""Test: does torch.compile work when L_cache=14 and input requires 15 rows?

Simulates the real pipeline:
1. Build model with L_cache=14 and num_registers=14
2. Compile with max-autotune-no-cudagraphs
3. Run forward on a 224x224 image (which gives 15x14 tokens after register prepend)
4. Verify L_cache auto-extends and a second forward is stable
"""

import pytest
import torch


pytest.importorskip("apex")

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def lcache_model():
    """Build a ViT5 model with L_cache=14 and num_registers=14."""
    from examples.vit5_imagenet.v3.gap_film_regs._base import get_config
    from nvsubquadratic.lazy_config import instantiate

    cfg = get_config(
        num_registers=14,
        num_film_layers=3,
        film_after_pos_embed=True,
        reg_init="zeros",
        train_do=False,
    )
    net = instantiate(cfg.net)
    net.eval()
    return net


@pytest.fixture
def compiled_lcache_model(lcache_model):
    """Compile the model with max-autotune-no-cudagraphs and compile-compatible FFT."""
    import nvsubquadratic.ops.fftconv as _fftconv

    _fftconv.COMPILE_COMPATIBLE = True
    return torch.compile(lcache_model, mode="max-autotune-no-cudagraphs")


@pytest.fixture
def dummy_input():
    """A 224x224 image input that produces 15x14 tokens after register prepend."""
    return {"input": torch.randn(1, 224, 224, 3)}


@_requires_cuda
def test_lcache_initial_values(lcache_model):
    """Verify L_cache is 14 on all blocks before any forward pass."""
    for i, block in enumerate(lcache_model.blocks):
        siren = block.sequence_mixer.inner_mixer.mixer.global_conv.kernel
        lc = siren.positional_embedding.L_cache
        assert lc == 14, f"block {i}: expected L_cache=14, got {lc}"


@_requires_cuda
def test_compiled_forward_succeeds(compiled_lcache_model, dummy_input):
    """Compiled forward pass on 224x224 input triggers L_cache auto-extension and succeeds."""
    with torch.no_grad():
        out = compiled_lcache_model(dummy_input)
    assert "logits" in out, "Output missing 'logits' key"
    assert out["logits"].ndim == 2, f"Expected 2D logits, got shape {out['logits'].shape}"


@_requires_cuda
def test_compiled_forward_stable_across_passes(compiled_lcache_model, dummy_input):
    """Two consecutive compiled forwards produce identical logits."""
    with torch.no_grad():
        out1 = compiled_lcache_model(dummy_input)
        out2 = compiled_lcache_model(dummy_input)

    diff = (out1["logits"] - out2["logits"]).abs().max().item()
    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5), (
        f"Logits differ across passes: max_diff={diff:.4e}"
    )
