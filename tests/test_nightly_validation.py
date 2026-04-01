"""Nightly validation tests for best ViT-5 models on ImageNet-1k.

Downloads the "best" checkpoint for each model variant from W&B and runs
a full test pass on ImageNet to ensure accuracy has not regressed.

Prerequisites (all provided by the SLURM container):
  - GPU with CUDA
  - NVIDIA DALI (``nvidia.dali``)
  - ImageNet data at ``/shared/data/image_datasets/imagenet``
  - ``WANDB_API_KEY`` environment variable

Run:
    source .env && PYTHONPATH=. python -m pytest tests/test_nightly_validation.py -m nightly -v -o addopts=""

See tests/README.md for all test suites, markers, and SLURM usage.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl

from experiments.default_cfg import AutoResumeConfig, StartFromCheckpointConfig
from experiments.utils.checkpointing import (
    StripCompiledPrefix,
    download_checkpoint,
    load_checkpoint_state_dict,
)
from nvsubquadratic.lazy_config import LazyConfig, instantiate


# ─── Constants ───────────────────────────────────────────────────────────────────

WANDB_ENTITY = "implicit-long-convs"
WANDB_PROJECT = "nvsubquadratic"
IMAGENET_PATH = "/shared/data/image_datasets/imagenet"

# ─── Skip conditions ────────────────────────────────────────────────────────────

_SKIP_REASONS: list[tuple[bool, str]] = [
    (not torch.cuda.is_available(), "CUDA not available"),
    ("WANDB_API_KEY" not in os.environ, "WANDB_API_KEY not set (run `source .env`)"),
    (not os.path.isdir(IMAGENET_PATH), f"ImageNet not found at {IMAGENET_PATH}"),
]

_skip_nightly = pytest.mark.skipif(
    any(cond for cond, _ in _SKIP_REASONS),
    reason="; ".join(reason for cond, reason in _SKIP_REASONS if cond),
)

try:
    import nvidia.dali  # noqa: F401

    _has_dali = True
except ImportError:
    _has_dali = False

_skip_no_dali = pytest.mark.skipif(not _has_dali, reason="NVIDIA DALI not installed")


# ─── State dict helpers ──────────────────────────────────────────────────────────

_SIREN_SEQUENTIAL_RE = re.compile(r"(\.kernel\.kernel_network)\.(\d+)\.(weight|bias)$")


def _remap_siren_sequential_to_modulelist(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename ``kernel_network.{2*i}`` keys to ``hidden_linears.{i}``.

    Older SIRENKernelND used an ``nn.Sequential`` called ``kernel_network``
    where even-indexed entries are Linear layers and odd-indexed are Sine
    activations.  The current code stores only the Linears in an
    ``nn.ModuleList`` called ``hidden_linears``.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        m = _SIREN_SEQUENTIAL_RE.search(key)
        if m:
            seq_idx = int(m.group(2))
            linear_idx = seq_idx // 2
            new_key = key[: m.start()] + f".kernel.hidden_linears.{linear_idx}.{m.group(3)}"
            remapped[new_key] = val
        else:
            remapped[key] = val
    return remapped


# ─── Shared validation helper ───────────────────────────────────────────────────


def _run_validation_check(
    get_config_fn,
    wandb_run_id: str,
    min_test_acc: float,
    *,
    compile_compatible_fftconv: bool = False,
    remap_siren_keys: bool = False,
) -> None:
    """Download a checkpoint from W&B and validate it against ImageNet.

    Args:
        get_config_fn: Callable that returns an ``ExperimentConfig`` with the
            model architecture and dataset definition.
        wandb_run_id: Short W&B run ID (e.g. ``"peeaqdkq"``).
        min_test_acc: Minimum acceptable ``test/acc`` (0-1 scale).
        compile_compatible_fftconv: If ``True``, enable the compile-compatible
            FFT conv path (needed for Hyena + FiLM models).
        remap_siren_keys: If ``True``, rename old ``kernel_network`` keys to
            ``hidden_linears`` for checkpoints trained before the SIREN
            refactor.
    """
    # 1. Load the training config and override for validation-only
    config = get_config_fn()
    config.train.do = False
    config.debug = True
    config.compile = False
    config.dataset.local_staging_dir = None  # read from /shared, no scratch needed
    config.autoresume = AutoResumeConfig(enabled=False)
    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        run_path=f"{WANDB_ENTITY}/{WANDB_PROJECT}/{wandb_run_id}",
        alias="best",
        strict=True,
        callbacks=[LazyConfig(StripCompiledPrefix)()],
    )

    # 2. Optional: enable compile-compatible FFT path for Hyena models
    if compile_compatible_fftconv:
        import nvsubquadratic.ops.fftconv as _fftconv

        _fftconv.COMPILE_COMPATIBLE = True

    # 3. Seed for reproducibility
    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # 4. Instantiate datamodule
    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    # 5. Instantiate network and lightning wrapper
    network = instantiate(config.net)
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)

    # 6. Download checkpoint and load weights
    run_path = config.start_from_checkpoint.run_path
    ckpt_path = download_checkpoint(run_path=run_path, alias="best")
    state_dict = load_checkpoint_state_dict(ckpt_path)

    # Remap old SIREN Sequential keys before StripCompiledPrefix, because
    # align_compiled_keys needs the key *names* (minus _orig_mod) to match
    # the model — so the kernel_network -> hidden_linears rename must come first.
    if remap_siren_keys:
        state_dict = _remap_siren_sequential_to_modulelist(state_dict)

    # Apply StripCompiledPrefix callback
    strip = StripCompiledPrefix()
    state_dict = strip(state_dict=state_dict, model=model)

    model.load_state_dict(state_dict, strict=True)

    # 7. Create a minimal trainer (no wandb, no checkpointing)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # 8. Run test pass
    results = trainer.test(model, datamodule=datamodule)

    # 9. Assert accuracy
    test_acc = results[0]["test/acc"]
    assert test_acc >= min_test_acc, (
        f"test/acc={test_acc:.4f} below threshold {min_test_acc:.4f} for W&B run {wandb_run_id}"
    )


# ─── Nightly test functions ─────────────────────────────────────────────────────


@pytest.mark.nightly
@_skip_nightly
@_skip_no_dali
def test_validate_film_model() -> None:
    """Validate the best FiLM-conditioned Hyena model (81.8% test/acc).

    Config: v3/vit5_small_pretrain_hyena_cls_row_gated_film_ema
    W&B run: peeaqdkq
    """
    from examples.vit5_imagenet.v3.vit5_small_pretrain_hyena_cls_row_gated_film_ema import (
        get_config,
    )

    _run_validation_check(
        get_config_fn=get_config,
        wandb_run_id="peeaqdkq",
        min_test_acc=0.813,
        compile_compatible_fftconv=True,
    )


@pytest.mark.nightly
@_skip_nightly
@_skip_no_dali
def test_validate_attention_model() -> None:
    """Validate the best attention baseline model (82.2% test/acc).

    Config: v2/vit5_small_pretrain_attention_ema
    W&B run: 44or24g1
    """
    from examples.vit5_imagenet.v2.vit5_small_pretrain_attention_ema import (
        get_config,
    )

    _run_validation_check(
        get_config_fn=get_config,
        wandb_run_id="44or24g1",
        min_test_acc=0.817,
    )


@pytest.mark.nightly
@_skip_nightly
@_skip_no_dali
def test_validate_gap_model() -> None:
    """Validate the best GAP Hyena model (81.5% test/acc).

    Config: v2/vit5_small_pretrain_hyena_gap_apex_gated_ema
    W&B run: tcji9tfx
    """
    from examples.vit5_imagenet.v2.vit5_small_pretrain_hyena_gap_apex_gated_ema import (
        get_config,
    )

    _run_validation_check(
        get_config_fn=get_config,
        wandb_run_id="tcji9tfx",
        min_test_acc=0.810,
        compile_compatible_fftconv=True,
        remap_siren_keys=True,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "nightly"])
