"""Post-creation override: add register-based FiLM conditioning to Hyena blocks.

Importable helper applied to a config whose ``net`` was built by
:func:`build_hybrid_net` from ``_base_config``.  Adds two pieces:

1. **Register pooling on every Hyena block** — ``RegisterPooling`` extracts
   the register tokens from the normalized input of each Hyena block and
   pools them into a single conditioning vector via a learnable
   softmax-weighted average.  The pooled vector has shape ``[B, hidden_dim]``.

2. **FiLM-conditioned SIREN kernel** — a ``KernelFiLMGenerator`` is attached
   to every Hyena block's kernel.  The generator maps the pooled vector to
   per-layer ``(gamma, beta)`` pairs that modulate every SIREN hidden layer
   (and the positional-embedding sine when ``film_after_pos_embed=True``).

The defaults below match the v3 ``peeaqdkq`` LAMB+FiLM baseline (best
ImageNet pretraining FiLM result, **81.83% test**), with two deliberate
deviations that we found to be more general:

- ``film_after_pos_embed = True`` (3 FiLM layers — modulates the first
  sine in addition to the two hidden linears).  The v3 best had this off,
  but v4 ablations and the user's recent intuition both prefer modulating
  the positional embedding.
- ``num_registers = 8`` is **fixed across patch sizes** so the FiLM
  conditioning shape is identical at p=4/8/16/32 and the helper does not
  depend on the spatial layout.

The mask + kernel choice is left untouched — call this helper *after*
:func:`apply_learnable_omega_blockdiag_overrides` (or any other kernel
override) to layer FiLM on top.
"""

from examples.vit5_imagenet.vit5_hybrid._base_config import (
    KERNEL_MLP_HIDDEN_DIM,
)
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling


# ─── FiLM defaults (matches v3 ``peeaqdkq`` winner, with pos-embed FiLM on) ──
FILM_HIDDEN_DIM = 64
FILM_NUM_LAYERS = 3  # 2 hidden + 1 on the positional sine
FILM_AFTER_POS_EMBED = True
FILM_INIT_TYPE = "identity"  # γ=1, β=0 at init — start from no-modulation baseline.

# Dedicated optimizer WD group for FiLM weights (biases always excluded).
# ``5e-3`` matches the best v3 LAMB+FiLM pretraining run (peeaqdkq, 81.83%).
# Avoid the ``True`` (no-WD) setting: a 25-epoch ft sweep (r14_ramix_sr_fwdnone)
# showed FiLM weight norms exploding to ~2.82 without any WD pressure.
FILM_WEIGHT_DECAY = 5e-3

# Fix the register count across patch sizes.  RegisterPooling produces a
# ``[B, hidden_dim]`` cond vector regardless of how many registers it
# averages, so changing ``num_registers`` does not change FiLM ``cond_dim``.
NUM_REGISTERS_DEFAULT = 8


def apply_film_overrides(
    config,
    *,
    num_registers: int = NUM_REGISTERS_DEFAULT,
    film_hidden_dim: int = FILM_HIDDEN_DIM,
    num_film_layers: int = FILM_NUM_LAYERS,
    film_after_pos_embed: bool = FILM_AFTER_POS_EMBED,
    film_init_type: str = FILM_INIT_TYPE,
    film_weight_decay: float = FILM_WEIGHT_DECAY,
) -> None:
    """Add register-pooling + FiLM conditioning to every Hyena block.

    Mutates ``config.net`` in place.  Must be called *after*
    :func:`build_hybrid_net` and after any kernel-replacement override
    (e.g. :func:`apply_learnable_omega_blockdiag_overrides`).

    Args:
        config: ``ExperimentConfig`` whose ``net`` was built by
            :func:`build_hybrid_net`.
        num_registers: Number of learnable register tokens.  Fixed across
            patch sizes so FiLM ``cond_dim`` does not change with resolution.
        film_hidden_dim: Bottleneck width of the FiLM generator MLP.
        num_film_layers: Total FiLM (γ, β) pairs.  With
            ``film_after_pos_embed=True`` and ``num_layers=3`` SIREN, set
            this to ``3`` (1 for the pos-embed sine + 2 hidden linears).
        film_after_pos_embed: If True, the first FiLM pair modulates the
            output of the positional-embedding sine.  Requires
            ``embedding_dim == mlp_hidden_dim`` in the SIREN kernel.
        film_init_type: ``"identity"`` (γ=1, β=0) or ``"small_random"``.
        film_weight_decay: Weight decay applied to FiLM **weight**
            parameters via a dedicated optimizer group.  FiLM biases are
            always excluded from WD regardless of this value.
    """
    config.net.num_registers = num_registers

    # cond_dim = hidden_dim because RegisterPooling produces a single
    # ``[B, hidden_dim]`` vector via a learnable softmax-weighted average.
    cond_dim = "${net.hidden_dim}"

    film_cfg = LazyConfig(KernelFiLMGenerator)(
        cond_dim=cond_dim,
        kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_film_layers=num_film_layers,
        film_hidden_dim=film_hidden_dim,
        no_weight_decay=film_weight_decay,
        init_type=film_init_type,
    )

    block = config.net.layer_types["H"]
    block.register_pooling_cfg = LazyConfig(RegisterPooling)(num_registers=num_registers)
    block.num_registers = num_registers

    kernel_cfg = block.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg
    kernel_cfg.film_cfg = film_cfg
    kernel_cfg.film_after_pos_embed = film_after_pos_embed
