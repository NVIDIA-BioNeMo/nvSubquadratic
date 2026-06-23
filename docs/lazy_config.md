# The lazy-config system

Every training run in this repository is described, end to end, by a single
Python config file. The network architecture, the dataset, the optimizer,
the Lightning wrapper, the schedule — all of it is data, not code wired
together at the call site. The mechanism that makes this work is the
**lazy-config** system in {py:mod}`nvsubquadratic.lazy_config`: a tiny
deferred-instantiation layer (think of it as a few-hundred-line stand-in for
Hydra / detectron2 lazy configs) that lets you *declare* an object — what
class to build and with what arguments — without *building* it yet.

This page explains why we do it this way, how the machinery works, and walks
through the patterns we actually use in the
[`examples/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples)
configs.

## Why configs?

We are moving all model building into config files on purpose. Three
properties motivate it:

- **One config per experiment makes research backtrackable.** A run is not a
  pile of CLI flags and a commit hash you have to reconstruct months later —
  it is a file. The file *is* the experiment. Every architectural choice
  (kernel size, number of blocks, whether QK-norm is on, which mixer) lives
  in one place, under version control, next to the runs it produced. To
  reproduce a result you point `run.py` at the same file; to understand what
  a run did you read one file top to bottom.

- **Base configs + overrides make ablations cheap and honest.** You write a
  base config once and define each ablation as a small file that *overwrites
  parts of it*. The dropout sweep, the mixer swap, the size variants — each
  is its own file, so every run still has a complete, self-contained config.
  There is no "remember to also pass `--dropout 0.1`" footgun: the ablation
  is the file. This is exactly how the
  [`examples/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples)
  configs are organised (see
  [the base-config and ablation pattern](#the-base-config-and-ablation-pattern)
  below).

- **The full architecture is serialisable and diffable.** Because a config
  is plain data (an OmegaConf tree of `__target__` dicts), it can be dumped
  to YAML, logged to W&B, printed as a tree, and *diffed* against another
  run. "What changed between run A and run B?" becomes a text diff of two
  configs.

The cost is that the config files look a little unusual the first time you
see one — deeply nested `LazyConfig(...)` calls with `"${...}"` strings
sprinkled through them. The rest of this page makes that syntax legible.

## The core idea: declare now, build later

A {py:class}`~nvsubquadratic.lazy_config.LazyConfig` wraps a target class or
callable. *Calling* it with keyword arguments does **not** construct the
object — it produces an OmegaConf `DictConfig` carrying a `__target__` key
plus the arguments:

```python
from nvsubquadratic.lazy_config import LazyConfig, instantiate
import torch

cfg = LazyConfig(torch.nn.LayerNorm)(normalized_shape=768, eps=1e-6)
# cfg is a DictConfig:
# {"__target__": "torch.nn.LayerNorm", "normalized_shape": 768, "eps": 1e-6}

norm = instantiate(cfg)  # NOW the LayerNorm is actually built
isinstance(norm, torch.nn.LayerNorm)  # True
```

Two steps, deliberately separated:

1. **Declare** — `LazyConfig(target)(**kwargs)` records *what* to build. No
   import of heavy framework code is forced at this point; the target can
   even be a dotted string like `"torch.nn.LayerNorm"`.
1. **Build** — {py:func}`~nvsubquadratic.lazy_config.instantiate` resolves
   `__target__` via `importlib`, processes the arguments, and calls the
   target.

That separation is the whole trick. Between declare and build, the config is
just data you can edit, override, interpolate, serialise, and diff.

## Nesting

Configs nest. A `LazyConfig` result can be passed as an argument to another
`LazyConfig`, so an entire module tree is one expression:

```python
block = LazyConfig(ResidualBlock)(
    sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
        hidden_dim=160,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(...),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(...),
        ),
    ),
    mlp_cfg=LazyConfig(MLP)(dim=160, activation="glu"),
)
```

When you `instantiate` the outer config, nested configs are resolved
recursively. There is one subtlety worth knowing
([deferred vs. eager nesting](#deferred-vs-eager-instantiation) below): by
default, nested **`nn.Module`** configs are passed *as config* to their
parent rather than pre-built, so each module constructs its own children and
can inspect or modify their configs first. Non-module callables (e.g. weight
init factories) are instantiated eagerly.

## Interpolation: `"${...}"`

Configs are OmegaConf trees, so any value can reference another value by its
dotted path. We use this constantly so that a dimension is written *once* and
flows everywhere:

```python
config.net = LazyConfig(ClassificationResNet)(
    hidden_dim=160,
    data_dim=2,
    in_proj_cfg=LazyConfig(torch.nn.Linear)(
        in_features="${net.in_channels}",
        out_features="${net.hidden_dim}",  # tracks net.hidden_dim
    ),
    norm_cfg=LazyConfig(torch.nn.LayerNorm)(
        normalized_shape="${net.hidden_dim}",
    ),
    ...,
)
```

Change `hidden_dim` in one place and every `"${net.hidden_dim}"` follows.
Interpolations are resolved at instantiation / override time, not when the
config is declared, which is why they survive being edited and overridden.

You can also reference across top-level sections — e.g. a kernel's cache
length tracks the dataset's canvas size:

```python
L_cache = "${dataset.canvas_size}"
```

### Inline arithmetic

Two small conveniences let you do math inside configs:

- **Plain arithmetic strings** are evaluated by `instantiate`. A value like
  `"3 * ${net.hidden_dim}"` resolves the interpolation and then evaluates
  the arithmetic (`"3 * 160"` → `480`). We use this for the Hyena short
  conv, which operates on a 3×-width tensor:

  ```python
  short_conv_cfg = LazyConfig(torch.nn.Conv2d)(
      in_channels="3 * ${net.hidden_dim}",
      out_channels="3 * ${net.hidden_dim}",
      groups="3 * ${net.hidden_dim}",
      ...,
  )
  ```

- **The `${eval:...}` resolver** handles arithmetic in CLI overrides and
  trainer interpolations, e.g.
  `"${eval:'${trainer.samples_per_epoch} // (${train.batch_size} * 2)'}"`.
  Only arithmetic on literal numbers is permitted — no function calls or
  attribute access — so configs stay safe to load. (The `${eval:...}` resolver
  additionally allows `**`; plain arithmetic strings support `+ - * / // %`
  but not power.)

## `PLACEHOLDER`: a hole to be filled later

{py:data}`~nvsubquadratic.lazy_config.PLACEHOLDER` is a sentinel marking a
field whose value isn't known yet at declaration time. It plays two roles:

1. **A required slot to be filled later.** A base config marks a field
   `PLACEHOLDER` to say "this *must* be supplied before the object is built."
   The hole is filled in one of two ways:

   - *By an experiment file, before running.* The spatial-recall base config
     sets `sequence_mixer_cfg=PLACEHOLDER`; each ablation file then asserts the
     hole is still empty and plugs in a mixer — a self-documenting contract:

     ```python
     block_cfg = LazyConfig(ResidualBlock)(
         sequence_mixer_cfg=PLACEHOLDER, ...  # filled in by the experiment file
     )
     ```

     ```python
     assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
     config.net.block_cfg.sequence_mixer_cfg = get_hyena_mixer_cfg()
     ```

   - *By code, at build time.* The optimizer is declared with
     `params=PLACEHOLDER` because the parameters don't exist until the network
     is constructed. The Lightning wrapper fills the slot when it builds the
     optimizer: `_build_optimizer` in
     {py:mod}`experiments.lightning_wrappers.base_lightning_wrapper` resolves
     the config to a dict and overwrites `params` with the real parameter
     groups. (Note this path constructs the optimizer directly rather than
     through `instantiate`.)

1. **A "don't build me yet" guard.** While `instantiate` walks an argument
   tree, any *nested* config that still contains a `PLACEHOLDER` is left as a
   config dict rather than constructed, so a half-specified subtree is never
   handed to a constructor mid-build — e.g. a `block_cfg` whose
   `sequence_mixer_cfg` hole hasn't been filled is passed through as config
   instead of built. This guard is a check on *nested* configs only: a bare
   top-level value like the optimizer's `params=PLACEHOLDER` is not itself
   guarded, which is why that slot is filled in by code (role 1) before the
   object is built.

## A full example, end to end

[`examples/mnist_classification/ccnn_4_160_hyena_rope_qknorm.py`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/mnist_classification/ccnn_4_160_hyena_rope_qknorm.py)
is a complete, self-contained config. Every config file exposes a
`get_config()` that returns an
{py:class}`~experiments.default_cfg.ExperimentConfig`. The skeleton:

```python
def get_config() -> ExperimentConfig:
    config = ExperimentConfig()  # typed dataclass of sensible defaults

    # 1. Dataset — a LazyConfig pointing at a LightningDataModule
    config.dataset = LazyConfig(MNISTDataModule)(
        data_dir=".data/mnist",
        batch_size=BATCH_SIZE,
        seed=config.seed,
        task="classification",
        ...,
    )

    # 2. Network — one big nested LazyConfig tree (the architecture)
    config.net = LazyConfig(ClassificationResNet)(
        in_channels=INPUT_CHANNELS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(
            in_features="${net.in_channels}", out_features="${net.hidden_dim}"
        ),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(...),
            ),
            mlp_cfg=LazyConfig(MLP)(dim="${net.hidden_dim}", activation="glu", ...),
            ...,
        ),
    )

    # 3. Lightning wrapper, optimizer (note params=PLACEHOLDER)
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 4. Plain typed sub-configs for training / schedule / logging
    config.train = TrainConfig(batch_size="${dataset.batch_size}", iterations=...)
    config.scheduler = SchedulerConfig(
        name="cosine", total_iterations="${train.iterations}"
    )
    config.wandb = WandbConfig(job_group="mnist_classification", ...)

    return config
```

Notice the mix: `dataset`, `net`, `lightning_wrapper_class`, and `optimizer`
are **`LazyConfig`s** (objects built lazily), while `train`, `scheduler`, and
`wandb` are **plain typed dataclasses** from
{py:mod}`experiments.default_cfg` (values read directly). You only set what
differs from the defaults.

### How `run.py` consumes it

[`experiments/run.py`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/experiments/run.py)
is the entry point. It:

```bash
PYTHONPATH=. python experiments/run.py \
    --config examples/mnist_classification/ccnn_4_160_hyena_rope_qknorm.py \
    dataset.batch_size=64 optimizer.lr=3e-4
```

1. Loads the file and calls `get_config()`.

1. Applies the `key=value` CLI overrides (after checking none of them clobber
   an interpolated field — see below).

1. Builds the objects exactly when needed:

   ```python
   datamodule = instantiate(config.dataset)
   network = instantiate(config.net)
   model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)
   ```

The config tree is also serialised and logged to W&B and printed to the
console as a Rich tree, so the exact specification of every run is captured.

## The base-config and ablation pattern

This is where the design pays off, and it is worth studying the
[`examples/spatial_recall_2d/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples/spatial_recall_2d)
directory to see it in practice.
Instead of copy-pasting a 150-line config per ablation, we factor the shared
structure into helper functions and keep each experiment file tiny.

[`spatial_recall_2d/base_config.py`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/spatial_recall_2d/base_config.py)
exposes `base_experiment_config(...)` which returns a fully-formed
`ExperimentConfig` with the network, optimizer, scheduler, and callbacks all
wired — but with the sequence mixer and the dataset left as `PLACEHOLDER`.
[`mixer_defaults.py`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/examples/spatial_recall_2d/mixer_defaults.py)
provides `get_hyena_mixer_cfg()`, `get_mamba_mixer_cfg()`, and
`get_attention_mixer_cfg()` — each a `LazyConfig` for one mixer family.

An individual ablation is then *small and complete*:

```python
# examples/spatial_recall_2d/emnist_regression_color_selection/ccnn_hyena_s.py
def get_config() -> ExperimentConfig:
    config = spatial_recall_2d_base_experiment_config(
        in_channels=3,
        out_channels=1,
        hidden_dim=256,  # the "S" size
        training_iterations=20_000,
        wandb_job_group="spatial_recall_2d_emnist_color_selection_s",
    )

    # Fill the mixer hole — swap this line for get_mamba/attention to ablate
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = get_hyena_mixer_cfg()

    # Fill the dataset hole
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_2d_dataset_config(
        target_size=16,
        canvas_size=64,
        batch_size=64,
        use_colored_frames=True,
        num_items=4,
        placement="random",
        with_mask=False,
        normalize_input=True,
    )
    return config
```

To ablate the mixer, you change one line and the filename — `ccnn_hyena_s.py`
→ `ccnn_mamba_xs.py`. To ablate size, you change `hidden_dim`. Each variant
is its own file, so each run still carries a *complete* config, yet the diff
between any two variants is one or two lines. That is the backtrackable,
ablation-friendly workflow the config system exists to enable.

Because helpers return `LazyConfig` trees built from `"${...}"`
interpolations, the swapped-in mixer automatically picks up `hidden_dim`,
`data_dim`, `num_blocks`, and `canvas_size` from the surrounding config — you
never restate them.

## Overriding from the command line

Any field can be overridden with `key=value` positional arguments to
`run.py`. Values are auto-typed (`int` → `float` → `None` → `bool` → tuple →
list → `str`), and dotted paths reach into nested configs:

```bash
PYTHONPATH=. python experiments/run.py --config <file> \
    train.batch_size=32 \
    optimizer.lr=3e-4 \
    net.hidden_dim=256
```

Two guardrails are worth knowing:

- **You cannot override an interpolated field.** If a field's current value
  is a `"${...}"` string, overriding it directly is rejected
  (`verify_no_interpolator_overwrites`). Override the *source* of the
  interpolation instead — e.g. set `net.hidden_dim=256`, not the dozen places
  that read `"${net.hidden_dim}"`.
- **Add genuinely new keys with `+`.** `key=value` requires the key to exist
  (typo protection); Hydra-style `+key=value` force-adds it.

Overrides also feed the deterministic run name, so a sweep over
`optimizer.lr` produces distinctly named, individually reproducible runs.

## Two details that explain the rest

### Deferred vs. eager instantiation

When `instantiate` walks the argument tree it decides, per nested config,
whether to build it now or pass it through as config:

- **`nn.Module` subclasses are passed through as config** (a `DictConfig`),
  not pre-built. The parent module receives the child's config and
  constructs it itself. This lets a network inspect or tweak block configs
  (injecting `drop_path_rate`, reading `"${net.num_blocks}"`) before building
  them.
- **Non-module callables are instantiated eagerly** — e.g. a weight-init
  factory like `partial_wang_init_fn_with_num_layers(num_layers=...)` is
  resolved to a function and handed to the module ready to use.

Passing `recursive_instantiate=True` overrides this and builds everything
top-down; the default (`False`) is what the module tree relies on.

### Serialisation

{py:func}`~nvsubquadratic.lazy_config.save_config` /
{py:func}`~nvsubquadratic.lazy_config.load_config` round-trip a config to
YAML via OmegaConf, and `config_to_dict` (used by `run.py`) flattens the
whole tree — `LazyConfig`s, dataclasses, function references — into a
JSON-serialisable dict for W&B and the console tree. This is what makes a run
fully recoverable from its logged config.

## Mental model / cheat sheet

- `LazyConfig(Target)(**kwargs)` → a config dict; nothing is built yet.
- `instantiate(cfg)` → the actual object.
- Nest `LazyConfig`s to describe a whole module tree as one expression.
- `"${a.b.c}"` references another field; resolved at build/override time.
- `"3 * ${net.hidden_dim}"` does inline arithmetic after interpolation.
- `PLACEHOLDER` marks a hole that must be filled and blocks premature builds.
- One file per experiment; a base helper + a one-line swap per ablation.
- Override with `key=value` on the CLI; never override a `"${...}"` field
  directly — change its source.

For the bigger picture of where configs sit in the stack, see
{doc}`architecture`; for runnable recipes, the
[`examples/`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic/tree/main/examples)
configs.

```

```
