.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. currentmodule:: nvsubquadratic

Core
====

Top-level utilities: the lazy-instantiation system that powers every
config file, weight-init helpers, QK-norm and rotary embedding
primitives, the QuACK-kernel capability probe, FID computation, and
testing helpers.

Lazy configuration
------------------

The lazy-instantiation system lets configs declare ``_target_``-shaped
specs that are deferred until ``instantiate`` is called.  This is what
every experiment config and most ``modules/`` constructors rely on.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~lazy_config.LazyConfig

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~lazy_config.instantiate

Initialisation helpers
----------------------

Truncated-normal and Wang/SmallInit factories used by SIREN, MLP, and
projection layers.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~utils.init.trunc_normal_init
   ~utils.init.trunc_normal_init_factory
   ~utils.init.small_init
   ~utils.init.wang_init
   ~utils.init.partial_wang_init_fn_with_num_layers

QK normalization & rotary position embeddings
---------------------------------------------

Shared building blocks consumed by the attention and Hyena mixers.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~utils.qk_norm.L2Norm

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~utils.qk_norm.apply_qk_norm
   ~utils.rope.apply_rope_1d_bhl
   ~utils.rope.apply_rope_2d_bhl
   ~utils.rope.apply_rope_3d_bhl
   ~utils.rope.apply_rope_1d_blh
   ~utils.rope.apply_rope_2d_blh
   ~utils.rope.apply_rope_3d_blh
   ~utils.rope.construct_rope_1d_cache_bhl
   ~utils.rope.construct_rope_2d_cache_bhl
   ~utils.rope.construct_rope_3d_cache_bhl
   ~utils.rope.construct_rope_1d_cache_blh
   ~utils.rope.construct_rope_2d_cache_blh
   ~utils.rope.construct_rope_3d_cache_blh

QuACK capability probe
----------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~utils.quack_utils.cuda_supports_quack

Metrics
-------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~metrics.cleanfid.compute_folder_fid

Testing helpers
---------------

Small numerical-comparison helpers used by the test suite.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~testing.utils.compute_relative_error
