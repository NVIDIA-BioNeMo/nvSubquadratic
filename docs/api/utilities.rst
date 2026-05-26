.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. currentmodule:: nvsubquadratic

Utilities & Metrics
===================

QK normalization, rotary position embeddings, weight-init helpers, and
evaluation metrics.

QK normalization & rotary position embeddings
---------------------------------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~utils.qk_norm.apply_qk_norm
   ~utils.quack_utils.cuda_supports_quack
   ~utils.rope.apply_rope_1d_bhl
   ~utils.rope.apply_rope_2d_bhl
   ~utils.rope.apply_rope_3d_bhl
   ~utils.rope.apply_rope_1d_blh
   ~utils.rope.apply_rope_2d_blh
   ~utils.rope.apply_rope_3d_blh
   ~utils.rope.construct_rope_1d_cache_bhl
   ~utils.rope.construct_rope_2d_cache_bhl
   ~utils.rope.construct_rope_3d_cache_bhl

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~utils.qk_norm.L2Norm

Metrics
-------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~metrics.cleanfid.compute_folder_fid
