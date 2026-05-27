.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. currentmodule:: nvsubquadratic

Parallel
========

Context-parallel communication primitives (zigzag splits / all-to-all)
shared by the mixer and conv modules above.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~parallel.utils.init_parallel_state
   ~parallel.utils.zigzag_split_across_group_ranks
   ~parallel.utils.zigzag_gather_from_group_ranks
   ~parallel.utils.setup_rank0_logging

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~parallel.a2a_comms.AllToAllSingleFunction
