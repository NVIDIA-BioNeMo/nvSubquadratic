.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. currentmodule:: experiments

Experiments
===========

The ``experiments`` package wires nvSubquadratic modules into reproducible
training pipelines built on PyTorch Lightning.  Each experiment is a
``LazyConfig`` of dataclasses (``ExperimentConfig``)
plus a wrapper subclass that defines the training step.

Entry points
------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~run.parse_args
   ~run.main
   ~trainer.construct_trainer

Configuration dataclasses
-------------------------

The dataclasses below describe the experiment surface.  They are
instantiated via :class:`~nvsubquadratic.lazy_config.LazyConfig` from
the per-experiment config files in ``examples/``.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~default_cfg.ExperimentConfig
   ~default_cfg.TrainConfig
   ~default_cfg.TrainerConfig
   ~default_cfg.SchedulerConfig
   ~default_cfg.WandbConfig
   ~default_cfg.AutoResumeConfig
   ~default_cfg.StartFromCheckpointConfig

Lightning wrappers
------------------

Task-specific wrappers around a common base.  Each wrapper defines
``training_step`` / ``validation_step`` / metrics for one task family.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~lightning_wrappers.base_lightning_wrapper.LightningWrapperBase
   ~lightning_wrappers.classification_wrapper.ClassificationWrapper
   ~lightning_wrappers.classification_wrapper.SoftTargetCrossEntropy
   ~lightning_wrappers.regression_wrapper.RegressionWrapper
   ~lightning_wrappers.well_lightning_wrapper.WELLRegressionWrapper
   ~lightning_wrappers.autoregressive_wrapper.AutoregressiveWrapper

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~lightning_wrappers.base_lightning_wrapper.construct_optimizer
   ~lightning_wrappers.base_lightning_wrapper.construct_scheduler

Callbacks
---------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~callbacks.film_monitor.FiLMMonitorCallback
   ~callbacks.image_grid_val_visualization.ValidationImageGridCallback
   ~callbacks.image_grid_val_visualization.ValidationVolumeGridCallback
   ~callbacks.iteration_speed.IterationSpeedCallback
   ~callbacks.mask_monitor.MaskMonitorCallback
   ~callbacks.model_ema.LabeledEMAWeightAveraging
   ~callbacks.omega_scale_monitor.OmegaScaleMonitorCallback
   ~callbacks.sequence_visualization_1d.Sequence1DVisualizationCallback
   ~callbacks.walltime_checkpointer.WalltimeCheckpointer
   ~callbacks.wandb_cache_cleanup.WandbCacheCleanupCallback

Data modules
------------

PyTorch Lightning ``LightningDataModule`` subclasses for each dataset
that experiments target.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~datamodules.mnist
   ~datamodules.emnist
   ~datamodules.tinyimagenet
   ~datamodules.spatial_recall_dataset
   ~datamodules.dali_imagenet_fused
   ~datamodules.pde.well

Utilities
---------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~utils.cli.get_deterministic_run_name
   ~utils.cli.load_config_from_file
   ~utils.checkpointing.download_checkpoint
   ~utils.checkpointing.load_checkpoint_state_dict
