# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for creating sequence mixers.

These factories are needed to ensure each residual block gets its own
mixer instance when using migrated dataclass configs.
"""

from nvsubq import Hyena, HyenaConfig, QKVSequenceMixer, QKVSequenceMixerConfig


def create_hyena_sequence_mixer(hyena_config: HyenaConfig, qkv_config: QKVSequenceMixerConfig) -> QKVSequenceMixer:
    """Factory: create a new QKVSequenceMixer with Hyena.

    Args:
        hyena_config: Configuration for Hyena mixer
        qkv_config: Configuration for QKV sequence mixer

    Returns:
        A new QKVSequenceMixer instance with a new Hyena instance
    """
    return QKVSequenceMixer(qkv_config, mixer=Hyena(hyena_config))
