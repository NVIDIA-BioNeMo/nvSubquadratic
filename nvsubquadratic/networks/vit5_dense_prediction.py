# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible alias for the general-purpose ViT5 dense network."""

from nvsubquadratic.networks.vit5_general_purpose import ViT5GeneralPurposeNet


ViT5DensePredictionNet = ViT5GeneralPurposeNet
