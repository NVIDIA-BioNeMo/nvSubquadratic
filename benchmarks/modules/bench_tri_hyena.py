# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TriHyena benchmark: window (box / gaussian / none), long-conv backend and execution mode.

Arch names are ``<window>-<torch|subq_ops>-<eager|compiled>`` (long and short conv backend); ``compiled`` applies
``torch.compile(mode="reduce-overhead")`` from the outside.  ``--end-node``
benchmarks the ending-node update instead of the starting-node one.

Usage (GPU):
    PYTHONPATH=. python benchmarks/modules/bench_tri_hyena.py
"""

from __future__ import annotations

import argparse

import torch
from _common import BenchConfig, PairMixer, add_common_args, build_tri_hyena, cast_params, run_sweep


ARCHS = [
    "box-subq_ops-compiled",
    "box-subq_ops-eager",
    "box-torch-eager",
    "gaussian-subq_ops-compiled",
    "none-subq_ops-compiled",
]


def make_builder(name: str, start_node: bool):
    window, backend, execution = name.split("-")

    def build(cfg: BenchConfig, seq_len: int) -> torch.nn.Module:
        torch.manual_seed(0)
        mixer = build_tri_hyena(
            cfg.hidden_dim,
            seq_len,
            window=window,
            max_offset=cfg.max_offset,
            fft_backend="torch_fft" if backend == "torch" else "subq_ops",
            short_conv_backend=backend,
            start_node=start_node,
        )
        norm = torch.nn.LayerNorm(cfg.hidden_dim)
        module = cast_params(PairMixer(mixer, norm).to(cfg.device), cfg.dtype)
        return torch.compile(module, mode="reduce-overhead") if execution == "compiled" else module

    return build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser, archs=ARCHS, csv_default="bench_tri_hyena.csv")
    parser.add_argument("--end-node", action="store_true", help="Benchmark the ending-node update.")
    args = parser.parse_args()
    cfg = BenchConfig.from_args(args)
    run_sweep(cfg, args, archs={name: make_builder(name, not args.end_node) for name in ARCHS})


if __name__ == "__main__":
    main()
