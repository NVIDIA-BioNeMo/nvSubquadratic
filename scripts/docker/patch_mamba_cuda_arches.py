#!/usr/bin/env python3

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

"""Patch mamba-ssm / causal-conv1d setup.py to honor TORCH_CUDA_ARCH_LIST.

Upstream hardcodes -gencode for sm_75..sm_120 (and ignores TORCH_CUDA_ARCH_LIST),
which OOMs / gcc-ICEs when building under QEMU arm64. This rewrites the gencode
block from TORCH_CUDA_ARCH_LIST and makes append_nvcc_threads honor NVCC_THREADS.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


# TORCH_CUDA_ARCH_LIST token -> sm number used in -gencode
ARCH_TO_SM = {
    "7.5": 75,
    "8.0": 80,
    "8.6": 86,
    "8.7": 87,
    "8.9": 89,
    "9.0": 90,
    "10.0": 100,
    "12.0": 120,
}


def parse_arch_list(raw: str) -> list[tuple[str, int]]:
    arches: list[tuple[str, int]] = []
    for part in raw.replace(",", ";").split(";"):
        token = part.strip()
        if not token:
            continue
        # Drop optional "a" suffix (e.g. 9.0a / 10.0a).
        base = token.rstrip("aA")
        if base not in ARCH_TO_SM:
            raise SystemExit(
                f"Unsupported arch {token!r} in TORCH_CUDA_ARCH_LIST={raw!r}. Known: {', '.join(ARCH_TO_SM)}"
            )
        arches.append((base, ARCH_TO_SM[base]))
    if not arches:
        raise SystemExit("TORCH_CUDA_ARCH_LIST is empty; nothing to compile")
    return arches


def gencode_block(arches: list[tuple[str, int]], indent: str) -> str:
    lines: list[str] = [f"{indent}# Patched by patch_mamba_cuda_arches.py from TORCH_CUDA_ARCH_LIST"]
    for base, sm in arches:
        lines.append(f'{indent}cc_flag.append("-gencode")')
        lines.append(f'{indent}cc_flag.append("arch=compute_{sm},code=sm_{sm}")  # {base}')
    return "\n".join(lines) + "\n"


# Matches the hardcoded compute_75 start through the version-gated arch block,
# stopping before the CXX11 ABI HACK comment present in both setup.py files.
GENCODE_BLOCK_RE = re.compile(
    r"(?P<indent>[ \t]*)cc_flag\.append\(\"-gencode\"\)\n"
    r"[ \t]*cc_flag\.append\(\"arch=compute_75,code=sm_75\"\)\n"
    r"(?:.*\n)*?"
    r"(?=[ \t]*# HACK: The compiler flag)",
    re.MULTILINE,
)

NVCC_THREADS_RE = re.compile(
    r"def append_nvcc_threads\(nvcc_extra_args\):\n"
    r"[ \t]*return nvcc_extra_args \+ \[\"--threads\", \"4\"\]\n"
)


def patch_file(path: Path, arches: list[tuple[str, int]]) -> None:
    text = path.read_text()
    match = GENCODE_BLOCK_RE.search(text)
    if match is None:
        raise SystemExit(f"{path}: could not find hardcoded gencode block to patch")

    indent = match.group("indent")
    text = GENCODE_BLOCK_RE.sub(gencode_block(arches, indent), text, count=1)

    new_threads, n = NVCC_THREADS_RE.subn(
        "def append_nvcc_threads(nvcc_extra_args):\n"
        '    return nvcc_extra_args + ["--threads", os.getenv("NVCC_THREADS", "4")]\n',
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"{path}: could not patch append_nvcc_threads")
    text = new_threads

    path.write_text(text)
    sm_list = ", ".join(f"sm_{sm}" for _, sm in arches)
    print(f"Patched {path}: gencodes -> {sm_list}; NVCC_THREADS via env")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(f"usage: {argv[0]} setup.py [setup.py ...]")

    raw = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if not raw:
        raise SystemExit("TORCH_CUDA_ARCH_LIST must be set when patching")

    arches = parse_arch_list(raw)
    for arg in argv[1:]:
        patch_file(Path(arg), arches)


if __name__ == "__main__":
    main(sys.argv)
