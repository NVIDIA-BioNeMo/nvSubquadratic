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

"""Assert every hard-coded torch / CUDA pin agrees with ``pyproject.toml``.

``pyproject.toml`` declares the supported range (``torch>=2.12.0,<2.13.0``), but the
Docker image, the conda bootstrap, the SLURM builders and the install docs each
repeat a *concrete* version and wheel index.  Nothing links them: the SLURM builder
replays the Dockerfile rather than parsing it, and the docs are prose.  They drift
silently when only one is edited.

That drift is not cosmetic.  If a script installs a torch outside the pyproject
range, the later ``pip install -e .`` re-resolves torch from PyPI and replaces the
CUDA-matched wheel *after* apex / mamba / causal-conv1d were already compiled
against it, leaving extensions built for a torch that is no longer installed.

This checker parses the range from ``pyproject.toml`` and enforces, across every
site listed in ``EXACT_PIN_SITES`` / ``SPECIFIER_MIRROR_SITES`` / ``CUDA_INDEX_FILES``:

1. every exact pin satisfies the pyproject specifier;
2. all exact pins for a package agree with each other;
3. all PyTorch wheel-index URLs use the same ``cuXYZ`` tag;
4. that CUDA tag agrees with the ``[cuda]`` extra (``...-cu13``) and the ``[dali]``
   extra (``nvidia-dali-cuda130``);
5. any file that mirrors the specifier itself quotes it verbatim.

Run manually with ``scripts/check_version_pins.py``; wired into pre-commit (and so
into CI's ``pre-commit run --all-files``) as ``version-pin-check``.
"""

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Where the concrete pins live ─────────────────────────────────────────────
# (path, package, regex).  Each regex must expose the version in group 1.
EXACT_PIN_SITES = [
    ("Dockerfile", "torch", r"^ARG\s+TORCH_VERSION=([0-9][0-9.]*)"),
    ("Dockerfile", "torchvision", r"^ARG\s+TORCHVISION_VERSION=([0-9][0-9.]*)"),
    ("setup_conda_env.sh", "torch", r"^TORCH_VERSION=([0-9][0-9.]*)"),
    ("setup_conda_env.sh", "torchvision", r"^TORCHVISION_VERSION=([0-9][0-9.]*)"),
    ("scripts/slurm/enroot/build_sqsh_slurm.sh", "torch", r"^TORCH_VERSION=\"\$\{TORCH_VERSION:-([0-9][0-9.]*)\}\""),
    (
        "scripts/slurm/enroot/build_sqsh_slurm.sh",
        "torchvision",
        r"^TORCHVISION_VERSION=\"\$\{TORCHVISION_VERSION:-([0-9][0-9.]*)\}\"",
    ),
    ("scripts/slurm/setup_env.sh", "torch", r"pip install\s+torch==([0-9][0-9.]*)"),
    ("scripts/slurm/setup_env.sh", "torchvision", r"torchvision==([0-9][0-9.]*)"),
    ("README.md", "torch", r"pip install torch==([0-9][0-9.]*)"),
    ("README.md", "torchvision", r"pip install torch==[0-9][0-9.]*\s+torchvision==([0-9][0-9.]*)"),
    ("docs/getting_started.md", "torch", r"pip install torch==([0-9][0-9.]*)"),
    ("docs/getting_started.md", "torchvision", r"pip install torch==[0-9][0-9.]*\s+torchvision==([0-9][0-9.]*)"),
]

# Files that repeat the *range* rather than a concrete version. Must match verbatim.
SPECIFIER_MIRROR_SITES = [
    (".github/workflows/install-matrix.yml", "torch", r'"torch(>=[0-9][0-9.]*,<[0-9][0-9.]*)"'),
    (".github/workflows/install-matrix.yml", "torchvision", r'"torchvision(>=[0-9][0-9.]*,<[0-9][0-9.]*)"'),
]

# Every PyTorch wheel index must point at the same CUDA build.
CUDA_INDEX_FILES = [
    "Dockerfile",
    "setup_conda_env.sh",
    "scripts/slurm/enroot/build_sqsh_slurm.sh",
    "scripts/slurm/setup_env.sh",
    "README.md",
    "docs/getting_started.md",
]
CUDA_INDEX_RE = re.compile(r"download\.pytorch\.org/whl/(cu\d+)")


def parse_version(text):
    """Turn ``"2.12.1"`` into ``(2, 12, 1)`` for ordered comparison."""
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def pad(version, length):
    """Right-pad a version tuple with zeros so comparisons are well defined."""
    return version + (0,) * (length - len(version))


def satisfies(version, specifier):
    """Return True if ``version`` satisfies a ``>=X,<Y`` style ``specifier``."""
    for clause in specifier.split(","):
        clause = clause.strip()
        match = re.match(r"(>=|<=|==|<|>)\s*([0-9][0-9.]*)", clause)
        if not match:
            raise ValueError(f"unparseable version clause: {clause!r}")
        op, bound_text = match.groups()
        bound = parse_version(bound_text)
        width = max(len(version), len(bound))
        left, right = pad(version, width), pad(bound, width)
        if op == ">=" and not left >= right:
            return False
        if op == "<=" and not left <= right:
            return False
        if op == "==" and left != right:
            return False
        if op == "<" and not left < right:
            return False
        if op == ">" and not left > right:
            return False
    return True


def read(rel_path):
    """Read a repo-relative file, or return None when it is absent."""
    path = REPO_ROOT / rel_path
    return path.read_text() if path.exists() else None


def find(rel_path, pattern, errors):
    """Return (value, line_number) for the first match, recording a miss as an error."""
    text = read(rel_path)
    if text is None:
        errors.append(f"{rel_path}: file not found (update the site tables in {Path(__file__).name})")
        return None, None
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        errors.append(f"{rel_path}: no match for {pattern!r} — the pin moved or was reworded; update this checker")
        return None, None
    line_no = text[: match.start()].count("\n") + 1
    return match.group(1), line_no


def load_pyproject(errors):
    """Extract the torch/torchvision specifiers and the CUDA/DALI extras."""
    text = read("pyproject.toml")
    if text is None:
        errors.append("pyproject.toml: not found")
        return {}, None, None

    specifiers = {}
    for package in ("torch", "torchvision"):
        match = re.search(rf'"{package}(>=[0-9][0-9.]*,<[0-9][0-9.]*)"', text)
        if match:
            specifiers[package] = match.group(1)
        else:
            errors.append(f"pyproject.toml: could not parse a '{package}>=..,<..' specifier")

    cuda_extra = re.search(r'cuda\s*=\s*\["subquadratic-ops-torch-cu(\d+)', text)
    dali_extra = re.search(r'dali\s*=\s*\["nvidia-dali-cuda(\d+)', text)
    return (
        specifiers,
        cuda_extra.group(1) if cuda_extra else None,
        dali_extra.group(1) if dali_extra else None,
    )


def main():
    """Check every pin site against pyproject.toml; exit non-zero on drift."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("files", nargs="*", help="Ignored; the checked set is fixed (pre-commit passes filenames).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print on failure.")
    args = parser.parse_args()

    errors = []
    specifiers, cuda_extra_major, dali_cuda_tag = load_pyproject(errors)

    # 1 + 2 — every concrete pin is in range, and they all agree.
    seen = {}
    rows = []
    for rel_path, package, pattern in EXACT_PIN_SITES:
        value, line_no = find(rel_path, pattern, errors)
        if value is None:
            continue
        rows.append((f"{rel_path}:{line_no}", package, value))
        seen.setdefault(package, []).append((rel_path, line_no, value))

        specifier = specifiers.get(package)
        if specifier and not satisfies(parse_version(value), specifier):
            errors.append(
                f"{rel_path}:{line_no}: {package}=={value} violates pyproject's "
                f'"{package}{specifier}" — a later `pip install -e .` would re-resolve '
                f"torch and replace the CUDA-matched wheel after extensions were built"
            )

    for package, entries in seen.items():
        distinct = {value for _, _, value in entries}
        if len(distinct) > 1:
            detail = ", ".join(f"{p}:{ln}={v}" for p, ln, v in entries)
            errors.append(f"{package}: pins disagree across files ({detail})")

    # 3 — one CUDA wheel index everywhere.
    tags = {}
    for rel_path in CUDA_INDEX_FILES:
        text = read(rel_path)
        if text is None:
            errors.append(f"{rel_path}: file not found (update CUDA_INDEX_FILES)")
            continue
        for match in CUDA_INDEX_RE.finditer(text):
            tags.setdefault(match.group(1), []).append(f"{rel_path}:{text[: match.start()].count(chr(10)) + 1}")
    if len(tags) > 1:
        detail = "; ".join(f"{tag} at {', '.join(sites)}" for tag, sites in sorted(tags.items()))
        errors.append(f"PyTorch wheel index disagrees on the CUDA build: {detail}")

    # 4 — the wheel index, the [cuda] extra and the [dali] extra name the same CUDA major.
    index_tag = next(iter(tags)) if len(tags) == 1 else None
    if index_tag:
        index_major = index_tag[2:4]  # cu130 -> "13"
        if cuda_extra_major and cuda_extra_major != index_major:
            errors.append(
                f"pyproject's [cuda] extra targets CUDA {cuda_extra_major} "
                f"(subquadratic-ops-torch-cu{cuda_extra_major}) but the wheel index is {index_tag}"
            )
        if dali_cuda_tag and dali_cuda_tag[:2] != index_major:
            errors.append(
                f"pyproject's [dali] extra targets nvidia-dali-cuda{dali_cuda_tag} but the wheel index is {index_tag}"
            )

    # 5 — mirrored specifiers are verbatim copies.
    for rel_path, package, pattern in SPECIFIER_MIRROR_SITES:
        value, line_no = find(rel_path, pattern, errors)
        if value is None:
            continue
        rows.append((f"{rel_path}:{line_no}", package, value))
        expected = specifiers.get(package)
        if expected and value != expected:
            errors.append(
                f'{rel_path}:{line_no}: mirrors "{package}{value}" but pyproject declares "{package}{expected}"'
            )

    if errors:
        print("Version pins are inconsistent:\n", file=sys.stderr)
        for error in errors:
            print(f"  ERROR  {error}", file=sys.stderr)
        print(
            "\nThe authoritative range lives in pyproject.toml. Update every site above to agree,"
            "\nor adjust scripts/check_version_pins.py if a pin legitimately moved.",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        spec_text = ", ".join(f"{pkg}{spec}" for pkg, spec in sorted(specifiers.items()))
        print(f"pyproject: {spec_text}" + (f" | index {index_tag}" if index_tag else ""))
        width = max(len(site) for site, _, _ in rows)
        for site, package, value in rows:
            print(f"  ok  {site:<{width}}  {package} {value}")
        print(f"\n{len(rows)} pin(s) consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
