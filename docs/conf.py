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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for the nvsubquadratic API reference."""

import os
import re as _re
import sys


sys.path.insert(0, os.path.abspath(".."))

github_version = os.environ.get("GITHUB_REF_NAME") or os.environ.get("CI_COMMIT_REF_NAME") or "main"

# -- Project information -----------------------------------------------------

project = "nvsubquadratic"


def _read_version():
    version_path = os.path.join(os.path.dirname(__file__), "..", "VERSION")
    return open(version_path).read().strip()


version = _read_version()
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
]

# Show `.. todo::` blocks (used in a few docstrings) instead of silently
# dropping them at build time.
todo_include_todos = True

templates_path = ["_templates"]

autodoc_typehints = "description"
autodoc_preserve_defaults = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
}

autodoc_mock_imports = [
    # CUDA / GPU-only deps — never installable on the docs runner.
    "subquadratic_ops_torch",
    "subquadratic_ops_torch._ext",
    "quack",
    "quack_kernels",
    "apex",
    "flash_attn",
    "dali",
    "nvidia",
    "nvidia.dali",
    # Pure-Python deps that the doc runner skips to keep the install lean.
    "einops",
    "megatron",
    "megatron.core",
    "omegaconf",
    "cleanfid",
    "diffusers",
    "pytorch_lightning",
    "lightning",
    "matplotlib",
    "PIL",
    "datasets",
    "h5py",
    "scipy",
    "the_well",
    "torch_fidelity",
    "torchmetrics",
    "torchvision",
    "timm",
    "wandb",
    "rich",
    "tqdm",
]

add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pytorch": ("https://docs.pytorch.org/docs/stable", None),
    "subquadratic_ops_torch": (
        "https://nvidia-bionemo.github.io/subquadraticOps-docs/",
        None,
    ),
}

_gh_repo = "https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private"
_gh_blob_base = f"{_gh_repo}/blob/{github_version}"

extlinks = {
    "subq-ops": (
        "https://nvidia-bionemo.github.io/subquadraticOps-docs/%s",
        "subquadratic-ops: %s",
    ),
    "ghsrc": (
        f"{_gh_blob_base}/%s",
        "%s",
    ),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "deflist",
    "colon_fence",
]

myst_heading_anchors = 3

doctest_global_setup = """
from typing import Any
import numpy as np
"""

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_templates/**"]

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_title = f"nvsubquadratic {version}"
html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_context = {
    "github_user": "NVIDIA-BioNeMo",
    "github_repo": "nvSubquadratic-private",
    "github_version": github_version,
    "doc_path": "docs",
}
html_theme_options = {
    "secondary_sidebar_items": ["page-toc"],
    "copyright_override": {"start": 2025},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 2,
}


# Match markdown links whose target escapes the docs/ tree via one or more
# `../`. The path must start with an alphanumeric/underscore so we don't
# rewrite intra-docs links like `[foo](./bar.md)` or `[foo](../README.md)`
# that resolve fine inside the rendered site. Anchored on the closing `](`
# of the markdown link to avoid touching unrelated parentheses.
_REL_REPO_LINK = _re.compile(r"\]\((?:\.\./)+([A-Za-z0-9_][^)]*)\)")


def _rewrite_repo_links(app, docname, source):
    """Rewrite markdown links like [text](../../foo/bar.py) to absolute GitHub URLs.

    Lets the source markdown stay readable on GitHub web (where the ``../``
    paths resolve correctly inside the repo) while the rendered Sphinx HTML
    points at the right blob URL on the active branch.  Intra-docs relative
    links and external URLs are left untouched.
    """
    text = source[0]
    new = _REL_REPO_LINK.sub(rf"]({_gh_blob_base}/\1)", text)
    if new != text:
        source[0] = new


def setup(app):
    """Register Sphinx extensions and event handlers for this build."""
    app.connect("source-read", _rewrite_repo_links)
