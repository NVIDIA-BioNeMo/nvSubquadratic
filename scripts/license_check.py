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

"""License header checker and enforcer for nvSubquadratic library.

This script ensures all Python (and Rust) files have proper NVIDIA license headers.
Designed to be invoked by pre-commit with `pass_filenames: true` and
`types_or: [python, rust]`; takes one or more file paths and auto-applies the
standard NVIDIA Apache-2.0 header to any file lacking one. Pre-commit then
detects the modification and fails the hook so the contributor can re-stage.
"""

import argparse
import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

year = datetime.now().year

license_text = textwrap.dedent("""\
    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """)

default_copyright_text = (
    f"SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)
default_combined_license = "\n".join([default_copyright_text, license_text])

copyright_regex_pattern = (
    r"SPDX-FileCopyrightText: Copyright \(c\) \d{4} NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.\n?\r?"
    r"(?:SPDX-FileCopyrightText: Copyright \(c\) \d{4}.*\n?\r?)*"
)
license_regex_pattern = copyright_regex_pattern + re.escape(license_text)
license_regex = re.compile(license_regex_pattern)

# Basic regex sanity checks.
assert re.compile(copyright_regex_pattern).match(default_copyright_text), "Default copyright text not valid"
assert license_regex.match(default_combined_license), "Default license text or regex is not valid"


def process_file(filepath: Path, dry_run: bool) -> bool:
    """Process a file to ensure it has a valid license block, or add a new one if it doesn't.

    Returns True if the file was (or would be) modified, False if it already had a valid header.
    """
    comment_start = get_comment_delimiter(filepath)
    try:
        lines = filepath.read_text().splitlines()
    except UnicodeDecodeError:
        logger.warning(f"Skipping {filepath} - unable to decode as text")
        return False

    start_line = 0
    if lines and lines[0].startswith("#!"):
        # Make sure there's a blank line after the shebang
        if len(lines) > 1 and lines[1] != "":
            lines.insert(1, "")
        start_line = 2

    license_block = []
    for line in lines[start_line:]:
        if line.startswith(comment_start):
            license_block.append(line)
        else:
            break

    def uncomment(text: str) -> str:
        return re.sub(rf"^{comment_start}\ ?", "", text)

    if "# noqa: license-check" in license_block:
        logger.info(f"Skipping {filepath} because it contains `# noqa: license-check`.")
        return False

    license_block_text = "\n".join(uncomment(line) for line in license_block) + "\n"
    if len(license_block) != 0 and license_regex.match(license_block_text):
        logger.info(f"Skipping {filepath} because it contains a valid license block.")
        return False

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Adding license block to {filepath}.")
    license_lines = "\n".join([default_copyright_text, license_text])
    license_lines = textwrap.indent(license_lines, comment_start + " ", predicate=lambda _: True)
    license_lines = "\n".join(line.rstrip() for line in license_lines.splitlines()) + "\n"
    lines.insert(start_line, license_lines)

    if not dry_run:
        filepath.write_text("\n".join(lines) + "\n")

    return True


def get_comment_delimiter(filepath: Path) -> str:
    """Get the comment delimiter for a file based on its extension."""
    match filepath.suffix:
        case ".py":
            return "#"
        case ".rs":
            return "//"
        case _:
            raise ValueError(f"Unsupported file type: {filepath}")


def main():
    """Main entry point for the license check script."""
    parser = argparse.ArgumentParser(description="Ensure files have proper license headers")
    parser.add_argument("files", nargs="+", help="Files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")

    args = parser.parse_args()

    for filename in args.files:
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        if filepath.suffix not in [".py", ".rs"]:
            raise ValueError(f"Unsupported file type: {filepath}")

        process_file(filepath, args.dry_run)


if __name__ == "__main__":
    main()
