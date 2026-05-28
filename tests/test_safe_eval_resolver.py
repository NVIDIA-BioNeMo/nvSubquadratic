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

# TODO: Add license header here

"""Tests for the safe arithmetic eval resolver (Fix 4: no arbitrary code execution).

Verifies that ``_safe_arithmetic_eval`` accepts valid arithmetic and rejects
dangerous expressions (imports, function calls, attribute access, etc.).
"""

from __future__ import annotations

import pytest

from experiments.utils.cli import _safe_arithmetic_eval


class TestSafeArithmeticEval:
    """Restricted eval resolver: arithmetic only, no code injection."""

    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("6 * 7", 42),
            ("15 / 4", 3.75),
            ("15 // 4", 3),
            ("15 % 4", 3),
            ("2 ** 10", 1024),
            ("-5", -5),
            ("+5", 5),
            ("(2 + 3) * 4", 20),
            ("100 // (4 * 2)", 12),
            ("3.14 * 2", 6.28),
            ("  42  ", 42),  # whitespace
        ],
    )
    def test_valid_arithmetic(self, expr: str, expected: float) -> None:
        result = _safe_arithmetic_eval(expr)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "eval('1+1')",
            "exec('x=1')",
            "(lambda: 1)()",
            "[x for x in range(10)]",
            "{'a': 1}",
            "print(42)",
            "os.getcwd()",
            "1 if True else 0",
        ],
    )
    def test_rejects_dangerous_expressions(self, expr: str) -> None:
        with pytest.raises(ValueError, match=r"(Unsupported operation|Invalid arithmetic)"):
            _safe_arithmetic_eval(expr)

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises(ValueError, match="Unsupported operation"):
            _safe_arithmetic_eval("(1).__class__")

    def test_rejects_syntax_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid arithmetic expression"):
            _safe_arithmetic_eval("2 +")
