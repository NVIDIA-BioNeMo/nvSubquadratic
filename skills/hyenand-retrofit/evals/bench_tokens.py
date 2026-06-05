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

"""Benchmark LLM token usage for the hyenand-retrofit skill.

For each eval case in evals.json, this spawns a fresh, non-interactive Claude
agent (`claude -p`) with the skill text injected inline, lets it actually carry
out the retrofit (read inputs, write the sibling output file), and records the
token / cost / latency figures Claude reports in its JSON result envelope.

Because every `claude -p` invocation is a cold process, it pays the full
system-prompt cache-creation cost once (no warm cache between runs). That fixed
overhead is part of what "invoking the skill" costs, so we report it explicitly
(cache_creation_tokens) alongside the task-driven input/output tokens.

Usage:
    python bench_tokens.py [--runs N] [--eval-id ID] [--model MODEL] [--dry-run]

    --runs N       Repetitions per eval case (default: 3)
    --eval-id ID   Run only the eval with this integer id (default: all)
    --model MODEL  Model to pin for reproducibility (default: claude-opus-4-8)
    --dry-run      Print the prompts and the exact CLI invocation; call nothing

Each run appends one JSON line to bench_results.jsonl with these fields:
    eval_id, eval_name, run, model, input_tokens, output_tokens,
    cache_creation_tokens, cache_read_tokens, billed_input_tokens,
    billed_total_tokens, cost_usd, duration_ms, num_turns,
    permission_denials, ts
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EVALS_FILE = pathlib.Path(__file__).parent / "evals.json"
RESULTS_FILE = pathlib.Path(__file__).parent / "bench_results.jsonl"
SKILL_FILE = pathlib.Path(__file__).resolve().parents[1] / "SKILL.md"

DEFAULT_MODEL = "claude-opus-4-8"

# Directories an eval is allowed to write into. Cleanup is scoped to these so we
# never touch the bench script, its results, or unrelated working-tree changes.
WRITE_DIRS = [
    "examples/vit5_imagenet/",
    "skills/hyenand-retrofit/evals/inputs/",
]


def load_evals():
    """Load eval cases from evals.json."""
    return json.loads(EVALS_FILE.read_text())["evals"]


def build_prompt(eval_case: dict) -> str:
    """Inject the full skill text + file hints + the eval prompt.

    Injecting SKILL.md inline reproduces the "skill is loaded" condition and
    folds the skill's fixed context cost into the measured input tokens.
    """
    skill_text = SKILL_FILE.read_text() if SKILL_FILE.exists() else ""
    preamble = f"<skill name='hyenand-retrofit'>\n{skill_text}\n</skill>\n\n" if skill_text else ""
    file_hints = ""
    if eval_case.get("files"):
        file_hints = (
            "Relevant files to read first (paths relative to repo root):\n"
            + "\n".join(f"  - {f}" for f in eval_case["files"])
            + "\n\n"
        )
    return preamble + file_hints + eval_case["prompt"]


def porcelain() -> dict[str, str]:
    """Map of path -> status code from `git status --porcelain`, scoped to WRITE_DIRS."""
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    status = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip()
        # Rename entries look like "old -> new"; keep the new path.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if any(path.startswith(d) for d in WRITE_DIRS):
            status[path] = code
    return status


def restore_tree(baseline: dict[str, str]):
    """Undo whatever the agent did inside WRITE_DIRS, leaving everything else alone.

    - Files newly untracked (not in baseline) are deleted.
    - Tracked files the agent modified are restored with `git checkout --`.
    """
    after = porcelain()
    for path, code in after.items():
        if path in baseline and baseline[path] == code:
            continue  # unchanged from before this run
        full = REPO_ROOT / path
        if code.startswith("??"):
            if full.exists():
                full.unlink()
        else:
            # Modified / added tracked file -> restore committed version.
            subprocess.run(["git", "checkout", "--", path], cwd=REPO_ROOT, capture_output=True, text=True)


def parse_result(stdout: str) -> dict:
    """Parse the `--output-format json` envelope (a single JSON object)."""
    obj = None
    try:
        obj = json.loads(stdout)
    except json.JSONDecodeError:
        # Fallback: scan for the last line that parses and has type==result.
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cand.get("type") == "result":
                obj = cand
    if obj is None:
        return {}

    usage = obj.get("usage", {})
    model_usage = obj.get("modelUsage", {}) or {}
    # Primary model = the one that produced the most output tokens.
    primary_model = ""
    if model_usage:
        primary_model = max(model_usage, key=lambda m: model_usage[m].get("outputTokens", 0))

    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cc = usage.get("cache_creation_input_tokens", 0)
    cr = usage.get("cache_read_input_tokens", 0)
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_creation_tokens": cc,
        "cache_read_tokens": cr,
        "billed_input_tokens": inp + cc + cr,
        "billed_total_tokens": inp + cc + cr + out,
        "cost_usd": round(obj.get("total_cost_usd", 0.0), 6),
        "duration_ms": obj.get("duration_ms", 0),
        "num_turns": obj.get("num_turns", 0),
        "permission_denials": len(obj.get("permission_denials", []) or []),
        "primary_model": primary_model,
        "is_error": obj.get("is_error", False),
    }


def run_one(eval_case: dict, run_index: int, model: str, dry_run: bool) -> dict:
    """Run one eval case via `claude -p` and return a result record."""
    prompt = build_prompt(eval_case)
    cmd = ["claude", "-p", "--output-format", "json", "--permission-mode", "acceptEdits", "--model", model]

    if dry_run:
        print(f"\n--- DRY RUN eval {eval_case['id']} ({eval_case['name']}) run {run_index} ---")
        print("CMD:", " ".join(cmd), "  (prompt via stdin)")
        print("PROMPT (first 400 chars of task portion):")
        print("   ", eval_case["prompt"][:400], "...")
        return {}

    baseline = porcelain()
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, input=prompt, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        restore_tree(baseline)
        return {
            "eval_id": eval_case["id"],
            "eval_name": eval_case["name"],
            "run": run_index,
            "model": model,
            "error": "timeout",
            "ts": int(time.time()),
        }
    wall_sec = round(time.monotonic() - t0, 1)

    if proc.returncode != 0:
        restore_tree(baseline)
        return {
            "eval_id": eval_case["id"],
            "eval_name": eval_case["name"],
            "run": run_index,
            "model": model,
            "error": f"exit {proc.returncode}: {proc.stderr[:160]}",
            "ts": int(time.time()),
        }

    parsed = parse_result(proc.stdout)
    restore_tree(baseline)

    record = {
        "eval_id": eval_case["id"],
        "eval_name": eval_case["name"],
        "run": run_index,
        "model": model,
        "wall_sec": wall_sec,
        "ts": int(time.time()),
        **parsed,
    }
    return record


def fmt_int(x):
    """Format a number with thousands separators, or return str(x)."""
    return f"{x:,}" if isinstance(x, (int, float)) else str(x)


def print_table(records: list[dict]):
    """Print per-run rows and per-eval averages for successful benchmark records."""
    rows = [r for r in records if "error" not in r]
    errs = [r for r in records if "error" in r]
    if rows:
        hdr = (
            f"{'eval':<24}{'run':>4}  {'in':>6} {'cache_cr':>8} {'cache_rd':>8} "
            f"{'out':>6} {'billed':>8} {'cost$':>7} {'turns':>5} {'sec':>6}"
        )
        print("\n" + hdr)
        print("-" * len(hdr))
        for r in rows:
            print(
                f"{r['eval_name']:<24}{r['run']:>4}  "
                f"{r['input_tokens']:>6,} {r['cache_creation_tokens']:>8,} "
                f"{r['cache_read_tokens']:>8,} {r['output_tokens']:>6,} "
                f"{r['billed_total_tokens']:>8,} {r['cost_usd']:>7.3f} "
                f"{r['num_turns']:>5} {r.get('wall_sec', 0):>6.1f}"
            )

        # Per-eval averages
        by = {}
        for r in rows:
            by.setdefault(r["eval_name"], []).append(r)
        print(
            f"\n{'AVG per eval':<24}{'n':>4}  {'in':>6} {'cache_cr':>8} {'cache_rd':>8} "
            f"{'out':>6} {'billed':>8} {'cost$':>7} {'turns':>5} {'sec':>6}"
        )
        print("-" * len(hdr))
        for name, rs in by.items():
            n = len(rs)

            def avg(k):
                return sum(r[k] for r in rs) / n

            print(
                f"{name:<24}{n:>4}  {round(avg('input_tokens')):>6,} "
                f"{round(avg('cache_creation_tokens')):>8,} {round(avg('cache_read_tokens')):>8,} "
                f"{round(avg('output_tokens')):>6,} {round(avg('billed_total_tokens')):>8,} "
                f"{avg('cost_usd'):>7.3f} {round(avg('num_turns')):>5} {avg('wall_sec'):>6.1f}"
            )
        total_cost = sum(r["cost_usd"] for r in rows)
        print(f"\nTotal cost across {len(rows)} runs: ${total_cost:.3f}  (model: {rows[0]['model']})")
    for e in errs:
        print(f"[ERROR] {e['eval_name']} run {e['run']}: {e['error']}")


def main():
    """CLI entry point for the token benchmark."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=3, metavar="N")
    p.add_argument("--eval-id", type=int, default=None, metavar="ID")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    evals = load_evals()
    if args.eval_id is not None:
        evals = [e for e in evals if e["id"] == args.eval_id]
        if not evals:
            sys.exit(f"No eval with id={args.eval_id}")

    collected = []
    for ec in evals:
        print(f"\n=== eval {ec['id']}: {ec['name']}  ({args.runs} run(s), model={args.model}) ===")
        for i in range(1, args.runs + 1):
            print(f"  run {i}/{args.runs} ...", end=" ", flush=True)
            rec = run_one(ec, i, args.model, args.dry_run)
            if args.dry_run:
                continue
            collected.append(rec)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if "error" in rec:
                print(f"ERROR: {rec['error']}")
            else:
                print(
                    f"in={rec['input_tokens']:,} cache_cr={rec['cache_creation_tokens']:,} "
                    f"out={rec['output_tokens']:,} billed={rec['billed_total_tokens']:,} "
                    f"${rec['cost_usd']:.3f} turns={rec['num_turns']} {rec['wall_sec']:.0f}s"
                )

    if not args.dry_run:
        print_table(collected)
        print(
            f"\nAppended {len([r for r in collected if 'error' not in r])} rows "
            f"to {RESULTS_FILE.relative_to(REPO_ROOT)}"
        )


if __name__ == "__main__":
    main()
