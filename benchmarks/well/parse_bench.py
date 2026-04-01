"""Parse bench_sequential.sh output and summarize results.

Usage:
    python scripts/parse_bench.py logs/bench-sweep_JOBID.out
"""

import re
import sys
from pathlib import Path


def parse_benchmarks(log_path: str) -> list[dict]:
    """Parse WELL benchmark log files and extract per-run metrics."""
    text = Path(log_path).read_bytes().decode("utf-8", "replace")

    # Split on BENCHMARK headers
    sections = re.split(r"={64}\nBENCHMARK: (.+?)\n={64}", text)
    # sections[0] is preamble, then alternating [label, content, label, content, ...]

    results = []
    for i in range(1, len(sections), 2):
        label = sections[i]
        content = sections[i + 1] if i + 1 < len(sections) else ""

        # Extract tqdm rates: step/total [elapsed, rate]
        # Handle both MM:SS and H:MM:SS formats
        tqdm_hms = re.findall(r"(\d+)/\d+ \[(\d+):(\d+):(\d+)<[^,]+,\s*(\d+\.\d+)it/s", content)
        tqdm_ms = re.findall(r"(\d+)/\d+ \[(\d+):(\d+)<[^,]+,\s*(\d+\.\d+)it/s", content)

        data = []
        for m in tqdm_hms:
            step = int(m[0])
            secs = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            rate = float(m[4])
            data.append((step, secs, rate))
        for m in tqdm_ms:
            step = int(m[0])
            secs = int(m[1]) * 60 + int(m[2])
            rate = float(m[3])
            data.append((step, secs, rate))

        # Deduplicate
        seen = set()
        clean = []
        for step, secs, rate in sorted(data):
            if (step, secs) not in seen:
                seen.add((step, secs))
                clean.append((step, secs, rate))

        # Compute wall-clock rate over last half of training (skip warmup)
        wc_rate = None
        if len(clean) >= 4:
            mid = len(clean) // 2
            start = clean[mid]
            end = clean[-1]
            delta_steps = end[0] - start[0]
            delta_s = end[1] - start[1]
            if delta_s > 0:
                wc_rate = delta_steps / delta_s

        last_rate = clean[-1][2] if clean else None

        # Extract wandb run URL
        wandb_url = None
        m = re.search(r"https://wandb\.ai/\S+/runs/(\w+)", content)
        if m:
            wandb_url = m.group(1)

        results.append(
            {
                "label": label,
                "steps": clean[-1][0] if clean else 0,
                "elapsed_s": clean[-1][1] if clean else 0,
                "wc_rate": wc_rate,
                "tqdm_rate": last_rate,
                "wandb_id": wandb_url,
            }
        )

    return results


def main():
    """Entry point: parse a log file and print a summary table."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/parse_bench.py <log_file>")
        sys.exit(1)

    results = parse_benchmarks(sys.argv[1])

    print(f"\n{'Label':<35} {'Steps':>6} {'Elapsed':>8} {'WC it/s':>8} {'tqdm':>8} {'WandB':>12}")
    print("-" * 90)
    for r in results:
        elapsed_str = f"{r['elapsed_s']}s" if r["elapsed_s"] else "?"
        wc_str = f"{r['wc_rate']:.2f}" if r["wc_rate"] else "?"
        tqdm_str = f"{r['tqdm_rate']:.2f}" if r["tqdm_rate"] else "?"
        wandb_str = r["wandb_id"] or "?"
        print(f"{r['label']:<35} {r['steps']:>6} {elapsed_str:>8} {wc_str:>8} {tqdm_str:>8} {wandb_str:>12}")


if __name__ == "__main__":
    main()
