"""One-off helper to summarize the per-axis / omega_0 ablation runs."""

import wandb


api = wandb.Api()
runs = {
    "42070 non-patch, both  (peraxis+omega30)": "implicit-long-convs/nvsubquadratic/gcgsp5sg",
    "42071 PATCH,      both (peraxis+omega30)": "implicit-long-convs/nvsubquadratic/tg8kfq1i",
    "42072 non-patch, baseline (scalar, om10)": "implicit-long-convs/nvsubquadratic/38z9ubpo",
    "42073 non-patch, peraxis only  (om10)   ": "implicit-long-convs/nvsubquadratic/iaaj7ij9",
    "42074 non-patch, omega30 only  (scalar) ": "implicit-long-convs/nvsubquadratic/iun0rqsu",
}

print("=== Summary (latest point) ===")
print(f"{'run':<44} {'state':>10} {'step':>7} {'train/loss_ep':>13} {'val/loss':>10}")
print("-" * 90)
for tag, path in runs.items():
    try:
        r = api.run(path)
        s = r.summary

        def fmt(k):
            v = s.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

        step = s.get("_step") or "?"
        print(f"{tag:<44} {r.state:>10} {step!s:>7} {fmt('train/loss_epoch'):>13} {fmt('val/loss'):>10}")
    except Exception as e:
        print(f"{tag:<44} ERR: {e}")

print()
print("=== Available logged keys per run ===")
for tag, path in runs.items():
    r = api.run(path)
    keys = sorted(k for k in r.summary.keys() if not k.startswith("_") and not k.startswith("gradient"))
    print(f"{tag}: {keys[:30]}")

print()
print("=== train/loss_step trajectory at matched steps (smoothed ±500) ===")
TARGETS = [2000, 5000, 10000, 15000, 20000, 23000, 25000, 30000, 35000, 40000, 45000, 49000]

for tag, path in runs.items():
    r = api.run(path)
    try:
        # Use scan_history to avoid sampling issues
        import pandas as pd

        rows = []
        for row in r.scan_history(keys=["trainer/global_step", "train/loss_step"]):
            rows.append(row)
        if not rows:
            # Try alternative key
            rows = []
            for row in r.scan_history(keys=["_step", "train/loss_step"]):
                rows.append(row)
        if not rows:
            print(f"{tag}: no history data")
            continue
        df = pd.DataFrame(rows).dropna(subset=["train/loss_step"])
        step_col = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"
        df = df.dropna(subset=[step_col])
        last_step = int(df[step_col].max())
        parts = [f"{tag:<44} last_step={last_step:>6}"]
        for t in TARGETS:
            if t > last_step:
                parts.append(f"{t}:   -  ")
                continue
            mask = (df[step_col] >= t - 500) & (df[step_col] <= t + 500)
            sub = df.loc[mask, "train/loss_step"]
            parts.append(f"{t}:{sub.mean():.4f}" if not sub.empty else f"{t}:  ?  ")
        print(" | ".join(parts))
    except Exception as e:
        print(f"{tag}: history ERR: {e}")
