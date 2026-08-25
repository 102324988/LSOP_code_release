"""E0: aggregate per-object metrics + profiles stats into a summary table.

Reads output/<obj>_1/, output/<obj>_2/, output/<obj>_stability/ and prints a
Markdown table (and JSON) with the key E0 verdict numbers.
"""
import argparse
import json
import os

import numpy as np

OBJS = ["torus", "vase", "rocky", "mug", "sphere", "bumpy"]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for obj in OBJS:
        row = {"object": obj}
        stab = load_json(os.path.join(args.root, f"{obj}_stability", "stability.json"))
        if stab:
            row.update({
                "depth_med": stab["depth_abs_diff"]["median"],
                "depth_p90": stab["depth_abs_diff"]["p90"],
                "depth_frac_gt": stab["depth_abs_diff"]["frac_gt_0.05"],
                "occ_iou": stab["occupied_interval_iou"]["mean"],
                "coverage": (stab["mean_coverage_run1"] + stab["mean_coverage_run2"]) / 2,
            })
        mets = []
        for run in (1, 2):
            m = load_json(os.path.join(args.root, f"{obj}_{run}", "profiles", "metrics.json"))
            if m:
                mets.append(m)
        if mets:
            row["inv_chamfer"] = float(np.mean([m["chamfer"] for m in mets]))
            row["inv_fscore"] = float(np.mean([m["fscore"] for m in mets]))
        rows.append(row)

    # Markdown table
    lines = ["| 物体 | coverage | 深度差 中位 | 深度差 p90 | 深度差>0.05 比例 | 占用IoU | 反解Chamfer | 反解F@0.02 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['object']} | {r.get('coverage', float('nan')):.3f} "
                     f"| {r.get('depth_med', float('nan')):.4f} "
                     f"| {r.get('depth_p90', float('nan')):.4f} "
                     f"| {r.get('depth_frac_gt', float('nan'))*100:.1f}% "
                     f"| {r.get('occ_iou', float('nan')):.3f} "
                     f"| {r.get('inv_chamfer', float('nan')):.4f} "
                     f"| {r.get('inv_fscore', float('nan')):.3f} |")
    print("\n".join(lines))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
            print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
