"""E0: summarize random-bg eval matrix -> stability table.

Reads eval_rb.json (one entry per model, from eval_matrix2.py) and produces:
  - per-model accuracy table (PSNR, scale, inversion stats)
  - per-object stability: how consistent two independent reconstructions
    (seed 100 vs 200, different init pcd) are on every metric.
Prints JSON of the merged summary; the caller turns it into the deliverable.
"""
import argparse
import json
import os
import sys

OBJS = ["torus", "vase", "rocky", "mug", "sphere", "bumpy"]
SEEDS = [100, 200]


def rel(a, b):
    """relative difference between two values, guarding b==0."""
    if abs(b) < 1e-9:
        return 0.0 if abs(a) < 1e-9 else 1.0
    return abs(a - b) / abs(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fin", default="eval_rb.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.fin) as f:
        rows = json.load(f)
    by_model = {r["model"]: r for r in rows if not r.get("missing")}

    out = {"models": rows, "stability": {}, "summary": {}}
    for obj in OBJS:
        ms = {s: by_model.get(f"{obj}_rb_{s}") for s in SEEDS}
        miss = [s for s, m in ms.items() if m is None]
        if miss:
            out["stability"][obj] = {"missing_seeds": miss}
            continue
        a, b = ms[100], ms[200]
        stab = {
            "psnr": [round(a["psnr"], 3), round(b["psnr"], 3)],
            "psnr_diff": round(abs(a["psnr"] - b["psnr"]), 3),
            "scale_max": [round(a["scale_max"], 4), round(b["scale_max"], 4)],
            "scale_p99": [round(a["scale_p99"], 4), round(b["scale_p99"], 4)],
            "inv_p50": [round(a["inv_p50"], 4), round(b["inv_p50"], 4)],
            "inv_p50_rel": round(rel(a["inv_p50"], b["inv_p50"]), 3),
            "inv_p90": [round(a["inv_p90"], 4), round(b["inv_p90"], 4)],
            "inv_frac02": [round(a["inv_frac02"], 4), round(b["inv_frac02"], 4)],
            "inv_frac02_diff": round(abs(a["inv_frac02"] - b["inv_frac02"]), 4),
            "gaussians": [a["gaussians"], b["gaussians"]],
        }
        out["stability"][obj] = stab
        # averaged row for the accuracy table
        out["summary"][obj] = {
            "psnr": round(0.5 * (a["psnr"] + b["psnr"]), 2),
            "scale_max": round(max(a["scale_max"], b["scale_max"]), 3),
            "scale_p99": round(max(a["scale_p99"], b["scale_p99"]), 3),
            "inv_p50": round(0.5 * (a["inv_p50"] + b["inv_p50"]), 4),
            "inv_p90": round(0.5 * (a["inv_p90"] + b["inv_p90"]), 4),
            "inv_frac02": round(0.5 * (a["inv_frac02"] + b["inv_frac02"]), 4),
        }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
