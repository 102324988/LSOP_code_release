"""E0: profile stability across two independent GOF reconstructions.

Per-ray metrics between run1 and run2 profiles of the same object:
  - surface-depth (argmax of stopping density) absolute difference
  - per-ray IoU of the occupied interval (opacity > threshold)
  - coverage (total stopping prob) difference
"""
import argparse
import json
import os

import numpy as np


def interval_iou(a, b):
    """IoU of binary occupancy intervals along the ray."""
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True)
    ap.add_argument("--run2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--occ_thresh", type=float, default=0.5)
    ap.add_argument("--depth_thresh", type=float, default=0.05)
    args = ap.parse_args()

    def load(prefix):
        return {
            "peak": np.load(os.path.join(prefix, "depth_peak.npy")),
            "cov": np.load(os.path.join(prefix, "coverage.npy")),
            "occ": np.load(os.path.join(prefix, "opacity_profiles.npy")),
        }

    r1, r2 = load(args.run1), load(args.run2)
    assert r1["peak"].shape == r2["peak"].shape

    d = np.abs(r1["peak"] - r2["peak"])
    ious = np.array([interval_iou(a > args.occ_thresh, b > args.occ_thresh)
                     for a, b in zip(r1["occ"].reshape(-1, r1["occ"].shape[-1]),
                                     r2["occ"].reshape(-1, r2["occ"].shape[-1]))])
    dcov = np.abs(r1["cov"] - r2["cov"])

    res = {
        "n_rays": int(d.size),
        "depth_abs_diff": {"median": float(np.median(d)), "mean": float(d.mean()),
                           "p90": float(np.percentile(d, 90)),
                           "frac_gt_0.05": float((d > args.depth_thresh).mean())},
        "occupied_interval_iou": {"mean": float(ious.mean()), "median": float(np.median(ious)),
                                  "frac_lt_0.9": float((ious < 0.9).mean())},
        "coverage_abs_diff": {"mean": float(dcov.mean()), "median": float(np.median(dcov))},
        "mean_coverage_run1": float(r1["cov"].mean()),
        "mean_coverage_run2": float(r2["cov"].mean()),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "stability.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[stability] depth|Δ|: med={res['depth_abs_diff']['median']:.4f} "
          f"mean={res['depth_abs_diff']['mean']:.4f} p90={res['depth_abs_diff']['p90']:.4f} "
          f"({res['depth_abs_diff']['frac_gt_0.05']*100:.1f}% rays > 0.05)")
    print(f"[stability] occ-interval IoU: mean={res['occupied_interval_iou']['mean']:.4f} "
          f"med={res['occupied_interval_iou']['median']:.4f}")


if __name__ == "__main__":
    main()
