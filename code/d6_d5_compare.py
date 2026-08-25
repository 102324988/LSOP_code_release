#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D6 §5.3 re-check: corrected v3 (soft) Chamfer vs D5 depth-regression Chamfer.

D5 predictions: per-object .npy = per-ray normalized depth in [0,1] -> r = p*rmax.
D6 v3 soft:     per-ray soft-argmax BIN index (fixed _soft.npy) -> r = b/95*rmax.
Same cov floor (0.02), same GT surf_points, same symmetric Chamfer score.
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
D5 = os.path.join(ROOT, "output", "gso_d5")
D5_PRED = os.path.join(ROOT, "d5_pred_v2split_norm")
D6_PRED = os.path.join(ROOT, "d6_pred_v3")
N_PHI, N_THETA, N_BINS = 128, 64, 96


def dirs_grid():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                    axis=-1).reshape(-1, 3).astype(np.float32)


def score(r, rmax, cov, dirs, floor, gt):
    keep = (cov >= floor) & (r >= 0.02) & (r < rmax)
    pc = (r[keep, None] * dirs[keep]).astype(np.float32)
    if len(pc) == 0:
        return None
    kd_pc, kd_gt = cKDTree(pc), cKDTree(gt)
    d1, _ = kd_gt.query(pc)
    d2, _ = kd_pc.query(gt)
    return float((d1.mean() + d2.mean()) / 2)


def main():
    dirs = dirs_grid()
    d5_objs = sorted(f[:-4] for f in os.listdir(D5_PRED) if f.endswith(".npy"))
    d6_objs = sorted({f[:-9] for f in os.listdir(D6_PRED) if f.endswith("_soft.npy")})
    shared = sorted(set(d5_objs) & set(d6_objs))
    print("d5=%d d6=%d shared=%d" % (len(d5_objs), len(d6_objs), len(shared)))
    w5 = w6 = 0
    rows = []
    for n in shared:
        pd = os.path.join(D4, n, "profiles")
        rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
        cov = np.load(os.path.join(pd, "coverage.npy"))
        gt = np.load(os.path.join(D5, n, "surf_points.npy")).astype(np.float32)
        r5 = np.load(os.path.join(D5_PRED, n + ".npy")).astype(np.float32) * rmax
        r6 = (np.load(os.path.join(D6_PRED, n + "_soft.npy")).astype(np.float32)
              / (N_BINS - 1) * rmax)
        ch5 = score(r5, rmax, cov, dirs, 0.02, gt)
        ch6 = score(r6, rmax, cov, dirs, 0.02, gt)
        if ch5 is None or ch6 is None:
            rows.append((n, ch5, ch6)); continue
        w5 += ch5 < ch6
        w6 += ch6 < ch5
        rows.append((n, ch5, ch6))
        print("%-42s D5=%.4f  v3soft=%.4f  (v3-D5=%.4f)  %s"
              % (n, ch5, ch6, ch6 - ch5, "v3 WINS" if ch6 < ch5 else "D5 wins"))
    r = [(n, a, b) for n, a, b in rows if a is not None and b is not None]
    if r:
        print("\nMEDIAN: D5=%.4f v3soft=%.4f  (wins D5:%d v3:%d ties:%d)"
              % (np.median([a for _, a, _ in r]), np.median([b for _, b, _ in r]),
                 w5, w6, len(r) - w5 - w6))


if __name__ == "__main__":
    main()
