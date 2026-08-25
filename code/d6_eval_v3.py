#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Eval for D6-v3 saved artifacts: full profiles -> hard-argmax vs soft-argmax
surface readouts, plus coverage-filtered readout (confidence ceiling).
Usage: python d6_eval_v3.py <pred_dir> [cov_floor]
Per object: <n>.npy (hard-argmax bin), <n>_soft.npy (soft-argmax bin),
<n>_prof.npy (full 96-bin profile), <n>_cov.npy (GT coverage).
Reports chamfer / F@0.05 / fa@0.05 for hard and soft readouts, each at the
base cov_floor and at an elevated cov_floor (drop low-confidence rays).
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/root/e0lab/e0"
D5 = os.path.join(ROOT, "output", "gso_d5")
N_PHI, N_THETA, N_BINS = 128, 64, 96


def dirs_grid():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                    axis=-1).reshape(-1, 3).astype(np.float32)


def score(depth, rmax, cov, dirs, floor, gt):
    r = (depth / (N_BINS - 1)) * rmax
    keep = (cov >= floor) & (r >= 0.02) & (r < rmax)
    pc = (r[keep, None] * dirs[keep]).astype(np.float32)
    if len(pc) == 0:
        return None
    kd_pc, kd_gt = cKDTree(pc), cKDTree(gt)
    d1, _ = kd_gt.query(pc)
    d2, _ = kd_pc.query(gt)
    ch = float((d1.mean() + d2.mean()) / 2)
    f1 = float((d1 < 0.05).mean()); f2 = float((d2 < 0.05).mean())
    f05 = 2 * f1 * f2 / (f1 + f2 + 1e-12)
    fa = float((d2 < 0.05).mean())
    return ch, f05, fa


def main():
    pred_dir = sys.argv[1]
    base_floor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    hi_floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    dirs = dirs_grid()
    objs = sorted({f[:-4] for f in os.listdir(pred_dir)
                   if f.endswith(".npy") and not any(
                       f.endswith(s) for s in ("_cov.npy", "_soft.npy", "_prof.npy"))})
    print("n_objects=%d  floors=%.2f/%.2f" % (len(objs), base_floor, hi_floor))
    print("%-40s %10s %10s %10s %10s" % ("", "hard-b", "soft-b", "hard-hi", "soft-hi"))
    agg = {"hard_b": [], "soft_b": [], "hard_hi": [], "soft_hi": []}
    for n in objs:
        hard = np.load(os.path.join(pred_dir, n + ".npy")).astype(np.float32)
        soft = np.load(os.path.join(pred_dir, n + "_soft.npy")).astype(np.float32)
        cov = np.load(os.path.join(pred_dir, n + "_cov.npy")).astype(np.float32)
        pd = os.path.join(ROOT, "output", "gso_d4", n, "profiles")
        rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
        gt = np.load(os.path.join(D5, n, "surf_points.npy")).astype(np.float32)
        vals = {"hard_b": score(hard, rmax, cov, dirs, base_floor, gt),
                "soft_b": score(soft, rmax, cov, dirs, base_floor, gt),
                "hard_hi": score(hard, rmax, cov, dirs, hi_floor, gt),
                "soft_hi": score(soft, rmax, cov, dirs, hi_floor, gt)}
        line = "%-40s" % n
        for k in ("hard_b", "soft_b", "hard_hi", "soft_hi"):
            s = vals[k]
            if s:
                line += " %5.3f/%4.2f/%4.2f" % (s[0], s[1], s[2])
                agg[k].append(s[0])
            else:
                line += "    no pts"
        print(line)
    print("median chamfer  hard-b=%.4f soft-b=%.4f hard-hi=%.4f soft-hi=%.4f"
          % tuple(np.median(agg[k]) for k in ("hard_b", "soft_b", "hard_hi", "soft_hi")))


if __name__ == "__main__":
    main()
