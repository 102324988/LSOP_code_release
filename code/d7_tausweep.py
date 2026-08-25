#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Confidence-filter tau sweep on the FIXED _soft.npy (regenerated from
_prof.npy by d7_softfix.py). Replaces the buggy-soft sweep in D6 §6.2 /
D7 §4.1. For each object, keep rays with predicted peak (prof.max(-1)) >= tau,
compute median soft Chamfer over the object set per tau.
Usage: python d7_tausweep.py <pred_dir>
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
D5 = os.path.join(ROOT, "output", "gso_d5")
N_PHI, N_THETA, N_BINS = 128, 64, 96
TAUS = [0.00, 0.15, 0.30, 0.45, 0.60, 0.75]


def dirs_grid():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                    axis=-1).reshape(-1, 3).astype(np.float32)


def score(depth_bin, rmax, cov, dirs, floor, gt, keep_extra=None):
    r = (depth_bin / (N_BINS - 1)) * rmax
    keep = (cov >= floor) & (r >= 0.02) & (r < rmax)
    if keep_extra is not None:
        keep = keep & keep_extra
    pc = (r[keep, None] * dirs[keep]).astype(np.float32)
    if len(pc) == 0:
        return None
    kd_pc, kd_gt = cKDTree(pc), cKDTree(gt)
    d1, _ = kd_gt.query(pc)
    d2, _ = kd_pc.query(gt)
    return float((d1.mean() + d2.mean()) / 2)


def main():
    pred_dir = sys.argv[1]
    dirs = dirs_grid()
    objs = sorted({f[:-9] for f in os.listdir(pred_dir) if f.endswith("_prof.npy")})
    acc = {t: [] for t in TAUS}
    for n in objs:
        soft = np.load(os.path.join(pred_dir, n + "_soft.npy")).astype(np.float32)
        prof = np.load(os.path.join(pred_dir, n + "_prof.npy")).astype(np.float32)
        cov = np.load(os.path.join(pred_dir, n + "_cov.npy")).astype(np.float32)
        pd = os.path.join(D4, n, "profiles")
        rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
        gt = np.load(os.path.join(D5, n, "surf_points.npy")).astype(np.float32)
        for t in TAUS:
            c = score(soft, rmax, cov, dirs, 0.02, gt,
                      keep_extra=prof.max(-1) >= t)
            if c is not None:
                acc[t].append(c)
    print("pred_dir=%s objects=%d" % (pred_dir, len(objs)))
    for t in TAUS:
        print("  tau=%.2f  median ch=%.4f  (n=%d)" % (t, np.median(acc[t]), len(acc[t])))


if __name__ == "__main__":
    main()
