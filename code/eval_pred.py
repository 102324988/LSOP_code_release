#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end eval of D5 baseline predictions: predicted depth_peak field ->
surface point cloud -> Chamfer/F-score vs GT surf_points (D4b assets).
Usage: python eval_pred.py [cov_floor]
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/root/e0lab/e0"
D5 = os.path.join(ROOT, "output", "gso_d5")
PRED = os.path.join(ROOT, "d5_pred")
N_PHI, N_THETA = 128, 64


def dirs_grid():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                    axis=-1).reshape(-1, 3).astype(np.float32)


def main():
    cov_floor = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    dirs = dirs_grid()
    files = sorted(f for f in os.listdir(PRED) if f.endswith(".npy"))
    print("n_pred_objects=%d (val subset)" % len(files))
    print("%-38s n_pts  chamfer f@.05  f@.10  fa@.05" % "name")
    med_ch = []
    for f in files:
        name = f[:-4]
        pred = np.load(os.path.join(PRED, f)).astype(np.float32)
        meta = json.load(open(os.path.join(D5, name, "meta.json")))
        rmax = meta["rmax"]
        cov = np.load(os.path.join(D5, name, "coverage.npy")) if os.path.exists(
            os.path.join(D5, name, "coverage.npy")) else np.ones(pred.shape[0])
        gt = np.load(os.path.join(D5, name, "surf_points.npy")).astype(np.float32)
        r = pred * rmax
        keep = (cov >= cov_floor) & (r >= 0.02)
        pc = (r[keep, None] * dirs[keep]).astype(np.float32)
        if len(pc) == 0:
            print("%-38s no pts" % name)
            continue
        kd_pc, kd_gt = cKDTree(pc), cKDTree(gt)
        d1, _ = kd_gt.query(pc)
        d2, _ = kd_pc.query(gt)
        ch = float((d1.mean() + d2.mean()) / 2)
        vals = [ch]
        for t in (0.05, 0.1):
            f1 = float((d1 < t).mean()); f2 = float((d2 < t).mean())
            vals.append(2 * f1 * f2 / (f1 + f2 + 1e-12))
        fa5 = float((d2 < 0.05).mean())
        med_ch.append(ch)
        print("%-38s %5d %.4f %.4f %.4f %.3f" % (name, len(pc), *vals, fa5))
    if med_ch:
        print("median chamfer=%.4f" % np.median(med_ch))


if __name__ == "__main__":
    main()
