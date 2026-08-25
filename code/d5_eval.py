#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Corrected eval for D5 predicted depth fields (sigmoid normalized-depth format).
Usage: python d5_eval.py <pred_dir> [cov_floor]
Pred files: per-object .npy holding per-ray normalized depth in [0,1].
radius = pred*rmax -> points -> Chamfer/F vs GT surf_points. depth-med =
|pred - depth_peak/rmax| median over covered rays (normalized units).
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


def main():
    pred_dir = sys.argv[1]
    cov_floor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    dirs = dirs_grid()
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".npy")
                   and not f.endswith("_cov.npy"))
    print("n_pred_objects=%d" % len(files))
    print("%-40s n_pts chamfer f@.05 f@.10 fa@.05 depth-med" % "name")
    med_ch, med_dep = [], []
    for f in files:
        name = f[:-4]
        pred = np.load(os.path.join(pred_dir, f)).astype(np.float32)  # normalized depth
        pd = os.path.join(ROOT, "output", "gso_d4", name, "profiles")
        rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
        cov = np.load(os.path.join(pd, "coverage.npy"))
        peak = np.load(os.path.join(pd, "depth_peak.npy")).astype(np.float32)
        gt = np.load(os.path.join(D5, name, "surf_points.npy")).astype(np.float32)
        r = pred * rmax  # normalized depth -> absolute radius
        keep = (cov >= cov_floor) & (r >= 0.02) & (r < rmax)
        pc = (r[keep, None] * dirs[keep]).astype(np.float32)
        dep = np.abs(pred - peak / rmax)
        med_dep.append(float(np.median(dep[cov >= cov_floor])))
        if len(pc) == 0:
            print("%-40s no pts" % name)
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
        print("%-40s %5d %.4f %.4f %.4f %.3f %.4f" % (name, len(pc), *vals, fa5,
                                                      dep[cov >= cov_floor].mean()))
    if med_ch:
        print("median chamfer=%.4f  median depth-med=%.4f" % (np.median(med_ch),
                                                              np.median(med_dep)))


if __name__ == "__main__":
    main()
