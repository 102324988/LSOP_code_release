#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Corrected D6/D7-family evaluator (authoritative for the reports).

Computes, per object, hard AND soft readouts for both depth-med and
end-to-end Chamfer, on top of the FIXED _soft.npy artifacts (regenerated
from _prof.npy by d7_softfix.py). Also reproduces the D6-report readout
table (hard/soft at cov_floor 0.02 and 0.10) so the "soft beats hard"
claim is re-checked on correct data.

Usage: python d7_eval_fixed.py <pred_dir> [cov_floor]
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
D5 = os.path.join(ROOT, "output", "gso_d5")
N_PHI, N_THETA, N_BINS = 128, 64, 96


def dirs_grid():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                    axis=-1).reshape(-1, 3).astype(np.float32)


def score(depth_bin, rmax, cov, dirs, floor, gt):
    r = (depth_bin / (N_BINS - 1)) * rmax
    keep = (cov >= floor) & (r >= 0.02) & (r < rmax)
    pc = (r[keep, None] * dirs[keep]).astype(np.float32)
    if len(pc) == 0:
        return None
    kd_pc, kd_gt = cKDTree(pc), cKDTree(gt)
    d1, _ = kd_gt.query(pc)
    d2, _ = kd_pc.query(gt)
    return float((d1.mean() + d2.mean()) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--cov_floor", type=float, default=0.02)
    args = ap.parse_args()
    dirs = dirs_grid()
    objs = sorted({f[:-4] for f in os.listdir(args.pred_dir)
                   if f.endswith(".npy") and not any(
                       f.endswith(s) for s in
                       ("_cov.npy", "_soft.npy", "_prof.npy", "_predpeak.npy",
                        "_gtprof.npy", "_gtpeak.npy"))})
    print("pred_dir=%s objects=%d" % (args.pred_dir, len(objs)))
    ds, dh, cs, ch = [], [], [], []
    for n in objs:
        soft = np.load(os.path.join(args.pred_dir, n + "_soft.npy")).astype(np.float32)
        hard = np.load(os.path.join(args.pred_dir, n + ".npy")).astype(np.float32)
        cov = np.load(os.path.join(args.pred_dir, n + "_cov.npy")).astype(np.float32)
        pd = os.path.join(D4, n, "profiles")
        peak = np.load(os.path.join(pd, "depth_peak.npy")).astype(np.float32)
        rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
        gt = np.load(os.path.join(D5, n, "surf_points.npy")).astype(np.float32)
        m = cov >= args.cov_floor
        gtn = peak[m] / rmax
        dmed_s = float(np.median(np.abs(soft[m] / (N_BINS - 1) - gtn)))
        dmed_h = float(np.median(np.abs(hard[m] / (N_BINS - 1) - gtn)))
        ch_s = score(soft, rmax, cov, dirs, args.cov_floor, gt)
        ch_h = score(hard, rmax, cov, dirs, args.cov_floor, gt)
        # readout-table rows at elevated floor
        ch_s10 = score(soft, rmax, cov, dirs, 0.10, gt)
        ch_h10 = score(hard, rmax, cov, dirs, 0.10, gt)
        ds.append(dmed_s); dh.append(dmed_h)
        if ch_s is not None:
            cs.append(ch_s)
        if ch_h is not None:
            ch.append(ch_h)
        fmt = lambda v: ("%.4f" % v) if v is not None else "   NA"
        print("%-42s dmed_s=%s dmed_h=%s ch_s=%s ch_h=%s "
              "ch_s10=%s ch_h10=%s"
              % (n, fmt(dmed_s), fmt(dmed_h), fmt(ch_s), fmt(ch_h),
                 fmt(ch_s10), fmt(ch_h10)))
    med = lambda v: np.median(v) if len(v) else float("nan")
    print("\nMEDIAN: dmed_s=%.4f dmed_h=%.4f ch_s=%.4f ch_h=%.4f (hard-cloud objects skipped: %d/%d)"
          % (med(ds), med(dh), med(cs), med(ch), 60-len(ch), 60))
    print("readout table (floor .02): soft=%.4f hard=%.4f"
          % (med(cs), med(ch)))


if __name__ == "__main__":
    main()
