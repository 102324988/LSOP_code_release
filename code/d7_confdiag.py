#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decisive D7 confidence diagnostic: does the predicted peak (confidence)
actually correlate with per-ray depth ERROR? For each object, on cov>=0.02
rays:
  corr(conf, err)  where err = |soft_depth_bin/95 - peak/rmax| (normalized),
                    conf = predicted peak height.
A strongly negative corr means low-confidence rays are indeed unreliable
(filtering works). Also computes corr(GT peak, err) and corr(GT cov, err) to
see which signal the GT offers. Runs on a pred_dir; call twice for D7 and D6.
Usage: python d7_confdiag.py <pred_dir>
"""
import json
import os
import sys

import numpy as np

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
N_BINS = 96

pred_dir = sys.argv[1]
objs = sorted({f[:-4] for f in os.listdir(pred_dir)
               if f.endswith(".npy") and not any(
                   f.endswith(s) for s in
                   ("_cov.npy", "_soft.npy", "_prof.npy", "_predpeak.npy",
                    "_gtprof.npy", "_gtpeak.npy"))})
is_d7 = os.path.exists(os.path.join(pred_dir, objs[0] + "_gtpeak.npy"))
print("pred_dir=%s objects=%d is_D7=%s" % (pred_dir, len(objs), is_d7))
rows = []
for n in objs:
    soft = np.load(os.path.join(pred_dir, n + "_soft.npy")).astype(np.float32)
    prof = np.load(os.path.join(pred_dir, n + "_prof.npy")).astype(np.float32)
    cov = np.load(os.path.join(pred_dir, n + "_cov.npy")).astype(np.float32)
    conf = prof.max(-1)
    pd = os.path.join(D4, n, "profiles")
    peak = np.load(os.path.join(pd, "depth_peak.npy")).astype(np.float32)
    rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
    m = cov >= 0.02
    err = np.abs(soft[m] / (N_BINS - 1) - peak[m] / rmax)
    cc = np.corrcoef(conf[m], err)[0, 1]
    # quantile check: error of bottom-conf quintile vs top-conf quintile
    q = np.percentile(conf[m], [20, 80])
    lo = err[conf[m] <= q[0]]; hi = err[conf[m] >= q[1]]
    row = (n, float(cc), float(lo.mean()), float(hi.mean()))
    if is_d7:
        gtpeak = np.load(os.path.join(pred_dir, n + "_gtpeak.npy")).astype(np.float32)[m]
        row += (float(np.corrcoef(gtpeak, err)[0, 1]),
                float(np.corrcoef(cov[m], err)[0, 1]))
    rows.append(row)

print("%-42s %8s %9s %9s%s" % (
    "object", "corrC_err", "err|loC", "err|hiC",
    "   corrGTp_err corrGcov_err" if is_d7 else ""))
for r in rows:
    print("%-42s %+8.3f %9.4f %9.4f%s" % (
        r[0][:42], r[1], r[2], r[3],
        ("   %+8.3f %+8.3f" % (r[4], r[5])) if is_d7 else ""))
ccs = [r[1] for r in rows]
gaps = [r[2] - r[3] for r in rows]
print("\nMEDIAN corr(conf,err) = %.3f  (negative = low conf -> high err = useful)"
      % np.median(ccs))
print("MEDIAN err(low-conf) - err(high-conf) = %.4f  (positive gap = filter helps)"
      % np.median(gaps))
print("objects with corr<-0.1: %d/%d" % (sum(c < -0.1 for c in ccs), len(rows)))
if is_d7:
    print("MEDIAN corr(GTpeak,err) = %.3f   corr(GTcov,err) = %.3f"
          % (np.median([r[4] for r in rows]), np.median([r[5] for r in rows])))
