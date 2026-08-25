#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which PER-RAY signals actually predict the model's depth ERROR?
For each val object, on cov>=0.02 rays, err = |soft/95 - peak/rmax|.
Signals tested (all computable from the model's own output, no GT):
  dis   = |soft_bin - argmax_bin|/95   (soft-vs-hard disagreement, ambiguity)
  ent   = -sum(p*log p) of the predicted profile (normalized)
  sharp = maxbin frac (predicted peak's share of the total mass)
  predpeak
Plus GT refs: GT peak, GT coverage.
Usage: python d7_sigdiag.py <pred_dir>
"""
import json
import os
import sys

import numpy as np

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
NB = 96

pred_dir = sys.argv[1]
objs = sorted({f[:-4] for f in os.listdir(pred_dir)
               if f.endswith(".npy") and not any(
                   f.endswith(s) for s in
                   ("_cov.npy", "_soft.npy", "_prof.npy", "_predpeak.npy",
                    "_gtprof.npy", "_gtpeak.npy"))})
print("pred_dir=%s objects=%d" % (pred_dir, len(objs)))
A = np.arange(NB, dtype=np.float32)
sig_cols = ["dis", "ent", "sharp", "predpeak", "gtpeak", "gtcov"]
med = {k: [] for k in sig_cols}
useful = {k: 0 for k in sig_cols}
print("%-30s %8s %8s %8s %8s %8s %8s" % (
    "object", *["corr(err,%s)" % c for c in sig_cols]))
for n in objs:
    soft = np.load(os.path.join(pred_dir, n + "_soft.npy")).astype(np.float32)
    prof = np.load(os.path.join(pred_dir, n + "_prof.npy")).astype(np.float32)
    cov = np.load(os.path.join(pred_dir, n + "_cov.npy")).astype(np.float32)
    pd = os.path.join(D4, n, "profiles")
    peak = np.load(os.path.join(pd, "depth_peak.npy")).astype(np.float32)
    rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
    m = cov >= 0.02
    err = np.abs(soft[m] / (NB - 1) - peak[m] / rmax)
    p = prof[m]
    pmax = p.max(-1, keepdims=True)
    pn = p / (pmax + 1e-9)
    hard = p.argmax(-1).astype(np.float32)
    dis = np.abs(soft[m] - hard) / (NB - 1)
    plogp = np.where(p > 1e-9, p * np.log(p + 1e-9), 0.0)
    ent = -plogp.sum(-1) / np.log(NB)  # [0,1]
    sharp = pmax[..., 0] / (p.sum(-1) + 1e-9)
    sigs = {"dis": dis, "ent": ent, "sharp": sharp, "predpeak": pmax[..., 0]}
    gt_peak = np.load(os.path.join(pred_dir, n + "_gtpeak.npy")).astype(np.float32)[m]
    sigs["gtpeak"] = gt_peak
    sigs["gtcov"] = cov[m]
    line = [n[:30]]
    for k in sig_cols:
        c = float(np.corrcoef(sigs[k], err)[0, 1])
        med[k].append(c)
        if c < -0.15:
            useful[k] += 1
        line.append("%+8.3f" % c)
    print(" ".join(line))
print("\nMEDIAN corr(err, ...):")
for k in sig_cols:
    print("  %-9s %+6.3f   (objects with corr<-0.15: %d/%d)"
          % (k, np.median(med[k]), useful[k], len(objs)))
