#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D7 unified evaluator. Runs on artifacts from either D6-v3 (max-norm profiles)
or D7 (raw-rescaled profiles), so the confidence-filter experiment is an
apples-to-apples A/B on identical code:
  - depth-med: soft-argmax bin/95 vs depth_peak/rmax on cov>=0.02 rays
  - end-to-end Chamfer / F@0.05 / fa@0.05 from the soft-argmax surface
  - confidence filter sweep: keep rays with predicted peak >= tau, tau in
    {0, .15, .3, .45, .6}; predicted peak = prof.max(1). In D6 the max-norm
    target saturated peaks to ~1 so this filter could not discriminate and
    monotonically hurt (D6 report sec 6.2); in D7 raw-rescaled peaks vary and
    should give a working confidence gate.
  - (D7 only, auto-detected via _gtpeak.npy): calibration corr between
    predicted and GT peak height on cov>=0.02 rays, and prof-L1.
Usage: python d7_eval.py <pred_dir> [--cov_floor 0.02]
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
    ch = float((d1.mean() + d2.mean()) / 2)
    f1 = float((d1 < 0.05).mean()); f2 = float((d2 < 0.05).mean())
    f05 = 2 * f1 * f2 / (f1 + f2 + 1e-12)
    fa = float((d2 < 0.05).mean())
    return ch, f05, fa, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--cov_floor", type=float, default=0.02)
    ap.add_argument("--taus", default="0.00,0.15,0.30,0.45,0.60")
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]
    dirs = dirs_grid()
    objs = sorted({f[:-4] for f in os.listdir(args.pred_dir)
                   if f.endswith(".npy") and not any(
                       f.endswith(s) for s in
                       ("_cov.npy", "_soft.npy", "_prof.npy", "_predpeak.npy",
                        "_gtprof.npy", "_gtpeak.npy"))})
    is_d7 = os.path.exists(os.path.join(args.pred_dir, objs[0] + "_gtpeak.npy"))
    print("pred_dir=%s objects=%d is_D7=%s taus=%s"
          % (args.pred_dir, len(objs), is_d7, taus))

    rows = []
    for n in objs:
        soft = np.load(os.path.join(args.pred_dir, n + "_soft.npy")).astype(np.float32)
        hard = np.load(os.path.join(args.pred_dir, n + ".npy")).astype(np.float32)
        prof = np.load(os.path.join(args.pred_dir, n + "_prof.npy")).astype(np.float32)
        cov = np.load(os.path.join(args.pred_dir, n + "_cov.npy")).astype(np.float32)
        predpeak = prof.max(-1)
        pd = os.path.join(D4, n, "profiles")
        peak = np.load(os.path.join(pd, "depth_peak.npy")).astype(np.float32)
        rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
        gt = np.load(os.path.join(D5, n, "surf_points.npy")).astype(np.float32)
        m = cov >= args.cov_floor
        gtn = peak[m] / rmax
        dmed_s = float(np.median(np.abs(soft[m] / (N_BINS - 1) - gtn)))
        dmed_h = float(np.median(np.abs(hard[m] / (N_BINS - 1) - gtn)))
        row = {"n": n, "dmed_s": dmed_s, "dmed_h": dmed_h}
        s_base = score(soft, rmax, cov, dirs, args.cov_floor, gt)
        s_hard = score(hard, rmax, cov, dirs, args.cov_floor, gt)
        row["ch_s"], row["f05_s"], row["fa_s"], row["rays_s"] = s_base
        row["ch_h"], _, _, _ = s_hard
        # confidence filter sweep (soft readout)
        row["ch_tau"] = {}
        for t in taus:
            s = score(soft, rmax, cov, dirs, args.cov_floor, gt,
                      keep_extra=predpeak >= t)
            row["ch_tau"][t] = None if s is None else s[0]
        if is_d7:
            gtpeak = np.load(os.path.join(args.pred_dir, n + "_gtpeak.npy")).astype(np.float32)
            gtprof = np.load(os.path.join(args.pred_dir, n + "_gtprof.npy")).astype(np.float32)
            cc = np.corrcoef(predpeak[m], gtpeak[m])[0, 1]
            row["corr"] = float(cc)
            row["prof_l1"] = float(np.abs(prof[m] - gtprof[m]).mean())
        rows.append(row)

    print("\n%-42s %8s %8s %8s %8s %8s  %s" % (
        "object", "dmed_s", "dmed_h", "ch_s", "ch_base", "ch_best", "tau*"))
    dmeds, chs, corrs, l1s = [], [], [], []
    best_gains = []
    for r in rows:
        tvals = [t for t in taus if r["ch_tau"][t] is not None]
        tstar = min(tvals, key=lambda t: r["ch_tau"][t]) if tvals else None
        ch_best = r["ch_tau"][tstar] if tstar is not None else float("nan")
        gain = r["ch_s"] - ch_best
        best_gains.append(gain)
        line = "%-42s %8.4f %8.4f %8.4f %8.4f %8.4f  tau=%s" % (
            r["n"], r["dmed_s"], r["dmed_h"], r["ch_s"], r["ch_s"], ch_best,
            tstar if tstar is not None else "-")
        if is_d7:
            line += "  corr=%.3f l1=%.4f" % (r["corr"], r["prof_l1"])
            corrs.append(r["corr"]); l1s.append(r["prof_l1"])
        print(line)
        dmeds.append(r["dmed_s"]); chs.append(r["ch_s"])
    print("\nMEDIAN over %d objects:" % len(rows))
    print("  depth-med soft = %.4f   hard = %.4f" % (np.median(dmeds), np.median([r["dmed_h"] for r in rows])))
    print("  chamfer soft   = %.4f (base tau=0)" % np.median(chs))
    print("  best-tau gain  = median %.4f  (objects improved: %d/%d)"
          % (np.median(best_gains), sum(g > 1e-4 for g in best_gains), len(best_gains)))
    if is_d7:
        print("  calib corr     = median %.3f" % np.median(corrs))
        print("  prof-L1        = median %.4f" % np.median(l1s))


if __name__ == "__main__":
    main()
