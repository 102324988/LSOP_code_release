#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3b morphology alternative: post-hoc power sharpening p^gamma.

Honesty check of the design-doc idea "decode-side p^gamma sharpening" —
does a NON-LEARNED power transform of the stored M3b predicted profiles
narrow FWHM toward GT without wrecking depth accuracy?

For each split (val/test), reload the stored sigmoid profiles and GT, apply
p^gamma for gamma in {1,2,4}, and re-evaluate the M3-battery numbers with
the exact m3b_infer protocol:
  - FWHM ratio (pred/GT, on covered rays with GT fwhm>=2). Scale-invariant,
    so the gamma effect is real and comparable to the 2.00/1.83 reported.
  - peak (raw sigmoid scale: p<1 lowers it — the honesty tension vs the
    peak>=0.5 gate).
  - dmed_s / dmed_h (soft/hard depth; sharpening moves soft toward hard).
Report per split per gamma, vs the D8 references.

Usage: python m3b_pgamma.py [--out m3b_pgamma]
"""
import argparse
import json
import os

import numpy as np

from m1_train_vae import N_BINS, load_object
from m3b_train_preray import fwhm_np

ROOT = "/root/e0lab/e0"
D8_CKPT = "d8_mean_pool"
GAMMAS = [1.0, 2.0, 4.0]
SPLITS = ["val", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="m3b_pgamma")
    args = ap.parse_args()
    d8meta = json.load(open(os.path.join(ROOT, D8_CKPT, "meta.json")))
    axis = np.arange(N_BINS)
    out = {}
    for split in SPLITS:
        names = d8meta["test_names"] if split == "test" else d8meta["val_names"]
        src = os.path.join(ROOT, "m3b_preray_" + split, "profiles")
        out[split] = {"n": len(names)}
        for gamma in GAMMAS:
            fwhm_rat, peaks, dmed_s, dmed_h = [], [], [], []
            for n in names:
                pred = np.load(os.path.join(src, n + "_prof.npy")).astype(np.float32)
                sh, cov, peak, rmax = load_object(n)
                cov = cov.numpy()
                m = cov >= 0.02
                p = pred[m] ** gamma
                # morphology (scale-invariant FWHM, raw peak value)
                fw_p = fwhm_np(p)
                fw_g = fwhm_np(sh.numpy()[m])
                good = fw_g >= 2
                fwhm_rat.append(fw_p[good] / np.maximum(fw_g[good], 1e-6))
                peaks.append(p.max(-1))
                # depth (soft/hard on covered rays, m3b_infer protocol)
                soft = (p * axis[None, :]).sum(-1) / (p.sum(-1) + 1e-6)
                hard = p.argmax(-1).astype(np.float32)
                gtn = peak.numpy()[m] / rmax
                dmed_s.append(np.median(np.abs(soft / (N_BINS - 1) - gtn)))
                dmed_h.append(np.median(np.abs(hard / (N_BINS - 1) - gtn)))
            fr = np.concatenate(fwhm_rat)
            out[split]["gamma=%.1f" % gamma] = {
                "fwhm_ratio_med": float(np.median(fr)),
                "fwhm_ratio_p25": float(np.percentile(fr, 25)),
                "fwhm_ratio_p75": float(np.percentile(fr, 75)),
                "peak_med": float(np.median(np.concatenate(peaks))),
                "dmed_s_med": float(np.median(dmed_s)),
                "dmed_s_mean": float(np.mean(dmed_s)),
                "dmed_h_med": float(np.median(dmed_h)),
            }
            r = out[split]["gamma=%.1f" % gamma]
            print("M3b %-4s gamma=%.1f  fwhm_ratio=%.2f [%.2f-%.2f]  "
                  "peak=%.3f  dmed_s=%.4f  dmed_h=%.4f"
                  % (split, gamma, r["fwhm_ratio_med"], r["fwhm_ratio_p25"],
                     r["fwhm_ratio_p75"], r["peak_med"], r["dmed_s_med"],
                     r["dmed_h_med"]), flush=True)
    out["_refs"] = {"D8_val_dmed_s": 0.0338, "D8_test_dmed_s": 0.0384,
                    "gate_fwhm": 1.5, "gate_peak": 0.5, "gt_fwhm": 1.0,
                    "M3b_val_dmed_s": 0.0364, "M3b_test_dmed_s": 0.0356,
                    "M3b_val_fwhm": 2.00, "M3b_test_fwhm": 1.83}
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, args.out, "summary.json"), "w"),
              indent=1)
    print("saved -> %s/summary.json" % args.out, flush=True)


if __name__ == "__main__":
    main()
