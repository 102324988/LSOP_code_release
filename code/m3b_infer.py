#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3b: authoritative evaluation (val/test) for the per-ray decoder.

Runs the frozen M3b model over a split, saves per-object profiles, and
reports the full M1/M2/D8-consistent battery:
  - depth dmed_s/dmed_h (soft/hard, d7_eval_fixed protocol, cov>=0.02)
  - Chamfer ch_s/ch_h (soft/hard clouds vs D5 GT surf points)
  - morphology: FWHM ratio (pred/GT, m1_diag protocol) + peak
  - collapse check: cross-object mean-profile correlation (->1 = collapsed)

Usage: python m3b_infer.py --m3b m3b_preray --split test --out m3b_preray_test
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from m1_train_vae import N_BINS, load_object, ray_encodings
from d7_eval_fixed import score, dirs_grid
from m3b_train_preray import Model, compute_grid, load_views, fwhm_np, \
    load_from_d8

ROOT = "/root/e0lab/e0"
D8_CKPT = "d8_mean_pool"
D5 = os.path.join(ROOT, "output", "gso_d5")  # GT surf points (d7 protocol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3b", default="m3b_preray")
    ap.add_argument("--out", default="m3b_preray_test")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    m3b = json.load(open(os.path.join(ROOT, args.m3b, "meta.json")))
    d8meta = json.load(open(os.path.join(ROOT, D8_CKPT, "meta.json")))
    names = d8meta["test_names"] if args.split == "test" else d8meta["val_names"]
    print("split=%s n=%d" % (args.split, len(names)), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, dirs = ray_encodings()
    dirs_np = dirs.numpy()
    ray_pe = ray_pe.to(device)

    model = Model(rdim=m3b.get("rdim", 128)).to(device)
    load_from_d8(model, os.path.join(ROOT, D8_CKPT, "model.pt"))
    model.load_state_dict(torch.load(os.path.join(ROOT, args.m3b, "head.pt"),
                                     map_location="cpu"), strict=False)
    model.eval()
    print("M3b model loaded (D8 backbone+proj, trained head).", flush=True)

    from multiprocessing.dummy import Pool
    with Pool(8) as pool:
        views = dict(pool.map(lambda n: (n, load_views(n)), names, chunksize=4))
        grids = dict(pool.map(lambda n: (n, compute_grid(n, dirs_np)),
                              names, chunksize=4))
        profs = dict(pool.map(lambda n: (n, load_object(n)), names, chunksize=4))
    print("data loaded.", flush=True)

    gt_dirs = dirs_grid()
    dmed_s, dmed_h, ch_s, ch_h = [], [], [], []
    fwhm_rat, peaks = [], []
    mean_profs = []
    per = []
    axis = np.arange(N_BINS)
    os.makedirs(os.path.join(ROOT, args.out, "profiles"), exist_ok=True)
    with torch.no_grad():
        for n in names:
            imgs, feats = views[n]
            grd = torch.from_numpy(grids[n])[None].to(device)
            logit = model(imgs[None].to(device), feats[None].to(device),
                          ray_pe, grd)
            pred = torch.sigmoid(logit)[0].cpu().numpy()
            np.save(os.path.join(ROOT, args.out, "profiles", n + "_prof.npy"),
                    pred.astype(np.float32))
            sh, cov, peak, rmax = profs[n]
            cov = cov.numpy()
            m = cov >= 0.02
            # depth (soft/hard on covered rays)
            soft_all = (pred * axis[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
            hard_all = pred.argmax(-1).astype(np.float32)
            soft, hard = soft_all[m], hard_all[m]
            gtn = peak.numpy()[m] / rmax
            dmed_s.append(np.median(np.abs(soft / (N_BINS - 1) - gtn)))
            dmed_h.append(np.median(np.abs(hard / (N_BINS - 1) - gtn)))
            # chamfer (score filters internally on FULL 8192 rays)
            gt = np.load(os.path.join(D5, n, "surf_points.npy")).astype(np.float32)
            ch_s.append(score(soft_all, rmax, cov, gt_dirs, 0.02, gt))
            ch_h.append(score(hard_all, rmax, cov, gt_dirs, 0.02, gt))
            # morphology
            fw_p = fwhm_np(pred[m])
            fw_g = fwhm_np(sh.numpy()[m])
            good = fw_g >= 2
            fwhm_rat.append(fw_p[good] / np.maximum(fw_g[good], 1e-6))
            peaks.append(pred[m].max(-1))
            mean_profs.append(pred[m].mean(0))
            per.append({"name": n,
                        "dmed_s": float(np.median(
                            np.abs(soft / (N_BINS - 1) - gtn)))})
            print("%-42s dmed_s=%.4f ch_s=%.4f" % (n, dmed_s[-1], ch_s[-1]),
                  flush=True)

    # collapse check: mean-profile correlation across objects
    M = np.stack(mean_profs)                      # (n, 96)
    M = M / (M.std(0, keepdims=True) + 1e-6)
    xcorr = float((M @ M.T).mean())
    fr = np.concatenate(fwhm_rat)
    stats = {"split": args.split, "n": len(names),
             "dmed_s_med": float(np.median(dmed_s)),
             "dmed_s_mean": float(np.mean(dmed_s)),
             "dmed_h_med": float(np.median(dmed_h)),
             "ch_s_med": float(np.median([c for c in ch_s if c is not None])),
             "ch_h_med": float(np.median([c for c in ch_h if c is not None])),
             "fwhm_ratio_med": float(np.median(fr)),
             "fwhm_ratio_p25": float(np.percentile(fr, 25)),
             "fwhm_ratio_p75": float(np.percentile(fr, 75)),
             "peak_med": float(np.median(np.concatenate(peaks))),
             "xcorr_mean": xcorr,
             "per_object": per}
    print("=== M3b %s ===" % args.split, flush=True)
    print("dmed_s med=%.4f (D8 test 0.0384)  dmed_h med=%.4f" % (
        stats["dmed_s_med"], stats["dmed_h_med"]), flush=True)
    print("ch_s med=%.4f (D8 0.2132)  ch_h med=%.4f" % (
        stats["ch_s_med"], stats["ch_h_med"]), flush=True)
    print("fwhm_ratio med=%.2f [%s-%s] (GT 1.0, D8 2.0, gate <=1.5)  peak=%.3f "
          "(GT 1.0, gate >=0.5)  xcorr=%.3f (collapse if ->1)"
          % (stats["fwhm_ratio_med"], stats["fwhm_ratio_p25"],
             stats["fwhm_ratio_p75"], stats["peak_med"], stats["xcorr_mean"]),
          flush=True)
    json.dump(stats, open(os.path.join(ROOT, args.out, "summary.json"), "w"),
              indent=1)
    print("saved -> %s/summary.json" % args.out, flush=True)


if __name__ == "__main__":
    main()
