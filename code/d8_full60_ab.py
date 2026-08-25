#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D8 vs D6-v3 vs D7c2 honest A/B on the shared D7-60 val objects.

The D7-60 val objects (d7c2_ca/meta.json val_names) are a strict subset of
D8's 90-val (D8 val = D7's 60 + 30 fresh). This script computes, per object,
dmed_s / dmed_h / ch_s / ch_h for the three model artifact sets on exactly
those 60 objects, then reports medians and pairwise win counts, so D8 can be
placed against the D6-v3 / D7c2 numbers in the D7 report WITHOUT shared-12
sampling bias.

Usage: python d8_full60_ab.py
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
D5 = os.path.join(ROOT, "output", "gso_d5")
N_PHI, N_THETA, N_BINS = 128, 64, 96
DIRS = None


def dirs_grid():
    global DIRS
    if DIRS is None:
        th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
        ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
        PH, TH = np.meshgrid(ph, th, indexing="ij")
        DIRS = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                         np.cos(TH)], axis=-1).reshape(-1, 3).astype(np.float32)
    return DIRS


def score(depth_bin, rmax, cov, floor, gt):
    dirs = dirs_grid()
    r = (depth_bin / (N_BINS - 1)) * rmax
    keep = (cov >= floor) & (r >= 0.02) & (r < rmax)
    pc = (r[keep, None] * dirs[keep]).astype(np.float32)
    if len(pc) == 0:
        return None
    kd_pc, kd_gt = cKDTree(pc), cKDTree(gt)
    d1, _ = kd_gt.query(pc)
    d2, _ = kd_pc.query(gt)
    return float((d1.mean() + d2.mean()) / 2)


def metrics(pred_dir, n):
    soft = np.load(os.path.join(pred_dir, n + "_soft.npy")).astype(np.float32)
    hard = np.load(os.path.join(pred_dir, n + ".npy")).astype(np.float32)
    cov = np.load(os.path.join(pred_dir, n + "_cov.npy")).astype(np.float32)
    pd = os.path.join(D4, n, "profiles")
    peak = np.load(os.path.join(pd, "depth_peak.npy")).astype(np.float32)
    rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
    gt = np.load(os.path.join(D5, n, "surf_points.npy")).astype(np.float32)
    m = cov >= 0.02
    gtn = peak[m] / rmax
    dmed_s = float(np.median(np.abs(soft[m] / (N_BINS - 1) - gtn)))
    dmed_h = float(np.median(np.abs(hard[m] / (N_BINS - 1) - gtn)))
    ch_s = score(soft, rmax, cov, 0.02, gt)
    ch_h = score(hard, rmax, cov, 0.02, gt)
    return dmed_s, dmed_h, ch_s, ch_h


def main():
    names = json.load(open(os.path.join(ROOT, "d7c2_ca", "meta.json")))["val_names"]
    models = {"D6v3": "d6_pred_v3_full", "D7c2": "d7c2_pred",
              "D8": "d8_mean_pool_val"}
    data = {k: {n: metrics(v, n) for n in names} for k, v in models.items()}
    # shared-12 from the D7 report (report used abbreviated names; resolve to
    # the actual GSO names inside names60 by unique prefix)
    shared12 = ["Asus_Sabertooth", "Epson_DURABrite", "Hasbro_Cranium",
                "My_Little_Pony", "Nordic_Ware", "Olive_Kids_Game_On_Munch",
                "Perricone_MD_AcylGlutathione_Eye_Lid", "Perricone_MD_Hypo",
                "Perricone_MD_Neuropeptide", "RESCUE_CREW", "Top_Paw_Dog_Bowl",
                "Toysmith"]
    def resolve(pre):
        hits = [n for n in names if n.startswith(pre)]
        assert len(hits) == 1, (pre, hits)
        return hits[0]
    shared12 = [resolve(p) for p in shared12]
    print("resolved shared-12:", shared12)
    print("=== SHARED-12 (dmed_s / ch_s) ===")
    for n in shared12:
        print("%-24s D6v3=%8.4f/%8.4f  D7c2=%8.4f/%8.4f  D8=%8.4f/%8.4f"
              % (n, data["D6v3"][n][0], data["D6v3"][n][2],
                 data["D7c2"][n][0], data["D7c2"][n][2],
                 data["D8"][n][0], data["D8"][n][2]))
    for k in models:
        v = np.array([data[k][n][0] for n in shared12])
        c = np.array([x for n in shared12 for x in [data[k][n][2]] if x is not None])
        print("  %s shared12: dmed_s med=%.4f ch_s med=%.4f" % (k, np.median(v), np.median(c)))

    print("\n=== FULL-60 (median + pairwise) ===")
    med = {}
    for k in models:
        ds = np.array([data[k][n][0] for n in names])
        ch = np.array([data[k][n][2] for n in names if data[k][n][2] is not None])
        dh = np.array([data[k][n][1] for n in names])
        none_ch = [n for n in names if data[k][n][2] is None]
        med[k] = (float(np.median(ds)), float(np.median(ch)) if len(ch) else float("nan"))
        print("  %s 60-val: dmed_s=%.4f  ch_s=%.4f  (dmed_h med=%.4f)  [ch None: %d -> %s]"
              % (k, med[k][0], med[k][1], float(np.median(dh)), len(none_ch), none_ch[:3]))
    pairs = [("D6v3", "D8"), ("D7c2", "D8"), ("D6v3", "D7c2")]
    for a, b in pairs:
        wins = {"dmed": [0, 0], "ch": [0, 0]}
        ties = {"dmed": 0, "ch": 0}
        for n in names:
            da, db = data[a][n][0], data[b][n][0]
            if da < db:
                wins["dmed"][0] += 1
            elif db < da:
                wins["dmed"][1] += 1
            else:
                ties["dmed"] += 1
            ca, cb = data[a][n][2], data[b][n][2]
            if ca is not None and cb is not None:
                if ca < cb:
                    wins["ch"][0] += 1
                elif cb < ca:
                    wins["ch"][1] += 1
                else:
                    ties["ch"] += 1
        print("  %s vs %s : dmed_s %d/%d (tie %d), ch_s %d/%d (tie %d)"
              % (a, b, wins["dmed"][0], wins["dmed"][1], ties["dmed"],
                 wins["ch"][0], wins["ch"][1], ties["ch"]))
    # hard-cloud robustness (no hard cloud -> empty point set)
    n_empty = 0
    for n in names:
        hard = np.load(os.path.join(models["D8"], n + ".npy")).astype(np.float32)
        cov = np.load(os.path.join(models["D8"], n + "_cov.npy")).astype(np.float32)
        pd = os.path.join(D4, n, "profiles")
        rmax = float(json.load(open(os.path.join(pd, "meta.json")))["rmax"])
        r = (hard / (N_BINS - 1)) * rmax
        keep = (cov >= 0.02) & (r >= 0.02) & (r < rmax)
        if not keep.any():
            n_empty += 1
    print("  D8 hard-empty objects: %d/60" % n_empty)


if __name__ == "__main__":
    main()
