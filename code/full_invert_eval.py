#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full-batch point-cloud inversion from saved spherical profiles (bypassing
backprojection+MC). Surface point at depth_peak along each ray, skip low/high
coverage rays, compare vs normalized GT. Writes d4_point_inv.csv + stats.
Usage: python full_invert_eval.py [cov_floor] [cov_max] [n_proc]
"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import trimesh
import trimesh.sample
from scipy.spatial import cKDTree

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
MESHES = "/root/gso/meshes"
RENDERS = "/root/gso/renders"
N_THETA, N_PHI = 64, 128
N_SAMP = 50000

_dirs = None


def get_dirs():
    global _dirs
    if _dirs is None:
        th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
        ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
        PH, TH = np.meshgrid(ph, th, indexing="ij")
        _dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                         axis=-1).reshape(-1, 3).astype(np.float32)
    return _dirs


def evaluate(name, cov_floor, cov_max):
    prof = os.path.join(D4, name, "profiles")
    pk_p, cv_p = os.path.join(prof, "depth_peak.npy"), os.path.join(prof, "coverage.npy")
    me_p = os.path.join(prof, "meta.json")
    gt_p = os.path.join(MESHES, name + ".glb")
    rd_p = os.path.join(RENDERS, name, "meta.json")
    if not all(os.path.exists(p) for p in (pk_p, cv_p, me_p, gt_p, rd_p)):
        return None
    peak = np.load(pk_p).astype(np.float32)
    cov = np.load(cv_p).astype(np.float32)
    rmax = float(json.load(open(me_p))["rmax"])
    dirs = get_dirs()
    r = peak * rmax
    keep = (cov >= cov_floor) & (cov <= cov_max) & (r >= 0.02)
    pc = (r[keep, None] * dirs[keep]).astype(np.float32)
    if len(pc) < 8:
        return None
    meta = json.load(open(rd_p))
    scale = meta["scale"]
    center = np.array(meta["bbox_center"], float)
    m = trimesh.load(gt_p, force="mesh")
    m.vertices = (np.asarray(m.vertices, float) - center) * scale
    gt = trimesh.sample.sample_surface(m, N_SAMP)[0].astype(np.float32)
    kd_pc, kd_gt = cKDTree(pc), cKDTree(gt)
    d1, _ = kd_gt.query(pc)
    d2, _ = kd_pc.query(gt)
    row = {"name": name, "n_pts": len(pc),
           "chamfer": float((d1.mean() + d2.mean()) / 2)}
    for t in (0.02, 0.05, 0.1):
        f1 = float((d1 < t).mean())
        f2 = float((d2 < t).mean())
        row["f@%s" % t] = 2 * f1 * f2 / (f1 + f2 + 1e-12)
        row["fa@%s" % t] = f2
    return row


def main():
    cov_floor = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    cov_max = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    n_proc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    names = sorted(os.listdir(D4))
    rows = []
    from functools import partial
    worker = partial(evaluate, cov_floor=cov_floor, cov_max=cov_max)
    with Pool(n_proc) as p:
        for r in p.imap_unordered(worker, names, chunksize=8):
            if r is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["name"])
    tag = "f%.2f_c%.2f" % (cov_floor, cov_max)
    out = os.path.join(ROOT, "d4_point_inv_%s.csv" % tag)
    cols = ["name", "n_pts", "chamfer", "f@0.02", "f@0.05", "f@0.1",
            "fa@0.02", "fa@0.05", "fa@0.1"]
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    stat = {}
    for t in (0.02, 0.05, 0.1):
        v = np.array([r["f@%s" % t] for r in rows], float)
        stat["f@%s" % t] = [float(np.percentile(v, q)) for q in (0, 25, 50, 75, 100)]
        stat["n>0@%s" % t] = int((v > 0).sum())
        stat["mean@%s" % t] = float(v.mean())
    stat["chamfer_med"] = float(np.median([r["chamfer"] for r in rows]))
    stat["chamfer_q1"] = float(np.percentile([r["chamfer"] for r in rows], 25))
    stat["chamfer_q3"] = float(np.percentile([r["chamfer"] for r in rows], 75))
    stat["n"] = len(rows)
    print(json.dumps(stat, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
