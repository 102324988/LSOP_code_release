#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Point-cloud inversion from saved spherical profiles: sample a surface point
at depth_peak along each ray (skip low-coverage rays), compare point cloud vs
normalized GT via Chamfer + F-score. This isolates whether the spherical-profile
information itself is sufficient for surface recovery (bypassing backprojection+MC).
"""
import csv
import json
import os
import sys

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


def gt_norm_cloud(name):
    meta = json.load(open(os.path.join(RENDERS, name, "meta.json")))
    scale = meta["scale"]
    center = np.array(meta["bbox_center"], float)
    m = trimesh.load(os.path.join(MESHES, name + ".glb"), force="mesh")
    m.vertices = (np.asarray(m.vertices, float) - center) * scale
    return trimesh.sample.sample_surface(m, N_SAMP)[0].astype(np.float32)


def surface_points(name, cov_floor, cov_max, pk_floor):
    prof = os.path.join(D4, name, "profiles")
    peak = np.load(os.path.join(prof, "depth_peak.npy")).astype(np.float32)
    cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(prof, "meta.json")))
    rmax = float(meta["rmax"])
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                    axis=-1).reshape(-1, 3).astype(np.float32)
    r = peak * rmax
    keep = (cov >= cov_floor) & (cov <= cov_max) & (r >= pk_floor)
    pts = (r[keep, None] * dirs[keep]).astype(np.float32)
    return pts, int(keep.sum())


def metrics(pc, gt):
    if len(pc) == 0 or len(gt) == 0:
        return None
    kd_pc, kd_gt = cKDTree(pc), cKDTree(gt)
    d1, _ = kd_gt.query(pc)  # pc -> gt
    d2, _ = kd_pc.query(gt)  # gt -> pc
    chamfer = float((d1.mean() + d2.mean()) / 2)
    out = {"chamfer": chamfer}
    for t in (0.02, 0.05, 0.1):
        f1 = float((d1 < t).mean())
        f2 = float((d2 < t).mean())
        out["f@%s" % t] = 2 * f1 * f2 / (f1 + f2 + 1e-12)
        out["fa@%s" % t] = f2  # gt coverage by inverted pts
    return out


def main():
    names = sys.argv[1:]
    cov_floor = float(sys.argv[len(sys.argv) - 1]) if len(sys.argv) > len(names) + 1 else 0.02
    for name in names:
        try:
            pc, n_keep = surface_points(name, cov_floor, 1.5, 0.02)
            gt = gt_norm_cloud(name)
            m = metrics(pc, gt)
            if m is None:
                print(name, "no pts kept")
                continue
            print("%-38s n_pts=%5d kept=%4d chamfer=%.4f f@.02=%.4f f@.05=%.4f "
                  "f@.1=%.4f fa@.05=%.3f"
                  % (name, len(pc), n_keep, m["chamfer"], m["f@0.02"], m["f@0.05"],
                     m["f@0.1"], m["fa@0.05"]))
        except Exception as e:  # noqa: BLE001
            print(name, "ERR", e)


if __name__ == "__main__":
    main()
