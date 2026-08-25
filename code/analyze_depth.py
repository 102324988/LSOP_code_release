#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quantify depth-peak localization vs GT surface along the same rays.
For each equirectangular direction (same 64x128 as D4), ray-cast the NORMALIZED
GT mesh from origin; first hit = GT surface radius. Compare to depth_peak.
Outputs median abs error, bias, and how depth_peak relates to GT surface.
"""
import json
import os
import sys

import numpy as np
import trimesh

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
MESHES = "/root/gso/meshes"
RENDERS = "/root/gso/renders"

N_THETA, N_PHI = 64, 128


def gt_mesh_norm(name):
    meta = json.load(open(os.path.join(RENDERS, name, "meta.json")))
    scale = meta["scale"]
    center = np.array(meta["bbox_center"], float)
    m = trimesh.load(os.path.join(MESHES, name + ".glb"), force="mesh")
    m.vertices = (np.asarray(m.vertices, float) - center) * scale
    return m


def dirs_grid():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                    axis=-1).reshape(-1, 3).astype(np.float32)


def gt_hits(m, dirs, rmax):
    """First intersection distance of each ray from origin along dir."""
    tri = m.triangles
    out = np.full(len(dirs), np.nan, dtype=np.float32)
    # trimesh ray-mesh intersection per ray (origin=0)
    origins = np.zeros_like(dirs)
    loc, idx_ray, _ = m.ray.intersects_location(origins, dirs)
    if len(loc):
        # distance per hit; keep first (smallest) per ray
        d = np.linalg.norm(loc, axis=1)
        for ray_i in np.unique(idx_ray):
            mask = idx_ray == ray_i
            out[ray_i] = d[mask].min()
    return out


def main():
    names = sys.argv[1:]
    for name in names:
        prof = os.path.join(D4, name, "profiles")
        peak = np.load(os.path.join(prof, "depth_peak.npy"))
        cov = np.load(os.path.join(prof, "coverage.npy"))
        meta = json.load(open(os.path.join(prof, "meta.json")))
        rmax = meta["rmax"]
        dp = peak * rmax
        try:
            m = gt_mesh_norm(name)
        except Exception as e:  # noqa: BLE001
            print(name, "gt load err", e)
            continue
        d = dirs_grid()
        gh = gt_hits(m, d, rmax)
        valid = ~np.isnan(gh)
        n_valid = int(valid.sum())
        if n_valid == 0:
            print(name, "no ray hits")
            continue
        err = dp[valid] - gh[valid]
        med_err = float(np.median(err))
        med_abs = float(np.median(np.abs(err)))
        bias = float(np.nanmean(err))
        # how often depth_peak < GT surface (inward bias) and by how much
        inward = float((dp[valid] < gh[valid]).mean())
        rel = float(np.median(np.abs(err) / gh[valid]))
        print("%-38s n_hits=%4d med_err=%+.3f med_abs=%.3f bias=%+.3f inward=%.2f "
              "rel_med=%.3f rmax=%.2f"
              % (name, n_valid, med_err, med_abs, bias, inward, rel, rmax))


if __name__ == "__main__":
    main()
