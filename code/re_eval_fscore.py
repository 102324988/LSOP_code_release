#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-evaluate inversion F-score across thresholds to test the 0.02 protocol
hypothesis (0.02 < half-voxel of the 64^3 grid -> floor effect).
For every object with mesh_inverted.ply, recompute bidirectional dists once and
report F-score at 0.02/0.03/0.04/0.05/0.08/0.10 plus f_a (GT-surface coverage).
Output: d4_fscore_grid.csv  (one row per object)
"""
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
THRES = (0.02, 0.03, 0.04, 0.05, 0.08, 0.10)
N = 50000


def evaluate_one(name):
    prof = os.path.join(D4, name, "profiles")
    mi_path = os.path.join(prof, "mesh_inverted.ply")
    gt_path = os.path.join("/root", "gso", "meshes", name + ".glb")
    if not os.path.exists(mi_path) or not os.path.exists(gt_path):
        return None
    try:
        mi = trimesh.load(mi_path, force="mesh")
        gt = trimesh.load(gt_path, force="mesh")
        if len(mi.faces) < 8 or len(gt.faces) < 8:
            return None
        pa = trimesh.sample.sample_surface(gt, N)[0]
        pb = trimesh.sample.sample_surface(mi, N)[0]
        kda, kdb = cKDTree(pa), cKDTree(pb)
        dab, _ = kdb.query(pa)   # GT point -> inverted mesh
        dba, _ = kda.query(pb)   # inverted -> GT
        row = {"name": name, "chamfer": float((dab.mean() + dba.mean()) / 2)}
        for t in THRES:
            fa = float((dab < t).mean())
            fb = float((dba < t).mean())
            fs = 2 * fa * fb / (fa + fb + 1e-12)
            row["f@%s" % t] = fs
            row["fa@%s" % t] = fa
            row["fb@%s" % t] = fb
        return row
    except Exception as e:  # noqa: BLE001
        return {"name": name, "error": str(e)}


def main():
    names = sorted(os.listdir(D4))
    if len(sys.argv) > 1:
        names = names[: int(sys.argv[1])]
    results = []
    with Pool(min(8, os.cpu_count() or 4)) as p:
        for r in p.imap_unordered(evaluate_one, names, chunksize=4):
            if r is not None:
                results.append(r)
    results.sort(key=lambda r: r["name"])
    out = os.path.join(ROOT, "d4_fscore_grid.csv")
    with open(out, "w") as f:
        cols = ["name", "chamfer"] + ["f@%s" % t for t in THRES]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    if results:
        import collections
        med = collections.OrderedDict()
        for t in THRES:
            vals = [r["f@%s" % t] for r in results if "f@%s" % t in r]
            med["med_f@%s" % t] = float(np.median(vals))
            med["mean_f@%s" % t] = float(np.mean(vals))
            med["q1_f@%s" % t] = float(np.percentile(vals, 25))
            med["q3_f@%s" % t] = float(np.percentile(vals, 75))
            med["n>0_f@%s" % t] = int(np.sum(np.array(vals) > 0))
        med["n"] = len(results)
        print(json.dumps(med, indent=1))
    print("wrote", out, "rows=", len(results))


if __name__ == "__main__":
    main()
