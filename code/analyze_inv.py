#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose inversion failure: does the inverted mesh radius systematically
shrink vs GT (ray origin at bbox center -> depth_peak biased inward)?

For representative objects, load GT (GSO frame) -> normalize with D2 meta.json
(scale, bbox_center), load mesh_inverted.ply (already normalized), compare
vertex-radius distributions (radius = dist to origin after bbox-center->0).
"""
import csv
import json
import os
import sys

import numpy as np
import trimesh

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
MESHES = "/root/gso/meshes"
RENDERS = "/root/gso/renders"


def norm_gt(name):
    meta = json.load(open(os.path.join(RENDERS, name, "meta.json")))
    scale = meta["scale"]
    center = np.array(meta["bbox_center"], float)
    m = trimesh.load(os.path.join(MESHES, name + ".glb"), force="mesh")
    v = np.asarray(m.vertices, float) - center
    v = v * scale
    return v


def mesh_radii(verts):
    r = np.linalg.norm(verts, axis=1)
    return r


def pick_representatives(n_best=2, n_mid=2, n_worst=2):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "d4_fscore_grid.csv"))))
    valid = [r for r in rows if r.get("f@0.02", "") != ""]
    valid.sort(key=lambda r: float(r["f@0.02"]))
    picks = []
    for r in valid[-n_best:][::-1]:
        picks.append(("best", r["name"], float(r["f@0.02"])))
    mid = valid[len(valid) // 2]
    picks.append(("mid", mid["name"], float(mid["f@0.02"])))
    for r in valid[:n_worst]:
        picks.append(("worst", r["name"], float(r["f@0.02"])))
    return picks


def main():
    picks = pick_representatives()
    print("tag,name,f@0.02,n_gt_verts,n_inv_verts,gt_r_med,inv_r_med,inv/gt_r_med,gt_r_p90,inv_r_p90")
    for tag, name, fs in picks:
        prof = os.path.join(D4, name, "profiles")
        inv_p = os.path.join(prof, "mesh_inverted.ply")
        try:
            gt_v = norm_gt(name)
            gt_r = mesh_radii(gt_v)
            row = [tag, name, "%.3f" % fs, str(len(gt_v))]
            if os.path.exists(inv_p):
                inv = trimesh.load(inv_p, force="mesh")
                inv_r = mesh_radii(np.asarray(inv.vertices, float))
                row += [str(len(inv_r)),
                        "%.3f" % np.median(gt_r), "%.3f" % np.median(inv_r),
                        "%.2f" % (np.median(inv_r) / np.median(gt_r)),
                        "%.3f" % np.percentile(gt_r, 90), "%.3f" % np.percentile(inv_r, 90)]
            else:
                row += ["-", "-", "-", "-", "-", "-"]
            print(",".join(row))
        except Exception as e:  # noqa: BLE001
            print(tag, name, "ERR", e)


if __name__ == "__main__":
    main()
