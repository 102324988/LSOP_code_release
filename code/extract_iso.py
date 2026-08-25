"""E0: extract the 0.5 iso-surface from the min-over-views opacity grid
(marching cubes) and compare against the GT mesh.

Question: does iso-surface extraction locate the outer surface better than
per-ray P-first-peak inversion (which on sphere lands at r~0.72 vs GT 0.95)?
Is the shell information in the field at all, or is the density smeared inward?
"""
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

HERE = os.path.expanduser("~/e0lab/e0")
OBJS = ["torus", "vase", "rocky", "mug", "sphere", "bumpy"]
N_SAMP = 30000
OUT = os.path.join(HERE, "isosurf")
os.makedirs(OUT, exist_ok=True)

for obj in OBJS:
    d = os.path.join(HERE, "grids", obj)
    grid = np.load(os.path.join(d, "grid.npy"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    bbox = np.array(meta["bbox"], dtype=np.float64)
    res = grid.shape[0]

    verts, faces, _, _ = marching_cubes(grid, level=0.5)
    w = bbox[0::2] + verts / (res - 1) * (bbox[1::2] - bbox[0::2])
    m = (np.abs(w) < 1.0).all(axis=1)      # keep central bbox (-1,1)^3
    w = w[m]
    print(f"[iso] {obj}: raw {verts.shape[0]} verts, kept {w.shape[0]}", flush=True)

    gt = trimesh.load(os.path.join(HERE, "data", "meshes", f"{obj}.ply"), force="mesh")
    if len(w):
        _, d_iso2gt, _ = gt.nearest.on_surface(w)
    else:
        d_iso2gt = np.array([])
    samp = gt.sample(N_SAMP)
    d_gt2iso = cKDTree(w).query(samp)[0] if len(w) else np.full(N_SAMP, np.inf)

    resm = dict(
        obj=obj, level=0.5, n_iso=int(len(w)),
        iso2gt_p50=float(np.median(d_iso2gt)) if len(w) else None,
        iso2gt_mean=float(np.mean(d_iso2gt)) if len(w) else None,
        gt2iso_mean=float(np.mean(d_gt2iso)),
        cov_01=float(np.mean(d_gt2iso < 0.1)),
        cov_02=float(np.mean(d_gt2iso < 0.2)),
        chamfer=float((np.mean(d_iso2gt) + np.mean(d_gt2iso)) / 2) if len(w) else None,
    )
    if obj == "sphere" and len(w):
        r = np.linalg.norm(w, axis=1)
        resm["radius_p10"] = float(np.percentile(r, 10))
        resm["radius_p50"] = float(np.percentile(r, 50))
        resm["radius_p90"] = float(np.percentile(r, 90))
    with open(os.path.join(OUT, f"{obj}.res.json"), "w") as f:
        json.dump(resm, f, indent=1)
    print(json.dumps(resm, ensure_ascii=False), flush=True)

print("ISO_ALL_DONE")
