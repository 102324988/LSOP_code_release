"""E0: end-to-end point-cloud evaluation of the P-weighted first-peak inversion.

Collects the 3042-ray first-peak hits (P(r)=T·o) into one point cloud per
object, then reports:
  - d_gt2cloud: GT surface sample -> nearest hit-cloud point  (coverage)
  - d_cloud2gt: hit-cloud point -> GT surface                  (accuracy)
  - coverage@t: fraction of GT surface within t of the cloud
  - chamfer    : symmetric mean of the two directions
Also dumps the hit cloud (world coords + per-point GT distance) to a .npy so
figures can be drawn locally.
"""
import argparse
import os
import sys

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from clean_ray_profile import ray_march  # noqa: E402

ap = argparse.ArgumentParser()
lp = ModelParams(ap)
pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
ap.add_argument("--obj", required=True)
ap.add_argument("--dumps", default="clouds", help="dir to write <obj>.npy hit clouds")
args = ap.parse_args()
ds = lp.extract(args)
pipe = pp.extract(args)

obj = args.obj
mesh = trimesh.load(os.path.join(ds.source_path, "..", "meshes", obj + ".ply"),
                    force="mesh")
mesh_v = mesh.vertices.astype(np.float64)
mesh_tri = trimesh.Trimesh(vertices=mesh_v, faces=mesh.faces)
tree = mesh_tri.triangles_tree

g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)

els = [20, 45, 70]
azs = np.linspace(0, 360, 6, endpoint=False)
Rcam = 3.2
tgt_r = np.linspace(0.0, 0.78, 13)
tgt_z = np.linspace(-0.30, 0.30, 13)
pts = []
for el_deg in els:
    el = np.deg2rad(el_deg)
    for az_deg in azs:
        az = np.deg2rad(az_deg)
        cam = np.array([Rcam*np.cos(el)*np.cos(az),
                        Rcam*np.cos(el)*np.sin(az), Rcam*np.sin(el)])
        ct, stt = np.cos(az), np.sin(az)
        for tr in tgt_r:
            for tz in tgt_z:
                tgtr = np.array([tr*ct, tr*stt, tz])
                dvec = tgtr - cam
                t, o, Tt, P = ray_march(torch.tensor(cam, dtype=torch.float32, device="cuda"),
                                        torch.tensor(dvec, dtype=torch.float32, device="cuda"),
                                        g, 0.02, 7.0, K=1200)
                thr = max(0.01, 0.05*P.max())
                for i in range(1, len(P)-1):
                    if P[i] > P[i-1] and P[i] >= P[i+1] and P[i] > thr:
                        pt = cam + (dvec/np.linalg.norm(dvec))*t[i]
                        if np.abs(pt).max() < 1.2:
                            pts.append(pt)
                        break  # first peak only
pts = np.array(pts)
print(f"[cloud] {obj}: hits={len(pts)}")

# accuracy: cloud -> GT surface (trimesh batch nearest, as in eval_matrix2)
_, d_cloud2gt, _ = mesh_tri.nearest.on_surface(pts.astype(np.float64)) if len(pts) \
    else (None, np.array([]), None)
# coverage: GT surface samples -> cloud
n_samp = 30000
samp = mesh_tri.sample(n_samp)
tree_c = cKDTree(pts.astype(np.float64))
d_gt2cloud = tree_c.query(samp)[0] if len(pts) else np.full(n_samp, np.inf)

res = dict(
    obj=obj, hits=int(len(pts)),
    cloud2gt_p50=float(np.median(d_cloud2gt)) if len(pts) else None,
    cloud2gt_mean=float(np.mean(d_cloud2gt)) if len(pts) else None,
    gt2cloud_mean=float(np.mean(d_gt2cloud)),
    gt2cloud_p50=float(np.median(d_gt2cloud)),
    coverage_01=float(np.mean(d_gt2cloud < 0.1)),
    coverage_02=float(np.mean(d_gt2cloud < 0.2)),
    chamfer_mean=float((np.mean(d_cloud2gt) + np.mean(d_gt2cloud)) / 2) if len(pts) else None,
)
os.makedirs(args.dumps, exist_ok=True)
np.save(os.path.join(args.dumps, f"{obj}.npy"),
        np.hstack([pts, d_cloud2gt[:, None]]))  # Nx4: xyz + gt-dist
import json
with open(os.path.join(args.dumps, f"{obj}.res.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"[cloud] wrote {args.dumps}/{obj}.npy and {obj}.res.json")
print(str(res).replace("'", '"'))
