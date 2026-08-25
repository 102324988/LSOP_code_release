"""E0: ray-nearest-gaussian surface inversion — uniform metric for both 3DGS
and PGSR gaussians.

P-first-peak ray-march (pixel_ray_march.py) assumes gaussians with a finite
opacity-field width (~scale 0.1): it samples the ray and accumulates local
occupancy, so P(r) peaks where the ray first crosses the surface. PGSR trains
its gaussians into near-zero-size points (scale ~1e-6) lying exactly on the
surface shell, so ray-march sampling essentially never lands on one and the
P-peak test finds nothing. That is not a PGSR failure — the surface is encoded
in the gaussian positions themselves.

This script inverts with a scale-agnostic "first visible gaussian" rule that
reduces to first-surface for both cases: for each pixel ray, collect candidate
gaussians near the ray (reference points along the ray, cKDTree kNN union),
project each candidate onto the ray (t = (p-o)·d), keep those with t>0 and
point-line distance below a fixed tolerance, take the smallest t among gaussians
with non-trivial opacity — that is the first visible surface point.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
import plyfile
from scipy.spatial import cKDTree

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from clean_ray_profile import quat2mat  # noqa: E402

ap = argparse.ArgumentParser()
lp = ModelParams(ap)
pp = PipelineParams(ap)
ap.add_argument("--ply", required=True)
ap.add_argument("--obj", required=True)
ap.add_argument("--dumps", default="ray_nearest")
ap.add_argument("--step", type=int, default=12)
ap.add_argument("--k", type=int, default=8, help="kNN per reference point")
ap.add_argument("--r_ref", type=float, default=0.25, help="kNN radius per ref point")
ap.add_argument("--t_tol", type=float, default=0.15, help="point-line dist tolerance")
ap.add_argument("--alpha_min", type=float, default=0.05)
ap.add_argument("--t_near", type=float, default=0.02)
ap.add_argument("--t_far", type=float, default=6.0)
ap.add_argument("--n_ref", type=int, default=40)
args = ap.parse_args()
ds = lp.extract(args)
pipe = pp.extract(args)

g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=None, shuffle=False)  # cameras only

v = plyfile.PlyData.read(args.ply)["vertex"]
xyz_n = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
alpha_n = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).ravel()
scl_n = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
print(f"[ray-nearest] {len(xyz_n)} gaussians, alpha>={args.alpha_min}: "
      f"{np.mean(alpha_n >= args.alpha_min):.3f}", flush=True)

kd = cKDTree(xyz_n)
xyz_t = torch.from_numpy(xyz_n.astype(np.float32)).cuda()
alpha_t = torch.from_numpy(alpha_n.astype(np.float32)).cuda()

mesh = trimesh.load(os.path.join(ds.source_path, "..", "meshes", args.obj + ".ply"),
                    force="mesh")
mesh_tri = trimesh.Trimesh(vertices=mesh.vertices.astype(np.float64), faces=mesh.faces)

t_ref = np.linspace(args.t_near, args.t_far, args.n_ref)

hits = []
for vi, cam in enumerate(scene.getTrainCameras()):
    W, H = cam.image_width, cam.image_height
    fx, fy = cam.focal_x, cam.focal_y
    cx, cy = W / 2.0, H / 2.0
    u = torch.arange(0, W, args.step, device="cuda").float() + args.step / 2
    vv = torch.arange(0, H, args.step, device="cuda").float() + args.step / 2
    V, U = torch.meshgrid(vv, u, indexing="ij")
    X = (U - cx) / fx
    Y = (V - cy) / fy
    d_cam = torch.stack([X, Y, torch.ones_like(X)], dim=-1)
    Rc = torch.tensor(cam.R, dtype=torch.float32, device="cuda")
    d_w = d_cam @ Rc.T
    d_w = d_w / d_w.norm(dim=-1, keepdim=True)
    d_w = d_w.reshape(-1, 3)                                     # (M,3)
    origin = torch.tensor(cam.camera_center, dtype=torch.float32, device="cuda")
    o_np = origin.cpu().numpy()
    M = d_w.shape[0]

    # candidate gaussians: union of kNN over reference points along the ray
    refs = o_np[None, :] + d_w.cpu().numpy()[:, None, :] * t_ref[None, :, None]
    refs = refs.reshape(-1, 3)                                   # (M*n_ref,3)
    dist, idx = kd.query(refs, k=args.k, workers=-1,
                         distance_upper_bound=args.r_ref)
    idx = idx.reshape(M, args.n_ref, args.k)
    cand = np.unique(idx.reshape(M, -1))                         # (Nc,) per view? -- unique per ray better
    # unique is too slow per ray over M; do per-ray via gather on GPU instead
    idx_t = torch.from_numpy(idx.astype(np.int64)).cuda()        # (M,n_ref,k)
    gi = idx_t.reshape(M, -1)                                    # (M,n_ref*k)
    # mask invalid (kNN returned index == N when beyond radius)
    valid = (gi < xyz_t.shape[0])
    gi_safe = gi.clamp(max=xyz_t.shape[0] - 1)
    p = xyz_t[gi_safe]                                           # (M,n_ref*k,3)
    tvec = (p - origin[None, None, :]) @ d_w[:, :, None]         # (M,n_ref*k,1)
    tvec = tvec[..., 0]                                          # (M,n_ref*k)
    pdist2 = ((p - origin[None, None, :]) ** 2).sum(-1) - tvec ** 2
    on_ray = (tvec > args.t_near) & (pdist2 < args.t_tol ** 2) & valid
    al = alpha_t[gi_safe]
    ok = on_ray & (al >= args.alpha_min)
    # first hit along ray: min t among ok
    t_sel = torch.where(ok, tvec, torch.full_like(tvec, float("inf")))
    t_min = t_sel.min(dim=1).values                            # (M,)
    has = torch.isfinite(t_min)
    hv = origin[None, :] + d_w[has] * t_min[has][:, None]
    mm = hv.norm(dim=1) < 1.2
    hv = hv[mm]
    if len(hv):
        hits.append(hv.cpu().numpy())
    if (vi + 1) % 8 == 0:
        print(f"  [ray-nearest] view {vi+1}/{len(scene.getTrainCameras())} done, "
              f"hits so far {sum(len(h) for h in hits)}", flush=True)

pts = np.vstack(hits) if hits else np.zeros((0, 3))
print(f"[cloud] {args.obj}: total hits={len(pts)}")

_, d_cloud2gt, _ = mesh_tri.nearest.on_surface(pts.astype(np.float64)) if len(pts) \
    else (None, np.array([]), None)
n_samp = 30000
samp = mesh_tri.sample(n_samp)
d_gt2cloud = cKDTree(pts.astype(np.float64)).query(samp)[0] if len(pts) else np.full(n_samp, np.inf)
r = np.linalg.norm(pts, axis=1)
resm = dict(
    obj=args.obj, step=args.step, hits=int(len(pts)),
    cloud2gt_p50=float(np.median(d_cloud2gt)) if len(pts) else None,
    cloud2gt_mean=float(np.mean(d_cloud2gt)) if len(pts) else None,
    gt2cloud_mean=float(np.mean(d_gt2cloud)),
    coverage_01=float(np.mean(d_gt2cloud < 0.1)),
    coverage_02=float(np.mean(d_gt2cloud < 0.2)),
    chamfer_mean=float((np.mean(d_cloud2gt) + np.mean(d_gt2cloud)) / 2) if len(pts) else None,
    hit_radius_p10=float(np.percentile(r, 10)) if len(pts) else None,
    hit_radius_p50=float(np.percentile(r, 50)) if len(pts) else None,
    hit_radius_p90=float(np.percentile(r, 90)) if len(pts) else None,
    hit_frac_r_lt_07=float(np.mean(r < 0.7)) if len(pts) else None,
)
os.makedirs(args.dumps, exist_ok=True)
np.save(os.path.join(args.dumps, f"{args.obj}.npy"),
        np.hstack([pts, d_cloud2gt[:, None]]) if len(pts) else pts)
with open(os.path.join(args.dumps, f"{args.obj}.res.json"), "w") as f:
    json.dump(resm, f, indent=1)
print(json.dumps(resm, ensure_ascii=False))
