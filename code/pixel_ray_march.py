"""E0: pixel-level full-surface inversion via pruned ray-march.

Semantics: 3DGS local occupancy along a camera ray (per-gaussian alpha =
sigmoid(opacity)*exp(-0.5*maha), exactly clean_ray_profile.ray_march, the
verified-correct form). For every subsampled pixel of every view, march the
ray through the *nearby* gaussians only (cKDTree k-NN pruning, since distant
gaussians contribute ~0), accumulate T, take the FIRST peak of P = T*alpha,
backproject to a surface hit. This is Path A's real application form
(image -> per-pixel surface -> point cloud) with full surface coverage.

Metrics identical to eval_cloud.py: cloud->GT accuracy, GT->cloud coverage,
symmetric Chamfer. Dumps the cloud as <obj>.npy.
"""
import argparse
import json
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
from clean_ray_profile import quat2mat  # noqa: E402

ap = argparse.ArgumentParser()
lp = ModelParams(ap)
pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
ap.add_argument("--obj", required=True)
ap.add_argument("--dumps", default="pixel_clouds")
ap.add_argument("--step", type=int, default=12, help="pixel subsample step")
ap.add_argument("--K", type=int, default=256)
ap.add_argument("--knn", type=int, default=12, help="nearby gaussians per sample")
ap.add_argument("--nn_radius", type=float, default=0.8, help="cKDTree cutoff")
ap.add_argument("--t_near", type=float, default=0.02)
ap.add_argument("--t_far", type=float, default=6.0)
ap.add_argument("--chunk", type=int, default=300000)
ap.add_argument("--verify_gt", action="store_true",
                help="cross-check inversion depth vs GT mesh first-hit")
args = ap.parse_args()
ds = lp.extract(args)
pipe = pp.extract(args)

g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)

# ---- per-gaussian geometry, moved to GPU once ----
xyz_n = g._xyz.detach().cpu().numpy().astype(np.float64)          # (N,3)
alpha_n = g.get_opacity.detach().cpu().numpy().ravel()            # (N,)
scl_n = g.get_scaling.detach().cpu().numpy()                      # (N,3)
R_n = quat2mat(g.get_rotation).detach().cpu().numpy()               # (N,3,3)
kd = cKDTree(xyz_n)
xyz_t = torch.from_numpy(xyz_n.astype(np.float32)).cuda()
alpha_t = torch.from_numpy(alpha_n.astype(np.float32)).cuda()
scl_t = torch.from_numpy(scl_n.astype(np.float32)).cuda()
R_t = torch.from_numpy(R_n.astype(np.float32)).cuda()

mesh = trimesh.load(os.path.join(ds.source_path, "..", "meshes", args.obj + ".ply"),
                    force="mesh")
mesh_tri = trimesh.Trimesh(vertices=mesh.vertices.astype(np.float64), faces=mesh.faces)

t = torch.linspace(args.t_near, args.t_far, args.K, device="cuda")

hits_world = []
gt_depth_diffs = []
for vi, cam in enumerate(scene.getTrainCameras()):
    W, H = cam.image_width, cam.image_height
    fx, fy = cam.focal_x, cam.focal_y
    cx, cy = W / 2.0, H / 2.0
    u = torch.arange(0, W, args.step, device="cuda").float() + args.step / 2
    v = torch.arange(0, H, args.step, device="cuda").float() + args.step / 2
    V, U = torch.meshgrid(v, u, indexing="ij")
    X = (U - cx) / fx
    Y = (V - cy) / fy
    d_cam = torch.stack([X, Y, torch.ones_like(X)], dim=-1)
    Rc = torch.tensor(cam.R, dtype=torch.float32, device="cuda")  # world->cam
    d_w = d_cam @ Rc.T
    d_w = d_w / d_w.norm(dim=-1, keepdim=True)
    d_w = d_w.reshape(-1, 3)                                     # (M,3)
    origin = torch.tensor(cam.camera_center, dtype=torch.float32, device="cuda")
    M = d_w.shape[0]

    S = origin[None, :] + d_w[:, None, :] * t[None, :, None]     # (M,K,3)
    Sf = S.reshape(-1, 3).cpu().numpy().astype(np.float64)       # (M*K,3)
    dist, idx = kd.query(Sf, k=args.knn, workers=-1,
                         distance_upper_bound=args.nn_radius)
    idx_t = torch.from_numpy(idx.astype(np.int64)).cuda()
    valid = torch.from_numpy(np.isfinite(dist)).cuda()           # (M*K,knn)
    Sg = torch.from_numpy(Sf.astype(np.float32)).cuda()

    a_all = torch.zeros(Sf.shape[0], device="cuda")
    for i0 in range(0, Sf.shape[0], args.chunk):
        b = slice(i0, min(i0 + args.chunk, Sf.shape[0]))
        Sb = Sg[b, None, :]                                      # (B,1,3)
        gi = idx_t[b]                                            # (B,knn)
        vb = valid[b]
        gi_safe = gi.clamp(max=xyz_t.shape[0] - 1)
        dxyz = Sb - xyz_t[gi_safe]                               # (B,knn,3)
        d_r = torch.einsum("bki,bkij->bkj", dxyz, R_t[gi_safe])
        maha = ((d_r / scl_t[gi_safe]) ** 2).sum(-1)             # (B,knn)
        contrib = alpha_t[gi_safe] * torch.exp(-0.5 * maha)
        contrib = torch.where(vb, contrib, torch.zeros_like(contrib))
        a_all[b] = torch.clamp(contrib.sum(-1), max=0.99)

    a = a_all.reshape(M, args.K)
    T = torch.cumprod(1.0 - a, dim=1)
    Pp = T * a
    thr = torch.clamp(0.05 * Pp.max(dim=1).values, min=0.01)
    Pc = Pp[:, 1:args.K-1]
    peak = (Pc > Pp[:, 0:args.K-2]) & (Pc >= Pp[:, 2:args.K]) & (Pc > thr[:, None])
    has_peak = peak.any(dim=1)
    idxv = peak.long().argmax(dim=1)[has_peak] + 1
    hv = origin[None, :] + d_w[has_peak] * t[idxv][:, None]
    m = hv.norm(dim=1) < 1.2
    hv = hv[m]
    if len(hv):
        hits_world.append(hv.cpu().numpy())
    if args.verify_gt and len(hv):
        dirs = d_w[has_peak][m].cpu().numpy()
        orig_np = origin.cpu().numpy()
        rayf = (getattr(mesh_tri.ray, "intersections_location", None)
                or getattr(mesh_tri.ray, "intersects_location"))
        pts, idxr, _ = rayf(np.repeat(orig_np[None, :], len(dirs), axis=0), dirs)
        t_all = np.linalg.norm(pts - orig_np, axis=1)   # every intersection, per ray
        t_gt = np.full(len(dirs), np.inf)
        np.minimum.at(t_gt, idxr, t_all)                # first hit per ray
        t_inv = t[idxv][m].cpu().numpy()
        ok = np.isfinite(t_gt)
        if ok.any():
            gt_depth_diffs.append(np.abs(t_inv[ok] - t_gt[ok]))
    if (vi + 1) % 8 == 0:
        print(f"  [pixel] view {vi+1}/{len(scene.getTrainCameras())} done, hits so far "
              f"{sum(len(h) for h in hits_world)}", flush=True)

pts = np.vstack(hits_world) if hits_world else np.zeros((0, 3))
print(f"[cloud] {args.obj}: total pixel-level hits={len(pts)}")

_, d_cloud2gt, _ = mesh_tri.nearest.on_surface(pts.astype(np.float64)) if len(pts) \
    else (None, np.array([]), None)
n_samp = 30000
samp = mesh_tri.sample(n_samp)
d_gt2cloud = cKDTree(pts.astype(np.float64)).query(samp)[0] if len(pts) else np.full(n_samp, np.inf)

resm = dict(
    obj=args.obj, step=args.step, hits=int(len(pts)),
    cloud2gt_p50=float(np.median(d_cloud2gt)) if len(pts) else None,
    cloud2gt_mean=float(np.mean(d_cloud2gt)) if len(pts) else None,
    gt2cloud_mean=float(np.mean(d_gt2cloud)),
    coverage_01=float(np.mean(d_gt2cloud < 0.1)),
    coverage_02=float(np.mean(d_gt2cloud < 0.2)),
    chamfer_mean=float((np.mean(d_cloud2gt) + np.mean(d_gt2cloud)) / 2) if len(pts) else None,
)
os.makedirs(args.dumps, exist_ok=True)
np.save(os.path.join(args.dumps, f"{args.obj}.npy"),
        np.hstack([pts, d_cloud2gt[:, None]]) if len(pts) else pts)
with open(os.path.join(args.dumps, f"{args.obj}.res.json"), "w") as f:
    json.dump(resm, f, indent=1)
print(json.dumps(resm, ensure_ascii=False))

if args.verify_gt:
    D = np.concatenate(gt_depth_diffs) if gt_depth_diffs else np.zeros(0)
    dr = (args.t_far - args.t_near) / (args.K - 1)
    ver = dict(
        obj=args.obj, step=args.step, n=int(len(D)),
        quant_step=float(dr),
        depth_p50=float(np.median(D)) if len(D) else None,
        depth_p90=float(np.percentile(D, 90)) if len(D) else None,
        depth_mean=float(np.mean(D)) if len(D) else None,
        frac_within_quant=float(np.mean(D < dr)) if len(D) else None,
    )
    with open(os.path.join(args.dumps, f"{args.obj}.verify.json"), "w") as f:
        json.dump(ver, f, indent=1)
    print(json.dumps(ver, ensure_ascii=False))
