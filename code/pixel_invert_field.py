"""E0: pixel-level full-surface inversion from the GOF opacity grid.

For every (subsampled) pixel of every training view, march the camera ray
through the field grid, accumulate T, find the FIRST peak of P = T*alpha,
backproject to a surface hit. This is Path A's real application form
(image -> per-pixel surface depth -> point cloud) and overcomes the 3042-ray
manual-grid coverage limit of eval_cloud.py.

Metrics identical to eval_cloud.py: cloud->GT accuracy (p50/mean), GT->cloud
coverage (@0.1/@0.2), symmetric Chamfer. Dumps the full cloud as <obj>.npy.
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

ap = argparse.ArgumentParser()
lp = ModelParams(ap)
pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
ap.add_argument("--grid", required=True, help="dir with grid.npy + meta.json")
ap.add_argument("--obj", required=True)
ap.add_argument("--dumps", default="pixel_clouds")
ap.add_argument("--step", type=int, default=8, help="pixel subsample step")
ap.add_argument("--K", type=int, default=160, help="samples along ray")
ap.add_argument("--t_near", type=float, default=0.02)
ap.add_argument("--t_far", type=float, default=6.0)
ap.add_argument("--chunk_px", type=int, default=200000)
args = ap.parse_args()
ds = lp.extract(args)
pipe = pp.extract(args)

grid = np.load(os.path.join(args.grid, "grid.npy"))  # (res,res,res)
with open(os.path.join(args.grid, "meta.json")) as f:
    meta = json.load(f)
bbox = np.array(meta["bbox"], dtype=np.float32)
res = grid.shape[0]
grid_t = torch.from_numpy(grid).float().cuda()
b0 = torch.tensor(bbox[0::2], dtype=torch.float32, device="cuda")
bs = torch.tensor(bbox[1::2] - bbox[0::2], dtype=torch.float32, device="cuda")


def trilinear(pts):
    """pts (M,3) world -> opacity (M,)."""
    g = (pts - b0) / bs * (res - 1)
    lo = torch.floor(g).long().clamp(0, res - 2)
    hi = lo + 1
    w = (g - lo.float()).clamp(0, 1)
    def gg(x, y, z):
        return grid_t[x, y, z]
    c000 = gg(lo[:, 0], lo[:, 1], lo[:, 2]); c100 = gg(hi[:, 0], lo[:, 1], lo[:, 2])
    c010 = gg(lo[:, 0], hi[:, 1], lo[:, 2]); c110 = gg(hi[:, 0], hi[:, 1], lo[:, 2])
    c001 = gg(lo[:, 0], lo[:, 1], hi[:, 2]); c101 = gg(hi[:, 0], lo[:, 1], hi[:, 2])
    c011 = gg(lo[:, 0], hi[:, 1], hi[:, 2]); c111 = gg(hi[:, 0], hi[:, 1], hi[:, 2])
    x1, x0 = w[:, 0], 1 - w[:, 0]
    y1, y0 = w[:, 1], 1 - w[:, 1]
    z1, z0 = w[:, 2], 1 - w[:, 2]
    return (x1 * (y1 * (z1 * c000 + z0 * c001) + y0 * (z1 * c010 + z0 * c011))
            + x0 * (y1 * (z1 * c100 + z0 * c101) + y0 * (z1 * c110 + z0 * c111)))


g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)

mesh = trimesh.load(os.path.join(ds.source_path, "..", "meshes", args.obj + ".ply"),
                    force="mesh")
mesh_tri = trimesh.Trimesh(vertices=mesh.vertices.astype(np.float64), faces=mesh.faces)

t = torch.linspace(args.t_near, args.t_far, args.K, device="cuda")
dr = (args.t_far - args.t_near) / (args.K - 1)

hits_world = []
for vi, cam in enumerate(scene.getTrainCameras()):
    W, H = cam.image_width, cam.image_height
    fx, fy = cam.focal_x, cam.focal_y
    cx, cy = W / 2.0, H / 2.0
    u = torch.arange(0, W, args.step, device="cuda").float() + args.step / 2
    v = torch.arange(0, H, args.step, device="cuda").float() + args.step / 2
    V, U = torch.meshgrid(v, u, indexing="ij")
    X = (U - cx) / fx
    Y = (V - cy) / fy
    d_cam = torch.stack([X, Y, torch.ones_like(X)], dim=-1)  # (Hsub,Wsub,3)
    R = torch.tensor(cam.R, dtype=torch.float32, device="cuda")  # world->cam
    d_world = d_cam @ R.T  # cam->world
    d_world = d_world / d_world.norm(dim=-1, keepdim=True)
    origin = torch.tensor(cam.camera_center, dtype=torch.float32, device="cuda")

    P = d_world.reshape(-1, 3)  # (M,3)
    M = P.shape[0]
    for i0 in range(0, M, args.chunk_px):
        d = P[i0:i0 + args.chunk_px]
        S = origin[None, :] + d[:, None, :] * t[None, :, None]  # (C,K,3)
        o = trilinear(S.reshape(-1, 3)).reshape(-1, args.K)     # (C,K)
        alpha = torch.clamp(o, 0, 0.99)
        T = torch.cumprod(1.0 - alpha, dim=1)
        Pp = T * alpha
        thr = torch.clamp(0.05 * Pp.max(dim=1).values, min=0.01)  # (C,)
        Pc = Pp[:, 1:args.K-1]                                   # candidate samples [1, K-2]
        peak = (Pc > Pp[:, 0:args.K-2]) & (Pc >= Pp[:, 2:args.K]) & (Pc > thr[:, None])
        has_peak = peak.any(dim=1)
        idx = peak.long().argmax(dim=1)          # first peak index (valid where has_peak)
        idxv = idx[has_peak] + 1
        hv = origin[None, :] + d[has_peak] * t[idxv][:, None]
        m = hv.norm(dim=1)  # drop points outside the eval bbox
        hv = hv[m < 1.2]
        if len(hv):
            hits_world.append(hv.cpu().numpy())
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
