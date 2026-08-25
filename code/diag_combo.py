"""Diagnose why combo produces points only in the top cap (theta<60).
Per view print depth_ok / rayne-hit counts and hit geometry, grouped by elevation.
"""
import argparse
import os

import numpy as np
import torch
import plyfile
from scipy.spatial import cKDTree

import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--source", required=True)
ap.add_argument("--step", type=int, default=12)
ap.add_argument("--t_high", type=float, default=0.10)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--r_ref", type=float, default=0.25)
ap.add_argument("--t_tol", type=float, default=0.15)
ap.add_argument("--alpha_min", type=float, default=0.05)
ap.add_argument("--t_near", type=float, default=0.02)
ap.add_argument("--t_far", type=float, default=6.0)
ap.add_argument("--n_ref", type=int, default=40)
args = ap.parse_args()

v = plyfile.PlyData.read(args.ply)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opacities = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
n = len(means)

colors = torch.zeros((n, 3), device="cuda")
means_t = torch.from_numpy(means).cuda()
scales_t = torch.from_numpy(scales).cuda()
quats_t = torch.from_numpy(quats).cuda()
opac_t = torch.from_numpy(opacities).cuda()
alpha_t = torch.from_numpy(opacities).cuda()
kd = cKDTree(means.astype(np.float64))

cam = {}
with open(os.path.join(args.source, "sparse", "0", "cameras.txt")) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cam[int(p[0])] = {"model": p[1], "w": int(p[2]), "h": int(p[3]),
                          "p": [float(x) for x in p[4:]]}
imgs = []
with open(os.path.join(args.source, "sparse", "0", "images.txt")) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or " " not in line:
            continue
        p = line.split()
        if len(p) < 10:
            continue
        imgs.append((int(p[0]), [float(x) for x in p[1:5]], [float(x) for x in p[5:8]],
                     int(p[8]), p[9]))
imgs.sort()
viewmats, Ks = [], []
for iid, qvec, tvec, cid, name in imgs:
    qw, qx, qy, qz = qvec
    R = np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]],
                 dtype=np.float32)
    t = np.array(tvec, dtype=np.float32)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R
    vm[:3, 3] = t
    c = cam[cid]
    fx, fy, cx, cy = c["p"][:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    viewmats.append(vm)
    Ks.append(K)
viewmats_t = torch.from_numpy(np.stack(viewmats)).cuda()
Ks_t = torch.from_numpy(np.stack(Ks)).cuda()
any_cam = cam[list(cam.keys())[0]]
H, W = any_cam["h"], any_cam["w"]
origins = np.stack([-R.T @ t for R, t in
                    ((vm[:3, :3], vm[:3, 3]) for vm in viewmats)], 0).astype(np.float32)
Rs = np.stack([vm[:3, :3] for vm in viewmats], 0).astype(np.float32)

# per-view elevation: from camera center -> polar angle of camera around object
cam_pol = np.degrees(np.arccos(np.clip(origins[:, 2] / np.linalg.norm(origins, axis=1), -1, 1)))
cam_az = np.degrees(np.arctan2(origins[:, 1], origins[:, 0]))
print("view | pol_deg | camdist | depth_ok | rayne_hit | rayne_ok_r<1.1 | rayne r50 | rayne theta50 | rayne theta<60 frac")

u = np.arange(0, W, args.step, dtype=np.float32) + args.step / 2
vv = np.arange(0, H, args.step, dtype=np.float32) + args.step / 2
V, U = np.meshgrid(vv, u, indexing="ij")
Uf = U.reshape(-1).astype(np.float32)
Vf = V.reshape(-1).astype(np.float32)
PIX = Uf.shape[0]
t_ref = np.linspace(args.t_near, args.t_far, args.n_ref)

for vi in range(len(viewmats)):
    out = gsplat.rasterization(
        means_t, quats_t, scales_t, opac_t, colors,
        viewmats_t[vi:vi + 1], Ks_t[vi:vi + 1], W, H, render_mode="D")
    Dz = out[0][0, ::args.step, ::args.step, 0].ravel().cpu().numpy()
    Av = out[1][0, ::args.step, ::args.step, 0].ravel().cpu().numpy()
    Kb = Ks[vi]
    fx, fy, cx, cy = Kb[0, 0], Kb[1, 1], Kb[0, 2], Kb[1, 2]

    o_np = origins[vi][None, :]
    d_cam = np.stack([(Uf - cx) / fx, (Vf - cy) / fy, np.ones(PIX)], 1)
    d_w = d_cam @ Rs[vi].T
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    refs = o_np + d_w[:, None, :] * t_ref[None, :, None]
    dist, idx = kd.query(refs.reshape(-1, 3), k=args.k, workers=-1,
                         distance_upper_bound=args.r_ref)
    gi = torch.from_numpy(idx.astype(np.int64)).cuda().reshape(PIX, -1)
    valid = (gi < n)
    gi_safe = gi.clamp(max=n - 1)
    p = means_t[gi_safe]
    o_c = torch.from_numpy(o_np).cuda()
    d_c = torch.from_numpy(d_w.astype(np.float32)).cuda()
    tvec = ((p - o_c) @ d_c[:, :, None])[..., 0]
    pdist2 = ((p - o_c) ** 2).sum(-1) - tvec ** 2
    on_ray = (tvec > args.t_near) & (pdist2 < args.t_tol ** 2) & valid
    ok_r = on_ray & (alpha_t[gi_safe] >= args.alpha_min)
    t_sel = torch.where(ok_r, tvec, torch.full_like(tvec, float("inf")))
    t_min = t_sel.min(dim=1).values.cpu().numpy()
    has_rayne = np.isfinite(t_min) & (t_min > 0)
    hit_pt = origins[vi][None, :] + d_w * t_min[:, None]

    depth_ok = (Av >= args.t_high) & np.isfinite(Dz) & (Dz > 0) & (Dz < args.t_far)
    hr = np.linalg.norm(hit_pt, axis=1)
    in_shell = (hr < 1.1)
    sel = has_rayne & in_shell
    if sel.any():
        rsel = hr[sel]
        thsel = np.degrees(np.arccos(np.clip(hit_pt[sel, 2] / rsel, -1, 1)))
        info = f"| {np.median(rsel):.3f} | {np.median(thsel):.1f} | {(thsel<60).mean():.2f}"
    else:
        info = "|  - |  - |  -"
    print(f"{vi:4d} | {cam_pol[vi]:7.1f} | {np.linalg.norm(origins[vi]):7.3f} "
          f"| {depth_ok.sum():8d} | {has_rayne.sum():9d} | {sel.sum():12d} {info}")
