"""E0: render depth maps from a Gaussian model (3DGS or PGSR ply) with gsplat,
back-project to a point cloud, and evaluate against the GT mesh.

This is the official inference form of PGSR (it supervises on rendered depth),
and it is the fairest surface-location metric that works for both 3DGS (finite
scale ~0.01) and PGSR (point-like scale ~1e-6) gaussians: every model answers
through its own renderer, exactly as an end user would use it.
"""
import argparse
import json
import os

import numpy as np
import torch
import trimesh
import plyfile
from scipy.spatial import cKDTree

import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--source", required=True, help="COLMAP dataset dir (data/sphere_rb)")
ap.add_argument("--obj", required=True)
ap.add_argument("--dumps", default="gsplat_depth")
ap.add_argument("--step", type=int, default=12, help="render subsample for eval")
ap.add_argument("--alpha_min", type=float, default=0.05)
ap.add_argument("--cam_style", default="standard", choices=["center", "standard"],
                help="center: colmap_writer 非标准约定（tvec=相机中心 C，合成数据）；"
                     "standard: 标准 COLMAP（t=-R·C，真实 SfM）。默认 standard。")
args = ap.parse_args()

# ---------------- load gaussians ----------------
v = plyfile.PlyData.read(args.ply)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opacities = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
n = len(means)
print(f"[depth] {n} gaussians, median scale {np.median(scales):.3g}, "
      f"median opacity {np.median(opacities):.3f}", flush=True)

colors = torch.zeros((n, 3), device="cuda")   # not used for depth
means_t = torch.from_numpy(means).cuda()
scales_t = torch.from_numpy(scales).cuda()
quats_t = torch.from_numpy(quats).cuda()
opac_t = torch.from_numpy(opacities).cuda()

# ---------------- cameras from COLMAP text ----------------
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
                 dtype=np.float32)          # world->cam
    t = np.array(tvec, dtype=np.float32)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R                                 # world->cam rotation
    if args.cam_style == "center":                 # colmap_writer 非标准：tvec 存相机中心 C
        vm[:3, 3] = -R @ t                         # W2C translation = -R@C
    else:                                          # 标准 COLMAP：tvec = -R·C
        vm[:3, 3] = t
    c = cam[cid]
    model, p = c["model"], c["p"]
    if model == "SIMPLE_PINHOLE":                  # f cx cy
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif model == "SIMPLE_RADIAL":                 # f cx cy k
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:                                          # PINHOLE / OPENCV / ... : fx fy cx cy [畸变]
        fx, fy, cx, cy = p[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    viewmats.append(vm)
    Ks.append(K)
viewmats = torch.from_numpy(np.stack(viewmats)).cuda()
Ks = torch.from_numpy(np.stack(Ks)).cuda()
any_cam = cam[list(cam.keys())[0]]
H, W = any_cam["h"], any_cam["w"]

# ---------------- render depth ----------------
print(f"[depth] rendering {len(viewmats)} views at {W}x{H}...", flush=True)
pts_all = []
for b0 in range(0, len(viewmats), 8):
    vm = viewmats[b0:b0 + 8]
    Kk = Ks[b0:b0 + 8]
    # render_mode="D" -> accumulated z-depth (camera coords). colors is
    # post-activation RGB here (sh_degree left None), needed by the rasterizer.
    out = gsplat.rasterization(
        means_t, quats_t, scales_t, opac_t, colors, vm, Kk, W, H,
        render_mode="D")
    depth = out[0][..., 0].cpu().numpy()                       # (B,H,W) camera-z
    alpha = out[1][..., 0].cpu().numpy()
    Kk_np = Kk.cpu().numpy()
    vm_np = vm.cpu().numpy()
    u = np.arange(0, W, args.step, dtype=np.float32) + args.step / 2
    vv = np.arange(0, H, args.step, dtype=np.float32) + args.step / 2
    V, U = np.meshgrid(vv, u, indexing="ij")
    for bi in range(depth.shape[0]):
        Dz = depth[bi][::args.step, ::args.step]
        Av = alpha[bi][::args.step, ::args.step]
        ok = (Av > args.alpha_min) & np.isfinite(Dz) & (Dz > 0) & (Dz < 6.0)
        if not ok.any():
            continue
        # camera-z back-projection (gsplat "D" = accumulated z-depth)
        Kb = Kk_np[bi]
        pc = np.stack([(U[ok] - Kb[0, 2]) / Kb[0, 0] * Dz[ok],
                       (V[ok] - Kb[1, 2]) / Kb[1, 1] * Dz[ok],
                       Dz[ok]], 1)                              # cam coords
        pc_h = np.concatenate([pc, np.ones((len(pc), 1))], 1)
        pw = (np.linalg.inv(vm_np[bi]) @ pc_h.T).T[:, :3]       # world
        keep = np.linalg.norm(pw, axis=1) < 1.2
        pts_all.append(pw[keep])
    if (b0 // 8) % 3 == 0:
        print(f"  rendered {b0 + 8}/{len(viewmats)} views, "
              f"valid pts so far {sum(len(p) for p in pts_all)}", flush=True)

pts = np.vstack(pts_all) if pts_all else np.zeros((0, 3))
print(f"[cloud] {args.obj}: depth point cloud = {len(pts)} pts")

gt = trimesh.load(os.path.join(args.source, "..", "meshes", args.obj + ".ply"),
                  force="mesh")
_, d_cloud2gt, _ = gt.nearest.on_surface(pts.astype(np.float64)) if len(pts) \
    else (None, np.array([]), None)
samp = gt.sample(30000)
d_gt2cloud = cKDTree(pts.astype(np.float64)).query(samp)[0] if len(pts) else np.full(30000, np.inf)
r = np.linalg.norm(pts, axis=1)
res = dict(
    obj=args.obj, n_pts=int(len(pts)),
    cloud2gt_p50=float(np.median(d_cloud2gt)) if len(pts) else None,
    cloud2gt_mean=float(np.mean(d_cloud2gt)) if len(pts) else None,
    gt2cloud_mean=float(np.mean(d_gt2cloud)),
    coverage_01=float(np.mean(d_gt2cloud < 0.1)),
    coverage_02=float(np.mean(d_gt2cloud < 0.2)),
    chamfer_mean=float((np.mean(d_cloud2gt) + np.mean(d_gt2cloud)) / 2) if len(pts) else None,
    radius_p50=float(np.percentile(r, 50)) if len(pts) else None,
    radius_p90=float(np.percentile(r, 90)) if len(pts) else None,
    frac_radius_lt_07=float(np.mean(r < 0.7)) if len(pts) else None,
)
os.makedirs(args.dumps, exist_ok=True)
np.save(os.path.join(args.dumps, f"{args.obj}.npy"), pts)
with open(os.path.join(args.dumps, f"{args.obj}.res.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, ensure_ascii=False))
