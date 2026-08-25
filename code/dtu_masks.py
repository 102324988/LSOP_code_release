#!/usr/bin/env python3
"""dtu_masks.py: 从 ObsMask 过滤的 stl GT 点云 + DTU 位姿生成逐图像前景 mask。

对每个位姿把物体点投影到图像，标记落点像素并做形态学膨胀，得到前景 mask PNG。
mask 名与图像名一致（rect_001_3_r5000.png），供 eval_no_gt --mask_dir / combo 掩膜用。
用法: python dtu_masks.py --scan 1 --src <.../MVS Data> --out <mask_dir> [--npos 49]
"""
import argparse
import os

import numpy as np
import plyfile
import scipy.io as sio
from scipy.linalg import rq
from scipy.ndimage import binary_dilation
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--scan", type=int, required=True)
ap.add_argument("--src", required=True, help=".../MVS Data")
ap.add_argument("--out", required=True)
ap.add_argument("--npos", type=int, default=49)
ap.add_argument("--radius", type=int, default=3, help="投影落点膨胀半径(px)")
args = ap.parse_args()

s = args.scan
cal_dir = os.path.join(args.src, "Calibration", "cal18")
stl_path = os.path.join(args.src, "Points", "stl", f"stl{s:03d}_total.ply")
obs_path = os.path.join(args.src, "ObsMask", f"ObsMask{s}_10.mat")

# ---- 物体点（ObsMask 3D 占用过滤）----
v = plyfile.PlyData.read(stl_path)["vertex"]
Xs = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
obs = sio.loadmat(obs_path)
BB, Res, grid = obs["BB"], obs["Res"].item(), obs["ObsMask"]
mn, mx = BB[0], BB[1]
inbb = ((Xs >= mn) & (Xs <= mx)).all(1)
idx = np.floor((Xs[inbb] - mn) / Res).astype(np.int64)
ok = ((idx[:, 0] < grid.shape[2]) & (idx[:, 1] < grid.shape[1]) & (idx[:, 2] < grid.shape[0])
      & (idx >= 0).all(1))
occ = grid[idx[ok][:, 2], idx[ok][:, 1], idx[ok][:, 0]] > 0
Xo = Xs[inbb][ok][occ]
print(f"[masks] 物体点: {len(Xo)}", flush=True)

# 预分解全部位姿
poses = []
for i in range(1, args.npos + 1):
    P = np.loadtxt(os.path.join(cal_dir, f"pos_{i:03d}.txt"))
    K, Rc = rq(P[:, :3])
    d = np.diag(np.sign(np.diag(K))); K, Rc = K @ d, d @ Rc
    t_cam = np.linalg.solve(K, P[:, 3])
    C = -Rc.T @ t_cam
    poses.append((K, Rc, C))

os.makedirs(args.out, exist_ok=True)
W, H = 1600, 1200
disk = np.ones((2 * args.radius + 1, 2 * args.radius + 1), dtype=bool) if args.radius > 0 else None

for i, (K, Rc, C) in enumerate(poses, start=1):
    xc = (Rc @ (Xo - C).T).T
    z = xc[:, 2]
    uv = (K @ xc.T).T
    u, vv = uv[:, 0] / z, uv[:, 1] / z
    vis = (z > 0) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    ui, vi = u[vis].astype(np.int32), vv[vis].astype(np.int32)
    m = np.zeros((H, W), dtype=np.uint8)
    m[vi, ui] = 255
    if disk is not None:
        m = binary_dilation(m > 0, disk).astype(np.uint8) * 255
    name = f"rect_{i:03d}_3_r5000.png"
    Image.fromarray(m).save(os.path.join(args.out, name))
    if i % 8 == 0:
        print(f"  {i}/{args.npos} done, fill={m.mean()/255:.3f}", flush=True)

print(f"[masks] {args.npos} masks -> {args.out}")
