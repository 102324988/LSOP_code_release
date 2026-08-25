#!/usr/bin/env python3
"""dtu_to_colmap.py: DTU 场景 → COLMAP 布局（GOF / 固定前段管线直接可消费）。

- images/         : 49 张 rect 图像（光照 _3_ 最漫射，标准 MVS 口径）
- sparse/0/cameras.txt  : PINHOLE（K 分解自 pos_XXX.txt 投影矩阵 P）
- sparse/0/images.txt   : GOF 约定（tvec=相机中心 C, qvec=quat(R_cw)，
                         即 qvec2rotmat(q).T=world→cam，与合成管线 colmap_writer 同约定）
- sparse/0/points3D.txt : ObsMask 过滤后的 stl GT 点降采样初始化（默认 40k）

DTU 帧 = 结构光左相机帧，与 stl GT 同帧 → 重建自动对齐 GT（chamfer 免 ICP）。
用法: python dtu_to_colmap.py --scan 1 --src <.../MVS Data> --out <数据目录> [--npos 49] [--ninit 40000]
"""
import argparse
import os

import numpy as np
import plyfile
import scipy.io as sio
from scipy.linalg import rq
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--scan", type=int, required=True)
ap.add_argument("--src", required=True, help=".../MVS Data")
ap.add_argument("--out", required=True, help="数据目录（将建 images/ sparse/0/）")
ap.add_argument("--npos", type=int, default=49, help="相机位姿数（scan1-41 用 49）")
ap.add_argument("--ninit", type=int, default=40000, help="初始化点数")
args = ap.parse_args()

s = args.scan
rect_dir = os.path.join(args.src, "Rectified", f"scan{s}")
cal_dir = os.path.join(args.src, "Calibration", "cal18")
stl_path = os.path.join(args.src, "Points", "stl", f"stl{s:03d}_total.ply")
obs_path = os.path.join(args.src, "ObsMask", f"ObsMask{s}_10.mat")

# ---- 位姿分解 ----
def pose_of(p):
    """pos_XXX.txt 3x4 投影矩阵 → (K, Rc, C)。x_cam=Rc·(X−C), 像素=K·x_cam。"""
    P = np.loadtxt(p)
    K, Rc = rq(P[:, :3])                      # K 上三角, Rc world->cam
    d = np.diag(np.sign(np.diag(K)))          # 修正 K 对角为正
    K, Rc = K @ d, d @ Rc
    t_cam = np.linalg.solve(K, P[:, 3])
    C = -Rc.T @ t_cam
    return K, Rc, C

# ---- stl 初始化点（ObsMask 过滤）----
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
print(f"[dtu] stl 物体点: {len(Xo)}", flush=True)

# ---- 写 COLMAP 布局 ----
os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
os.makedirs(os.path.join(args.out, "sparse", "0"), exist_ok=True)

K0, Rc0, C0 = pose_of(os.path.join(cal_dir, "pos_001.txt"))
fx, fy, cx, cy = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
with open(os.path.join(args.out, "sparse", "0", "cameras.txt"), "w") as f:
    f.write("# Camera list with one line of data per camera:\n")
    f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write(f"1 PINHOLE 1600 1200 {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

imgs = []
with open(os.path.join(args.out, "sparse", "0", "images.txt"), "w") as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    for i in range(1, args.npos + 1):
        K, Rc, C = pose_of(os.path.join(cal_dir, f"pos_{i:03d}.txt"))
        q = Rotation.from_matrix(Rc.T).as_quat()          # [x,y,z,w] = R_cw
        name = f"rect_{i:03d}_3_r5000.png"
        src = os.path.join(rect_dir, name)
        dst = os.path.join(args.out, "images", name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        f.write(f"{i} {q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} "
                f"{C[0]:.8f} {C[1]:.8f} {C[2]:.8f} 1 {name}\n\n")
        imgs.append(name)

rng = np.random.default_rng(0)
sel = rng.choice(len(Xo), size=min(args.ninit, len(Xo)), replace=False)
with open(os.path.join(args.out, "sparse", "0", "points3D.txt"), "w") as f:
    f.write("# 3D point list with one line of data per point:\n")
    f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
    for j, pi in enumerate(sel):
        x, y, z = Xo[pi]
        f.write(f"{j+1} {x:.4f} {y:.4f} {z:.4f} 200 200 200 0.1\n")

print(f"[dtu] 完成: {args.out}  ({len(imgs)} 图, {min(args.ninit, len(Xo))} 初始化点)")
print(f"[dtu] K=({fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f})  相机0中心=({C0[0]:.1f},{C0[1]:.1f},{C0[2]:.1f})")
