"""test_geom_conv.py: 判定哪个相机约定对应物理真实几何。

方法:对 view_0000,在 center 与 GOF 两种约定下渲染深度 -> 反投影为世界点云
-> 与 GT 网格表面(顶点)最近邻距离。距离小者 = 该约定几何正确。

用法:python test_geom_conv.py --ply <model.ply> --source <colmap> --mesh <gt.ply> [--views 0-8]
"""
import argparse
import os

import numpy as np
import plyfile
import torch
from scipy.spatial import cKDTree

import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--source", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--views", default="0-4")
args = ap.parse_args()

# ---- 高斯模型 ----
v = plyfile.PlyData.read(args.ply)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opac = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
rgb = (0.5 + 0.28209479177387814 * dc).clip(0, 1).astype(np.float32)
if "filter_3D" in v.data.dtype.names:
    filt = np.asarray(v["filter_3D"], dtype=np.float32).reshape(-1, 1)
    det1 = np.prod(scales ** 2, axis=1)
    scales = np.sqrt(scales ** 2 + filt ** 2)
    det2 = np.prod(scales ** 2, axis=1)
    opac = opac * np.sqrt(det1 / np.maximum(det2, 1e-12))
means_t = torch.from_numpy(means).cuda()
scales_t = torch.from_numpy(scales).cuda()
quats_t = torch.from_numpy(quats).cuda()
opac_t = torch.from_numpy(opac).cuda()
rgb_t = torch.from_numpy(rgb).cuda()

# ---- 相机 ----
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

# ---- GT 网格表面点 ----
mv = plyfile.PlyData.read(args.mesh)["vertex"]
mesh_pts = np.stack([mv["x"], mv["y"], mv["z"]], 1).astype(np.float64)
tree = cKDTree(mesh_pts)
# 网格中位数尺度(用 bbox 半对角粗略)
mesh_scale = float(np.median(np.linalg.norm(mesh_pts - mesh_pts.mean(0), axis=1)))

def backproj(depth, alpha, K, W, H, step=3):
    u = np.arange(0, W, step, dtype=np.float32) + step / 2
    vv = np.arange(0, H, step, dtype=np.float32) + step / 2
    V, U = np.meshgrid(vv, u, indexing="ij")
    Uf, Vf = U.reshape(-1), V.reshape(-1)
    Dz = depth[::step, ::step].ravel()
    Av = alpha[::step, ::step].ravel()
    ok = (Av > 0.3) & np.isfinite(Dz) & (Dz > 0)
    pc = np.stack([(Uf[ok] - K[0, 2]) / K[0, 0] * Dz[ok],
                   (Vf[ok] - K[1, 2]) / K[1, 1] * Dz[ok], Dz[ok]], 1)
    return pc, ok.sum()

lo, hi = (int(x) for x in args.views.split("-"))
for gi in range(lo, hi):
    iid, qvec, tvec, cid, name = imgs[gi]
    qw, qx, qy, qz = qvec
    Rw2c = np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]],
                    dtype=np.float32)
    t = np.array(tvec, dtype=np.float32)
    c = cam[cid]
    model, p = c["model"], c["p"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    Hh, Ww = c["h"], c["w"]

    vms = []
    for R_, t_ in ((Rw2c, -Rw2c @ t), (Rw2c.T, -Rw2c.T @ t), (Rw2c, t)):
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = R_
        m[:3, 3] = t_
        vms.append(m)
    for conv_name, vm in [("center(物理)", vms[0]), ("GOF(E0fix)", vms[1]),
                          ("standard", vms[2])]:
        vmt = torch.from_numpy(vm.reshape(1, 4, 4)).cuda()
        Kt = torch.from_numpy(K.reshape(1, 3, 3)).cuda()
        out = gsplat.rasterization(means_t, quats_t, scales_t, opac_t, rgb_t, vmt, Kt,
                                   Ww, Hh, render_mode="RGB+D")
        dep = out[0][0, :, :, 3].cpu().numpy()
        alp = out[1][0, :, :, 0].cpu().numpy()
        pc, npts = backproj(dep, alp, K, Ww, Hh)
        if npts == 0:
            print(f"view {gi} {conv_name}: 无有效点")
            continue
        # 世界坐标
        vm_inv = np.linalg.inv(vm)
        pcw = (vm_inv @ np.concatenate([pc, np.ones((npts, 1))], 1).T).T[:, :3]
        dist, _ = tree.query(pcw.astype(np.float64))
        rel = float(np.median(dist) / mesh_scale)
        print(f"view {gi} {conv_name}: npts={npts:5d}  网格NN中位={np.median(dist):.4f}  "
              f"相对中位尺度={rel:.3f}  90%={np.percentile(dist,90):.4f}")
