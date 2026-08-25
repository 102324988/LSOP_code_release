"""compare_gof_render.py: 用 GOF E0-fix 相机的精确 W2C 渲染单视角,与 GOF render.py 输出对比。

判定:我的 gsplat 渲染路径是否忠实复现 GOF 后训练渲染(同相机、同 3D filter)。
若 PSNR 高 -> 路径正确,合成数据低 PSNR 是相机约定差异;若低 -> 我的 gsplat 路径有 bug。
用法:python compare_gof_render.py --ply <ply> --source <colmap> --gof_png <render.py输出的00000.png> [--view 0]
"""
import argparse
import os

import numpy as np
import plyfile
import torch
from PIL import Image

import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--source", required=True)
ap.add_argument("--gof_png", required=True)
ap.add_argument("--view", type=int, default=0)
args = ap.parse_args()

v = plyfile.PlyData.read(args.ply)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opac = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
rgb = (0.5 + 0.28209479177387814 * dc).clip(0, 1).astype(np.float32)

# 3D filter 与 GOF 一致
if "filter_3D" in v.data.dtype.names:
    filt = np.asarray(v["filter_3D"], dtype=np.float32).reshape(-1, 1)
    det1 = np.prod(scales ** 2, axis=1)
    scales = np.sqrt(scales ** 2 + filt ** 2)
    det2 = np.prod(scales ** 2, axis=1)
    opac = opac * np.sqrt(det1 / np.maximum(det2, 1e-12))
    print(f"[cmp] filter 施加, opac 衰减中位={np.median(np.sqrt(det1/np.maximum(det2,1e-12))):.4f}", flush=True)

means_t = torch.from_numpy(means).cuda()
scales_t = torch.from_numpy(scales).cuda()
quats_t = torch.from_numpy(quats).cuda()
opac_t = torch.from_numpy(opac).cuda()
rgb_t = torch.from_numpy(rgb).cuda()

# 相机:GOF E0-fix 约定  R = qvec2rotmat(q).T, W2C = [[R, -R@tvec]]
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
iid, qvec, tvec, cid, name = imgs[args.view]
qw, qx, qy, qz = qvec
Rw2c = np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                 [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                 [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]],
                dtype=np.float32)
t = np.array(tvec, dtype=np.float32)
R = Rw2c.T                                   # GOF 读法: transpose(qvec2rotmat)
vm = np.eye(4, dtype=np.float32)
vm[:3, :3] = R
vm[:3, 3] = -R @ t                           # E0-fix W2C 平移
c = cam[cid]
model, p = c["model"], c["p"]
fx = fy = p[0] if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL") else p[0]
cx, cy = p[1], p[2]
if model not in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
    fx, fy, cx, cy = p[:4]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
H, W = c["h"], c["w"]

out = gsplat.rasterization(means_t, quats_t, scales_t, opac_t, rgb_t,
                           torch.from_numpy(vm).cuda().reshape(1, 4, 4),
                           torch.from_numpy(K).cuda().reshape(1, 3, 3), W, H,
                           render_mode="RGB+D")
mine = out[0][0, :, :, :3].cpu().numpy()

gof = np.asarray(Image.open(args.gof_png).convert("RGB")).astype(np.float32) / 255.0
if gof.shape[:2] != (H, W):
    gof = np.asarray(Image.fromarray((gof * 255).astype(np.uint8)).resize((W, H))
                     ).astype(np.float32) / 255.0

mse = float(((mine - gof) ** 2).mean())
psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
diff = np.abs(mine - gof).mean(2)
print(f"[cmp] 我的 gsplat vs GOF render: PSNR={psnr:.2f}  mean_abs={diff.mean():.4f}")
print(f"[cmp] 像素级 diff 分布: p50={np.median(diff):.3f} p90={np.percentile(diff,90):.3f} "
      f"p99={np.percentile(diff,99):.3f} max={diff.max():.3f}")
# alpha 一致性:GOF 渲染无 alpha,略。输出各自均值供参考
print(f"[cmp] 我的渲染 RGB 均值={mine.mean():.3f}  GOF RGB 均值={gof.mean():.3f}")
np.save("/tmp/mine_rgb.npy", mine)
