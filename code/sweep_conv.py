"""sweep_conv.py: 扫描 (相机约定, 3D filter) 组合,找出训练时的渲染配置。

对 view_0000 渲染模型,与 GT 图像比 PSNR。训练日志 PSNR≈27,唯一接近的组合即训练配置。
用法:python sweep_conv.py --ply <ply> --source <colmap> --gt_img <GT图路径>
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
ap.add_argument("--gt_img", required=True)
ap.add_argument("--view", type=int, default=0)
args = ap.parse_args()

v = plyfile.PlyData.read(args.ply)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opac = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
rgb = (0.5 + 0.28209479177387814 * dc).clip(0, 1).astype(np.float32)
filt = np.asarray(v["filter_3D"], dtype=np.float32).reshape(-1, 1) if "filter_3D" in v.data.dtype.names else None
if filt is not None:
    print(f"[sweep] filter_3D 存在, 中位={np.median(filt):.4f}")

means_t = torch.from_numpy(means).cuda()
quats_t = torch.from_numpy(quats).cuda()
rgb_t = torch.from_numpy(rgb).cuda()

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
c = cam[cid]
model, p = c["model"], c["p"]
if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
    fx = fy = p[0]; cx, cy = p[1], p[2]
else:
    fx, fy, cx, cy = p[:4]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
H, W = c["h"], c["w"]

gt = np.asarray(Image.open(args.gt_img).convert("RGB")).astype(np.float32) / 255.0
if gt.shape[:2] != (H, W):
    gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize((W, H))
                    ).astype(np.float32) / 255.0

combos = [
    ("center+filter",   Rw2c,     -Rw2c @ t,   True),
    ("center+raw",      Rw2c,     -Rw2c @ t,   False),
    ("standard+filter", Rw2c,     t,           True),
    ("standard+raw",    Rw2c,     t,           False),
    ("gof+filter",      Rw2c.T,   -Rw2c.T @ t, True),
    ("gof+raw",         Rw2c.T,   -Rw2c.T @ t, False),
    ("origin-Rw2cT+f",  Rw2c.T,   np.zeros(3), True),
    ("origin-Rw2cT+r",  Rw2c.T,   np.zeros(3), False),
    ("origin-Rw2c+f",   Rw2c,     np.zeros(3), True),
    ("origin-Rw2c+r",   Rw2c,     np.zeros(3), False),
]

print(f"\nview {args.view} PSNR vs GT (训练日志参考 ≈27):")
for name, Rr, tt, use_filter in combos:
    sc = scales.copy()
    op = opac.copy()
    if use_filter and filt is not None:
        det1 = np.prod(sc ** 2, axis=1)
        sc = np.sqrt(sc ** 2 + filt ** 2)
        det2 = np.prod(sc ** 2, axis=1)
        op = op * np.sqrt(det1 / np.maximum(det2, 1e-12))
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = Rr
    vm[:3, 3] = tt
    out = gsplat.rasterization(
        torch.from_numpy(means).cuda(), quats_t, torch.from_numpy(sc).cuda(),
        torch.from_numpy(op).cuda(), rgb_t,
        torch.from_numpy(vm).cuda().reshape(1, 4, 4),
        torch.from_numpy(K).cuda().reshape(1, 3, 3), W, H, render_mode="RGB+D")
    r = out[0][0, :, :, :3].cpu().numpy()
    a = out[1][0, :, :, 0].cpu().numpy()
    fg = a > 0.5
    if fg.sum() < 50:
        ps = "  无fg"
    else:
        mse = float(((r[fg] - gt[fg]) ** 2).mean())
        ps = f"{10*np.log10(1/max(mse,1e-10)):6.2f}"
    print(f"  {name:20s}: PSNR(fg) {ps}   fg_px={fg.sum():6d}")
