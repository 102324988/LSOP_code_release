#!/usr/bin/env python3
"""diag_render_psnr.py: 对照 GOF 自带全图 PSNR 校验 eval_no_gt 渲染约定。

对训练视角算三种口径的全图 PSNR（整图 / α 前景 / mask 内），与 GOF 自带 26.89 对照。
"""
import os
import sys

import numpy as np
import plyfile
import torch
from PIL import Image

import gsplat

PLY = sys.argv[1]
SRC = sys.argv[2]
MASKS = sys.argv[3] if len(sys.argv) > 3 else None

v = plyfile.PlyData.read(PLY)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opac = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
names = list(v.data.dtype.names)
frest = [c for c in names if c.startswith("f_rest_")]
n_bands = int(round((len(frest) / 3 + 1) ** 0.5))
sh_degree = n_bands - 1
dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
if n_bands > 1:
    rest = np.stack([v[f"f_rest_{j}"] for j in range(len(frest))], 1).astype(np.float32)
    features = np.concatenate([dc, rest], 1)   # [N, 3*n_bands^2]
    colors_sh = torch.from_numpy(features.reshape(-1, n_bands * n_bands, 3)).cuda()   # [N, K, 3]
    print(f"[diag] 全 SH: sh_degree={sh_degree}, 特征列={features.shape[1]}", flush=True)
else:
    colors_sh = None
rgb = (0.5 + 0.28209479177387814 * dc).clip(0, 1).astype(np.float32)
if "filter_3D" in v.data.dtype.names:
    filt = np.asarray(v["filter_3D"], np.float32).reshape(-1, 1)
    det1 = np.prod(scales ** 2, axis=1); scales = np.sqrt(scales ** 2 + filt ** 2)
    det2 = np.prod(scales ** 2, axis=1); opac = opac * np.sqrt(det1 / np.maximum(det2, 1e-12))

cam = {}
for line in open(os.path.join(SRC, "sparse", "0", "cameras.txt")):
    if line.startswith("#") or not line.strip():
        continue
    p = line.split(); cam[int(p[0])] = {"w": int(p[2]), "h": int(p[3]), "p": [float(x) for x in p[4:]]}
imgs = []
for line in open(os.path.join(SRC, "sparse", "0", "images.txt")):
    line = line.strip()
    if not line or line.startswith("#") or " " not in line:
        continue
    p = line.split()
    if len(p) < 10:
        continue
    imgs.append((int(p[0]), [float(x) for x in p[1:5]], [float(x) for x in p[5:8]], int(p[8]), p[9]))
imgs.sort()
c0 = cam[list(cam.keys())[0]]; H, W = c0["h"], c0["w"]

def load_img(name):
    for cand in (os.path.join(SRC, name), os.path.join(SRC, "images", os.path.basename(name))):
        if os.path.exists(cand):
            im = np.asarray(Image.open(cand).convert("RGB")).astype(np.float32) / 255.0
            if im.shape[:2] != (H, W):
                im = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((W, H))).astype(np.float32) / 255.0
            return im
    raise FileNotFoundError(name)

def load_mask(name):
    if not MASKS:
        return None
    stem = os.path.splitext(os.path.basename(name))[0]
    for ext in (".png", ".jpg"):
        c = os.path.join(MASKS, stem + ext)
        if os.path.exists(c):
            m = np.asarray(Image.open(c).convert("L"))
            if m.shape[:2] != (H, W):
                m = np.asarray(Image.fromarray(m).resize((W, H)))
            return m > 127
    return None

full, full_sh, fg, msk = [], [], [], []
for iid, qvec, tvec, cid, name in imgs:
    qw, qx, qy, qz = qvec
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]], np.float32)
    R = R.T; t = np.array(tvec, np.float32)
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = -R @ t
    K = cam[cid]["p"]
    Kk = torch.tensor([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], dtype=torch.float32).cuda()[None]
    vmt = torch.from_numpy(vm)[None].cuda()
    gt = load_img(name)
    # f_dc-only（旧口径）
    out = gsplat.rasterization(torch.from_numpy(means).cuda(), torch.from_numpy(quats).cuda(),
                               torch.from_numpy(scales).cuda(), torch.from_numpy(opac).cuda(),
                               torch.from_numpy(rgb).cuda(), vmt, Kk, W, H, render_mode="RGB+D")
    rend = out[0][0, :, :, :3].cpu().numpy()
    a = out[1][0, :, :, 0].cpu().numpy()
    full.append(10 * np.log10(1 / max(((rend - gt) ** 2).mean(), 1e-10)))
    fga = a > 0.5
    fg.append(10 * np.log10(1 / max(((rend - gt)[fga] ** 2).mean(), 1e-10)) if fga.any() else np.nan)
    # 全 SH（正确口径）
    if colors_sh is not None:
        out2 = gsplat.rasterization(torch.from_numpy(means).cuda(), torch.from_numpy(quats).cuda(),
                                    torch.from_numpy(scales).cuda(), torch.from_numpy(opac).cuda(),
                                    colors_sh, vmt, Kk, W, H, sh_degree=sh_degree, render_mode="RGB+D")
        rend2 = out2[0][0, :, :, :3].cpu().numpy()
        full_sh.append(10 * np.log10(1 / max(((rend2 - gt) ** 2).mean(), 1e-10)))
    m = load_mask(name)
    if m is not None:
        msk.append(10 * np.log10(1 / max(((rend - gt)[m] ** 2).mean(), 1e-10)) if m.any() else np.nan)
print(f"[diag] 全图 PSNR(f_dc) mean={np.nanmean(full):.2f}")
if full_sh:
    print(f"[diag] 全图 PSNR(全SH) mean={np.nanmean(full_sh):.2f} (前3: {[round(x,1) for x in full_sh[:3]]})")
print(f"[diag] α前景 PSNR(f_dc) mean={np.nanmean(fg):.2f}")
if msk:
    print(f"[diag] mask内 PSNR mean={np.nanmean(msk):.2f}")
