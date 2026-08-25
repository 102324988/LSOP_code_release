"""diag_rgb.py: 诊断渲染 RGB 保真度 —— 低 PSNR 是"阈值过严"还是"RGB 渲染错误"。

对每个视角:按 alpha 分箱统计 |render-GT| 颜色误差、不同 alpha 阈值下的 PSNR,
输出 fg 像素的空间分布特征。回答:GOF 低不透明度场下,渲染 RGB 是否可信。
用法:python diag_rgb.py --ply <ply> --source <colmap_dir> [--cam_style center]
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
ap.add_argument("--cam_style", default="center", choices=["center", "standard"])
ap.add_argument("--views", default="0-48")
args = ap.parse_args()

v = plyfile.PlyData.read(args.ply)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opac = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
rgb = (0.5 + 0.28209479177387814 * dc).clip(0, 1).astype(np.float32)

means_t = torch.from_numpy(means).cuda()
scales_t = torch.from_numpy(scales).cuda()
quats_t = torch.from_numpy(quats).cuda()
opac_t = torch.from_numpy(opac).cuda()
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

viewmats, Ks, names = [], [], []
for iid, qvec, tvec, cid, name in imgs:
    qw, qx, qy, qz = qvec
    R = np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]],
                 dtype=np.float32)
    t = np.array(tvec, dtype=np.float32)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R
    if args.cam_style == "center":
        vm[:3, 3] = -R @ t
    else:
        vm[:3, 3] = t
    c = cam[cid]
    model, p = c["model"], c["p"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    viewmats.append(vm); Ks.append(K); names.append(name)
viewmats_np = np.stack(viewmats).astype(np.float32)
Ks_np = np.stack(Ks).astype(np.float32)
H, W = cam[list(cam.keys())[0]]["h"], cam[list(cam.keys())[0]]["w"]
N = len(viewmats)


def load_img(name, H, W):
    base = os.path.basename(name)
    candidates = [os.path.join(args.source, name),
                  os.path.join(args.source, "images", name),
                  os.path.join(args.source, "images", base)]
    path = next((c for c in candidates if os.path.exists(c)), None)
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    if im.shape[:2] != (H, W):
        im = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((W, H))
                        ).astype(np.float32) / 255.0
    return im


bins = [0.0, 0.1, 0.3, 0.5, 0.7, 1.01]
bin_err = {k: [] for k in range(len(bins) - 1)}     # 每 alpha 箱的 mean abs err
thr_psnr = {t: [] for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)}
fg_areas = []
lo, hi = (int(x) for x in args.views.split("-"))

for b0 in range(0, N, 8):
    vm = torch.from_numpy(viewmats_np[b0:b0 + 8]).cuda()
    Kk = torch.from_numpy(Ks_np[b0:b0 + 8]).cuda()
    out = gsplat.rasterization(means_t, quats_t, scales_t, opac_t, rgb_t, vm, Kk, W, H,
                               render_mode="RGB+D")
    rc = out[0].cpu().numpy()
    ra = out[1].cpu().numpy()
    for bi in range(rc.shape[0]):
        gi = b0 + bi
        if not (lo <= gi < hi):
            continue
        r, a = rc[bi, :, :, :3], ra[bi, :, :, 0]
        gt = load_img(names[gi], H, W)
        err = np.abs(r - gt).mean(2)                      # per-pixel mean abs err
        for k in range(len(bins) - 1):
            m = (a >= bins[k]) & (a < bins[k + 1])
            if m.sum() > 50:
                bin_err[k].append(float(err[m].mean()))
        for t in thr_psnr:
            fg = a > t
            if fg.sum() >= 50:
                mse = float(((r[fg] - gt[fg]) ** 2).mean())
                thr_psnr[t].append(10 * np.log10(1.0 / max(mse, 1e-10)))
        fg50 = a > 0.5
        if fg50.sum() >= 50:
            ys, xs = np.where(fg50)
            fg_areas.append((fg50.sum(), xs.max() - xs.min(), ys.max() - ys.min()))
    print(f"  {min(b0 + 8, N)}/{N}", flush=True)

print("\n==== 每 alpha 箱 平均绝对颜色误差 ====")
for k in range(len(bins) - 1):
    if bin_err[k]:
        print(f"  alpha in [{bins[k]}, {bins[k+1]:.2f}): mean_abs_err = "
              f"{np.mean(bin_err[k]):.4f}  (n_views={len(bin_err[k])})")

print("\n==== 不同 alpha 阈值下 PSNR ====")
for t in thr_psnr:
    vals = thr_psnr[t]
    if vals:
        print(f"  alpha > {t}: PSNR = {np.mean(vals):6.2f}  (n_views={len(vals)})")
    else:
        print(f"  alpha > {t}: 无足够 fg 像素")

if fg_areas:
    px = [a[0] for a in fg_areas]; w_ = [a[1] for a in fg_areas]; h_ = [a[2] for a in fg_areas]
    print(f"\n==== fg(alpha>0.5) 空间特征 (n_views={len(fg_areas)}) ====")
    print(f"  像素数 mean={np.mean(px):.0f}  bbox宽 mean={np.mean(w_):.0f}px  高 mean={np.mean(h_):.0f}px")
    print(f"  图像 {W}x{H}, fg 覆盖 {np.mean(px)/(W*H)*100:.1f}%")
