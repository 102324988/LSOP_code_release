#!/usr/bin/env python3
"""dtu_viz.py: DTU 固定前段结果可视化。

图 1：combo 物体表面点云按到 GT 表面距离着色（3D scatter，多种采样）+ cloud2gt 距离直方图。
图 2（可选 --compare_json）：合成 §5.7 vs DTU 归一化指标对比表（PNG）。

用法: python dtu_viz.py --cloud <combo_mask.npy> --stl <stl> --obs <obs.mat> \
      --out <out_prefix> [--tag scan1_20000]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plyfile
import scipy.io as sio
from scipy.spatial import cKDTree

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False
CJK = {"fontfamily": "Noto Sans CJK JP"}

ap = argparse.ArgumentParser()
ap.add_argument("--cloud", required=True)
ap.add_argument("--stl", required=True)
ap.add_argument("--obs", required=True)
ap.add_argument("--out", required=True, help="输出 PNG 前缀")
ap.add_argument("--tag", default="scan1")
args = ap.parse_args()

# ---- 点云与 GT ----
cloud = np.load(args.cloud)[:, :3].astype(np.float64)
v = plyfile.PlyData.read(args.stl)["vertex"]
Xs = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
obs = sio.loadmat(args.obs)
BB, Res, grid = obs["BB"], obs["Res"].item(), obs["ObsMask"]
mn, mx = BB[0], BB[1]
inbb = ((Xs >= mn) & (Xs <= mx)).all(1)
idx = np.floor((Xs[inbb] - mn) / Res).astype(np.int64)
ok = ((idx[:, 0] < grid.shape[2]) & (idx[:, 1] < grid.shape[1]) & (idx[:, 2] < grid.shape[0])
      & (idx >= 0).all(1))
occ = grid[idx[ok][:, 2], idx[ok][:, 1], idx[ok][:, 0]] > 0
gt = Xs[inbb][ok][occ]
tree_gt = cKDTree(gt)
d_c2g, _ = tree_gt.query(cloud, k=1, workers=-1)
print(f"[viz] cloud={len(cloud)} gt={len(gt)} d_c2g p50={np.median(d_c2g):.2f} "
      f"mean={np.mean(d_c2g):.2f} p90={np.percentile(d_c2g, 90):.2f} (mm)", flush=True)

# ---- 图 1：点云按距离着色 + 直方图 ----
rng = np.random.default_rng(0)
n_max = 40000
sel = rng.choice(len(cloud), min(n_max, len(cloud)), replace=False)
cs, ds = cloud[sel], d_c2g[sel]

fig = plt.figure(figsize=(13, 5.2))
# 左：3D scatter
ax = fig.add_subplot(1, 2, 1, projection="3d")
sc = ax.scatter(cs[:, 0], cs[:, 1], cs[:, 2], c=ds, cmap="magma_r", s=0.6, alpha=0.7)
ax.set_xlabel("x (mm)", fontsize=9, labelpad=1)
ax.set_ylabel("y (mm)", fontsize=9, labelpad=1)
ax.set_zlabel("z (mm)", fontsize=9, labelpad=1)
ax.set_title(f"组合判据反解表面点云（{args.tag}）按到 GT 距离着色",
             fontsize=10, **CJK)
cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.05)
cb.set_label("到 GT 表面距离 (mm)", fontsize=9, **CJK)
ax.view_init(elev=18, azim=-55)
ax.set_box_aspect((np.ptp(cs[:, 0]), np.ptp(cs[:, 1]), np.ptp(cs[:, 2])))
ax.tick_params(labelsize=7)

# 右：直方图
ax2 = fig.add_subplot(1, 2, 2)
ax2.hist(np.clip(d_c2g, 0, 40), bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
for m, c, lab in [(np.median(d_c2g), "#C44E52", f"p50={np.median(d_c2g):.1f}mm"),
                  (np.mean(d_c2g), "#DD8452", f"mean={np.mean(d_c2g):.1f}mm"),
                  (np.percentile(d_c2g, 90), "#8C6DAF", f"p90={np.percentile(d_c2g, 90):.1f}mm")]:
    ax2.axvline(m, color=c, lw=1.4, ls="--", label=lab)
ax2.set_xlabel("cloud→GT 最近邻距离 (mm)", fontsize=10, **CJK)
ax2.set_ylabel("点计数", fontsize=10, **CJK)
ax2.set_title("cloud→GT 距离分布（壳厚偏移）", fontsize=10, **CJK)
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=8)
ax2.set_xlim(0, 40)
ax2.grid(alpha=0.25, ls=":")
fig.tight_layout()
fig.savefig(args.out + "_cloud_dist.png", dpi=150, bbox_inches="tight")
print(f"[viz] saved {args.out}_cloud_dist.png", flush=True)

# ---- 图 2：GT 点被覆盖的比例随距离阈值 ----
fig2, ax3 = plt.subplots(figsize=(6.5, 4))
tt = np.linspace(0, 20, 200)
tree_c = cKDTree(cloud)
d_g2c, _ = tree_c.query(gt[::8], k=1, workers=-1)
ax3.plot(tt, [(d_g2c < t).mean() for t in tt], color="#4C72B0", lw=2)
ax3.axvline(2, color="#C44E52", ls="--", lw=1.2, label="t=2mm")
ax3.axvline(5, color="#DD8452", ls="--", lw=1.2, label="t=5mm")
ax3.set_xlabel("距离阈值 t (mm)", fontsize=10, **CJK)
ax3.set_ylabel("GT 表面被覆盖比例", fontsize=10, **CJK)
ax3.set_title(f"GT 表面覆盖曲线（{args.tag}）", fontsize=10, **CJK)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.25, ls=":")
ax3.set_ylim(0, 1.01)
fig2.tight_layout()
fig2.savefig(args.out + "_coverage.png", dpi=150, bbox_inches="tight")
print(f"[viz] saved {args.out}_coverage.png", flush=True)
