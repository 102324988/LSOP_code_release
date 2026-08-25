#!/usr/bin/env python3
"""dtu_compare.py: scan1 vs scan6 DTU 真实照片验证对比图。

图 1：cloud→GT 最近邻距离直方图（0-20mm，两场景重叠）+ 图例 p50/p90。
图 2：GT 表面覆盖曲线（gt2cloud@t，0-10mm）+ cov@5mm 标注。
用法: python dtu_compare.py <out_prefix>
数据: output/dtu_scan{1,6}_real2/eval_20000/combo_mask/scan{1,6}.npy
      ~/e0lab/data/dtu_raw/Points/stl/stl0{1,6}_total.ply
      ~/e0lab/data/dtu_raw/SampleSet/MVS Data/ObsMask/ObsMask{1,6}_10.mat
"""
import sys

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

SCANS = {
    "scan1": {"cloud": "output/dtu_scan1_real2/eval_20000/combo_mask/scan1.npy",
              "stl": "/home/ubuntu/e0lab/data/dtu_raw/Points/stl/stl001_total.ply",
              "obs": "/home/ubuntu/e0lab/data/dtu_raw/SampleSet/MVS Data/ObsMask/ObsMask1_10.mat",
              "color": "#4C72B0"},
    "scan6": {"cloud": "output/dtu_scan6_real2/eval_20000/combo_mask/scan6.npy",
              "stl": "/home/ubuntu/e0lab/data/dtu_raw/Points/stl/stl006_total.ply",
              "obs": "/home/ubuntu/e0lab/data/dtu_raw/SampleSet/MVS Data/ObsMask/ObsMask6_10.mat",
              "color": "#C44E52"},
}
OUT = sys.argv[1] if len(sys.argv) > 1 else "output/dtu_compare"

res = {}
for tag, p in SCANS.items():
    C = np.load(p["cloud"])[:, :3].astype(np.float64)
    v = plyfile.PlyData.read(p["stl"])["vertex"]
    Xs = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    obs = sio.loadmat(p["obs"])
    BB, Res, grid = obs["BB"], obs["Res"].item(), obs["ObsMask"]
    mn, mx = BB[0], BB[1]
    inbb = ((Xs >= mn) & (Xs <= mx)).all(1)
    idx = np.floor((Xs[inbb] - mn) / Res).astype(np.int64)
    ok = ((idx[:, 0] < grid.shape[2]) & (idx[:, 1] < grid.shape[1]) & (idx[:, 2] < grid.shape[0])
          & (idx >= 0).all(1))
    occ = grid[idx[ok][:, 2], idx[ok][:, 1], idx[ok][:, 0]] > 0
    gt = Xs[inbb][ok][occ]
    tree_gt = cKDTree(gt)
    d_c2g, _ = tree_gt.query(C, k=1, workers=-1)
    tree_c = cKDTree(C)
    d_g2c, _ = tree_c.query(gt[::8], k=1, workers=-1)
    res[tag] = {"d_c2g": d_c2g, "d_g2c": d_g2c, "color": p["color"],
                "p50": float(np.median(d_c2g)), "p90": float(np.percentile(d_c2g, 90)),
                "cov5": float((d_g2c < 5).mean())}
    print(f"[cmp] {tag}: cloud={len(C)} gt={len(gt)} d_c2g p50={res[tag]['p50']:.2f} "
          f"p90={res[tag]['p90']:.1f} cov@5mm={res[tag]['cov5']:.3f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"wspace": 0.25})

# 左：cloud→GT 距离直方图
ax = axes[0]
for tag, r in res.items():
    ax.hist(np.clip(r["d_c2g"], 0, 20), bins=60, alpha=0.62, color=r["color"],
            edgecolor="white", linewidth=0.25,
            label=f"{tag}  p50={r['p50']:.1f}mm / p90={r['p90']:.0f}mm")
ax.axvline(5, color="#8C6DAF", ls="--", lw=1.2)
ax.text(5.15, ax.get_ylim()[1] * 0.97, "5mm", fontsize=8, color="#8C6DAF", **CJK)
ax.set_xlabel("cloud→GT 最近邻距离 (mm)", fontsize=10, **CJK)
ax.set_ylabel("点计数", fontsize=10, **CJK)
ax.set_title("组合判据点云到 GT 表面距离分布", fontsize=10, **CJK)
ax.legend(fontsize=8.5)
ax.set_xlim(0, 20)
ax.grid(alpha=0.25, ls=":")

# 右：GT 覆盖曲线
ax = axes[1]
tt = np.linspace(0, 10, 120)
for tag, r in res.items():
    ax.plot(tt, [(r["d_g2c"] < t).mean() for t in tt], color=r["color"], lw=2,
            label=f"{tag}  cov@5mm={r['cov5']*100:.1f}%")
ax.axvline(5, color="#8C6DAF", ls="--", lw=1.2)
ax.set_xlabel("距离阈值 t (mm)", fontsize=10, **CJK)
ax.set_ylabel("GT 表面被覆盖比例", fontsize=10, **CJK)
ax.set_title("GT 表面覆盖曲线（真实照片验证）", fontsize=10, **CJK)
ax.legend(fontsize=8.5)
ax.set_ylim(0, 1.01)
ax.grid(alpha=0.25, ls=":")

fig.tight_layout()
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"[cmp] saved {OUT}.png", flush=True)
