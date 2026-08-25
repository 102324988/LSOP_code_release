#!/usr/bin/env python3
"""dtu_gt_eval.py: combo 反解点云 vs DTU stl GT 点云（同 DTU 帧，免对齐）chamfer 评估。

DTU 帧 = 结构光左相机帧；combo 点云经 ObsMask 过滤的 stl 初始化 + 官方位姿渲染，
与 stl GT 天然同帧 —— 无需 ICP/缩放，直接双向最近邻。

指标：
  cloud2gt p50/mean  每个反解点到最近 GT 点的距离（反解精度）
  gt2cloud mean      每个 GT 点到最近反解点的距离（反解覆盖）
  coverage@t         GT 点被反解点覆盖（距离<t）的比例
  chamfer_mean       双向平均

用法: python dtu_gt_eval.py --cloud <combo.npy> --stl <stlXXX_total.ply> \
      --obs <ObsMaskXXX_10.mat> --out <res.json> [--tag scan1]
"""
import argparse
import json
import os

import numpy as np
import plyfile
import scipy.io as sio
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--cloud", required=True, help="combo 输出的 npy（N,3 或 N,4，最后列可能带距离）")
ap.add_argument("--stl", required=True, help="DTU stl GT 点云 ply")
ap.add_argument("--obs", required=True, help="ObsMask mat（3D 占用网格，过滤 GT 到物体）")
ap.add_argument("--out", required=True, help="输出 json 路径")
ap.add_argument("--tag", default="scan1")
args = ap.parse_args()

# ---- combo 点云 ----
P = np.load(args.cloud)
if P.ndim == 2 and P.shape[1] >= 3:
    cloud = P[:, :3].astype(np.float64)
else:
    raise SystemExit(f"cloud 格式未知: shape={P.shape}")
print(f"[dtu_gt] combo 点云: {len(cloud)}", flush=True)

# ---- stl GT 物体点（ObsMask 3D 占用过滤）----
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
print(f"[dtu_gt] stl 物体点（ObsMask 过滤）: {len(gt)}", flush=True)

if len(cloud) == 0 or len(gt) == 0:
    print(json.dumps({"error": "空点云"}))
    raise SystemExit(1)

# ---- 双向最近邻（同帧免对齐）----
tree_gt = cKDTree(gt)
d_c2g, _ = tree_gt.query(cloud, k=1, workers=-1)         # cloud->gt
tree_c = cKDTree(cloud)
d_g2c, _ = tree_c.query(gt, k=1, workers=-1)             # gt->cloud

res = {
    "tag": args.tag,
    "n_cloud": int(len(cloud)),
    "n_gt": int(len(gt)),
    "cloud2gt_p50": float(np.median(d_c2g)),
    "cloud2gt_mean": float(np.mean(d_c2g)),
    "cloud2gt_p90": float(np.percentile(d_c2g, 90)),
    "gt2cloud_mean": float(np.mean(d_g2c)),
    "coverage_2mm": float(np.mean(d_g2c < 2)),
    "coverage_5mm": float(np.mean(d_g2c < 5)),
    "coverage_10mm": float(np.mean(d_g2c < 10)),
    "chamfer_mean": float((np.mean(d_c2g) + np.mean(d_g2c)) / 2),
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, ensure_ascii=False))
