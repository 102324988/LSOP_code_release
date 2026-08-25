"""convert_camstyle.py: 在 COLMAP 文本稀疏重建里转换相机平移约定。

  center   — colmap_writer 非标准：images.txt 的 tvec 存相机中心 C（合成数据）
  standard — 标准 COLMAP：images.txt 的 tvec 存 t = -R·C（真实 SfM / colmap 输出）

方向：
  center→standard: t_new = -R @ C          （tvec 读作 C）
  standard→center: t_new = C = -R^T @ t    （tvec 读作 t）

用法：python convert_camstyle.py <src_sparse> <dst_sparse> <center|standard>
  src_sparse 是含 images.txt 的目录（通常是 .../sparse/0），dst_sparse 目录会被创建。
  保持 qvec 不变、cameras.txt 不变、points3D 原样拷贝。
"""
import argparse
import os
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src", help="源 sparse 目录（含 images.txt）")
ap.add_argument("dst", help="目标 sparse 目录（会创建）")
ap.add_argument("target", choices=["center", "standard"],
                help="目标约定：center 表示把 tvec 写成相机中心；standard 表示把 tvec 写成 -R·C")
args = ap.parse_args()

os.makedirs(args.dst, exist_ok=True)


def qvec_to_R(q):
    qw, qx, qy, qz = q
    return np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]],
                    dtype=np.float64)


src_style = "standard" if args.target == "center" else "center"
out_lines = []
with open(os.path.join(args.src, "images.txt")) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            out_lines.append(line)
            continue
        p = line.split()
        if len(p) < 10:
            out_lines.append(line)
            continue
        qvec = [float(x) for x in p[1:5]]
        t = np.array([float(x) for x in p[5:8]], np.float64)
        R = qvec_to_R(qvec)
        if src_style == "center":            # 读到的 t 是相机中心 C → 写标准 t=-RC
            t_new = -R @ t
        else:                                # 读到的 t 是 -RC → 写 center C = -R^T t
            t_new = -R.T @ t
        p[5:8] = [f"{v:.9f}" for v in t_new]
        out_lines.append(" ".join(p))

with open(os.path.join(args.dst, "images.txt"), "w") as f:
    f.write("\n".join(out_lines) + "\n")

for name in ("cameras.txt", "points3D.txt"):
    if os.path.exists(os.path.join(args.src, name)):
        shutil.copy(os.path.join(args.src, name), os.path.join(args.dst, name))

print(f"[convert] {args.src} ({src_style}) -> {args.dst} ({args.target})")
