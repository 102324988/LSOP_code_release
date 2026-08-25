"""gen_masks.py: 由 GT mesh + 相机位姿生成前景 mask（eval_no_gt silhouette 路径用）。

原理：mask 只需在 eval_no_gt 的 step 采样网格像素上精确（回投影采样点正好落在
step 网格），故在 step 网格上用 trimesh 光线求交 GT mesh，再上采样为全分辨率 PNG。
相机约定与 eval_no_gt 的 --cam_style 一致（默认 gof）。

用法：python gen_masks.py --source <colmap> --mesh <gt.ply> --out <mask_dir>
      [--cam_style gof] [--step 12]
"""
import argparse
import os

import numpy as np
import plyfile
import trimesh
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cam_style", default="gof", choices=["center", "standard", "gof"])
ap.add_argument("--step", type=int, default=12)
args = ap.parse_args()

mesh = trimesh.load(args.mesh, force="mesh")
print(f"[masks] mesh: {len(mesh.faces)} faces, bbox 对角={mesh.bounding_box.extents.max():.3f}", flush=True)

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

os.makedirs(args.out, exist_ok=True)
ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh) if hasattr(trimesh.ray, "ray_triangle") else trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

for gi, (iid, qvec, tvec, cid, name) in enumerate(imgs):
    qw, qx, qy, qz = qvec
    R = np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]],
                 dtype=np.float32)
    t = np.array(tvec, dtype=np.float32)
    if args.cam_style == "gof":
        R = R.T
        center = t                                 # GOF camera_center=Tm=tvec（光线起点！）
        Rc = R
    elif args.cam_style == "center":
        center = t
        Rc = R
    else:                       # standard: 文件 R 即 world->cam
        center = -R.T @ t
        Rc = R
    c = cam[cid]
    model, p = c["model"], c["p"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[:4]
    H, W = c["h"], c["w"]
    ncols = int(np.ceil(W / args.step)); nrows = int(np.ceil(H / args.step))
    u = np.arange(0, W, args.step, dtype=np.float32) + args.step / 2
    vv = np.arange(0, H, args.step, dtype=np.float32) + args.step / 2
    V, U = np.meshgrid(vv, u, indexing="ij")
    Uf, Vf = U.ravel(), V.ravel()
    # 相机系光线方向 -> 世界系
    d_cam = np.stack([(Uf - cx) / fx, (Vf - cy) / fy, np.ones(len(Uf))], 1)
    d_w = d_cam @ Rc                            # 行向量: 相机方向→世界 = d_cam @ Rc
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    origins = np.broadcast_to(center[None, :], (len(Uf), 3))
    hit = ray.intersects_any(origins.astype(np.float64), d_w.astype(np.float64))
    # 上采样到全分辨率
    small = (hit.reshape(nrows, ncols).astype(np.uint8) * 255)
    full = np.asarray(Image.fromarray(small).resize((W, H), Image.NEAREST))
    stem = os.path.splitext(os.path.basename(name))[0]
    Image.fromarray(full).save(os.path.join(args.out, stem + ".png"))
    if (gi + 1) % 8 == 0:
        print(f"  {gi+1}/{len(imgs)} done, hit_frac={hit.mean():.3f}", flush=True)

print(f"[masks] {len(imgs)} masks -> {args.out}")
