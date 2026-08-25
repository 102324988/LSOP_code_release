"""E0: combined-criterion per-pixel surface inversion (v2).

Per pixel choose:
  - rendered depth (gsplat "D", camera-z) when accumulated alpha >= t_high
    (high-confidence accuracy backbone);
  - ray-nearest first-visible gaussian otherwise (coverage fill for grazing /
    low-opacity pixels where depth has no trustworthy value).

Key difference from v1: ray-nearest is computed on ALL pixels (coverage
backbone), depth merely REPLACES it on high-alpha pixels. v1 computed ray-nearest
only on low-alpha pixels, but those are exactly where ray-nearest hits poorly
(grazing) — coverage collapsed.
"""
import argparse
import json
import os

import numpy as np
import torch
import trimesh
import plyfile
from PIL import Image
from scipy.spatial import cKDTree

import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--source", required=True, help="COLMAP dataset dir (data/X_rb)")
ap.add_argument("--obj", required=True)
ap.add_argument("--dumps", default="combo_6obj")
ap.add_argument("--step", type=int, default=12)
ap.add_argument("--t_high", type=float, default=0.50,
                help="accumulated-alpha threshold: above -> rendered depth (GOF 正确约定下 0.5 为合理默认)")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--r_ref", type=float, default=0.25)
ap.add_argument("--t_tol", type=float, default=0.15)
ap.add_argument("--alpha_min", type=float, default=0.05)
ap.add_argument("--t_near", type=float, default=0.02)
ap.add_argument("--t_far", type=float, default=6.0)
ap.add_argument("--n_ref", type=int, default=40)
ap.add_argument("--no_filter", action="store_true",
                help="不施加 GOF 3D filter（诊断/对比用）")
ap.add_argument("--cam_style", default="gof", choices=["center", "standard", "gof"],
                help="center: colmap_writer 非标准约定（tvec=相机中心 C，合成数据）；"
                     "standard: 标准 COLMAP（t=-R·C，真实 SfM）；"
                     "gof: GOF 训练内部约定 R=qvec2rotmat(q).T、W2C=[[R,-R@tvec]]（忠实复现 GOF 渲染，默认）。")
ap.add_argument("--no_gt", action="store_true",
                help="无 GT 网格模式（真实照片）：跳过 cloud2gt/coverage/chamfer 指标，"
                     "仍输出点云 npy + 几何卫生统计")
ap.add_argument("--auto_scale", action="store_true",
                help="场景尺度自适应：以相机到质心距离 d_ref 与高斯中位尺度为基准缩放 "
                     "r_ref/t_tol/t_near/t_far（真实照片尺度未知时用，合成归一化数据无需）")
ap.add_argument("--mask_dir", default=None,
                help="前景 mask 目录（按图像 basename 匹配）：只在这些像素反解，"
                     "输出纯物体表面点云（DTU 用 ObsMask 投影 mask）")
args = ap.parse_args()

# ---------------- gaussians ----------------
v = plyfile.PlyData.read(args.ply)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opacities = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
n = len(means)
print(f"[combo] {n} gaussians", flush=True)

# ---- GOF 3D filter（训练损失在滤波后渲染上计算，评估必须一致；旧 PLY 无该列则退化为无滤波）----
if "filter_3D" in v.data.dtype.names and not args.no_filter:
    filt = np.asarray(v["filter_3D"], dtype=np.float32).reshape(-1, 1)   # [N,1]
    det1 = np.prod(scales ** 2, axis=1)                                  # prod(raw²)
    scales = np.sqrt(scales ** 2 + filt ** 2)                            # scales² + f² 再开方
    det2 = np.prod(scales ** 2, axis=1)                                  # prod(raw² + f²)
    coef = np.sqrt(det1 / np.maximum(det2, 1e-12))                       # [N] 逐高斯衰减
    opacities = opacities * coef
    print(f"[combo] 施加 GOF 3D filter: filter 中位={np.median(filt):.4f}  "
          f"opacity 衰减 coef 中位={np.median(coef):.4f}", flush=True)

colors = torch.zeros((n, 3), device="cuda")
means_t = torch.from_numpy(means).cuda()
scales_t = torch.from_numpy(scales).cuda()
quats_t = torch.from_numpy(quats).cuda()
opac_t = torch.from_numpy(opacities).cuda()
alpha_t = torch.from_numpy(opacities).cuda()
kd = cKDTree(means.astype(np.float64))

# ---------------- cameras ----------------
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
viewmats, Ks, centers = [], [], []
for iid, qvec, tvec, cid, name in imgs:
    qw, qx, qy, qz = qvec
    R = np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]],
                 dtype=np.float32)
    t = np.array(tvec, dtype=np.float32)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R                                 # world->cam rotation
    if args.cam_style == "gof":                    # GOF 帧：R=qvec2rotmat(q).T, W2C=[[R,-R@tvec]], 相机中心=tvec
        R = R.T
        vm[:3, :3] = R
        center = t                                 # GOF camera_center=Tm=tvec（光线起点！）
        vm[:3, 3] = -R @ t                         # W2C 平移 = -Rc@C
    elif args.cam_style == "center":               # colmap_writer 非标准：tvec 存相机中心 C
        center = t
        vm[:3, 3] = -R @ t                         # W2C translation = -R@C
    else:                                          # 标准 COLMAP：tvec = -R·C
        center = -R.T @ t
        vm[:3, 3] = t
    c = cam[cid]
    model, p = c["model"], c["p"]
    if model == "SIMPLE_PINHOLE":                  # f cx cy
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif model == "SIMPLE_RADIAL":                 # f cx cy k
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:                                          # PINHOLE / OPENCV / OPENCV_FISHEYE / ... : fx fy cx cy [畸变]
        fx, fy, cx, cy = p[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    viewmats.append(vm)
    Ks.append(K)
    centers.append(center)
viewmats_t = torch.from_numpy(np.stack(viewmats)).cuda()
Ks_t = torch.from_numpy(np.stack(Ks)).cuda()
any_cam = cam[list(cam.keys())[0]]
H, W = any_cam["h"], any_cam["w"]
origins = np.stack(centers, 0).astype(np.float32)
Rs = np.stack([vm[:3, :3] for vm in viewmats], 0).astype(np.float32)

# pixel grid
u = np.arange(0, W, args.step, dtype=np.float32) + args.step / 2
vv = np.arange(0, H, args.step, dtype=np.float32) + args.step / 2
V, U = np.meshgrid(vv, u, indexing="ij")
Uf = U.reshape(-1).astype(np.float32)
Vf = V.reshape(-1).astype(np.float32)
PIX = Uf.shape[0]

# ---- 尺度自适应（真实照片：几何参数按场景尺度缩放）----
if args.auto_scale:
    centroid = means.mean(0)
    d_ref = float(np.median(np.linalg.norm(origins - centroid, axis=1)))
    med_scale = float(np.median(np.linalg.norm(scales, axis=1)))
    args.r_ref = max(0.08 * d_ref, 8 * med_scale)
    args.t_tol = max(0.05 * d_ref, 3 * med_scale)
    args.t_near = 0.01 * d_ref
    args.t_far = 2.0 * d_ref
    print(f"[combo] auto_scale: d_ref={d_ref:.1f} med_scale={med_scale:.3f} -> "
          f"r_ref={args.r_ref:.2f} t_tol={args.t_tol:.2f} "
          f"t_near={args.t_near:.3f} t_far={args.t_far:.1f}", flush=True)

t_ref = np.linspace(args.t_near, args.t_far, args.n_ref)
print(f"[combo] {len(viewmats)} views x {PIX} px", flush=True)

pts_all = []
n_depth = n_rayne = 0
for vi in range(len(viewmats)):
    # ---- render depth + alpha ----
    out = gsplat.rasterization(
        means_t, quats_t, scales_t, opac_t, colors,
        viewmats_t[vi:vi + 1], Ks_t[vi:vi + 1], W, H, render_mode="D")
    Dz = out[0][0, ::args.step, ::args.step, 0].ravel().cpu().numpy()     # (PIX,)
    Av = out[1][0, ::args.step, ::args.step, 0].ravel().cpu().numpy()
    Kb = Ks[vi]
    fx, fy, cx, cy = Kb[0, 0], Kb[1, 1], Kb[0, 2], Kb[1, 2]

    # ---- mask 采样：只在该视角前景像素上反解（--mask_dir 时）----
    Msk = None
    if args.mask_dir:
        stem = os.path.splitext(os.path.basename(imgs[vi][4]))[0]
        cand = os.path.join(args.mask_dir, stem + ".png")
        if os.path.exists(cand):
            m = np.asarray(Image.open(cand).convert("L")) > 127
            Msk = m[np.clip(Vf, 0, H - 1).astype(np.int64),
                    np.clip(Uf, 0, W - 1).astype(np.int64)]     # (PIX,) bool

    # ---- ray-nearest on ALL pixels (coverage backbone) ----
    o_np = origins[vi][None, :]
    d_cam = np.stack([(Uf - cx) / fx, (Vf - cy) / fy, np.ones(PIX)], 1)
    d_w = d_cam @ Rs[vi]                         # 行向量: 相机方向→世界 = d_cam @ Rc（Rc=vm[:3,:3]）
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    refs = o_np + d_w[:, None, :] * t_ref[None, :, None]
    dist, idx = kd.query(refs.reshape(-1, 3), k=args.k, workers=-1,
                         distance_upper_bound=args.r_ref)
    gi = torch.from_numpy(idx.astype(np.int64)).cuda().reshape(PIX, -1)
    valid = (gi < n)
    gi_safe = gi.clamp(max=n - 1)
    p = means_t[gi_safe]
    o_c = torch.from_numpy(o_np).cuda()                       # (1,3)
    d_c = torch.from_numpy(d_w.astype(np.float32)).cuda()
    tvec = ((p - o_c) @ d_c[:, :, None])[..., 0]              # (PIX,n_ref*k)
    pdist2 = ((p - o_c) ** 2).sum(-1) - tvec ** 2
    on_ray = (tvec > args.t_near) & (pdist2 < args.t_tol ** 2) & valid
    ok_r = on_ray & (alpha_t[gi_safe] >= args.alpha_min)
    t_sel = torch.where(ok_r, tvec, torch.full_like(tvec, float("inf")))
    t_min = t_sel.min(dim=1).values.cpu().numpy()             # (PIX,)
    has_rayne = np.isfinite(t_min) & (t_min > 0)
    hit_pt = origins[vi][None, :] + d_w * t_min[:, None]      # (PIX,3)

    # ---- combine per pixel ----
    depth_ok = (Av >= args.t_high) & np.isfinite(Dz) & (Dz > 0) & (Dz < args.t_far)
    use_depth = depth_ok
    use_rayne = (~depth_ok) & has_rayne
    if Msk is not None:
        use_depth = use_depth & Msk
        use_rayne = use_rayne & Msk

    if use_depth.any():
        Xn = (Uf[use_depth] - cx) / fx
        Yn = (Vf[use_depth] - cy) / fy
        z = Dz[use_depth]
        pc = np.stack([Xn * z, Yn * z, z], 1)
        pc_h = np.concatenate([pc, np.ones((len(pc), 1))], 1)
        pw = (np.linalg.inv(viewmats[vi]) @ pc_h.T).T[:, :3]
        pts_all.append(pw)
        n_depth += len(pw)
    if use_rayne.any():
        hv = hit_pt[use_rayne]
        mm = np.linalg.norm(hv, axis=1) < 1.2
        if mm.any():
            pts_all.append(hv[mm])
            n_rayne += int(mm.sum())
    if (vi + 1) % 8 == 0:
        print(f"  view {vi+1}/{len(viewmats)}, pts {sum(len(p) for p in pts_all)}",
              flush=True)

pts = np.vstack(pts_all) if pts_all else np.zeros((0, 3))
print(f"[combo] {args.obj}: total pts={len(pts)} (depth {n_depth} / rayne {n_rayne})")

# ---- GT 网格对比（合成数据有；真实照片没有 → 优雅降级为几何卫生统计）----
gt_path = os.path.join(args.source, "..", "meshes", args.obj + ".ply")
has_gt = False
d_cloud2gt = d_gt2cloud = None
if not args.no_gt:
    try:
        gt = trimesh.load(gt_path, force="mesh")
        has_gt = True
        print(f"[combo] GT mesh: {gt_path} ({len(gt.faces)} faces)", flush=True)
    except Exception as e:
        print(f"[combo] ! 无 GT 网格（{gt_path}）：{type(e).__name__}，"
              f"跳过 cloud2gt/coverage/chamfer，输出几何卫生统计", flush=True)
        has_gt = False
if has_gt:
    _, d_cloud2gt, _ = gt.nearest.on_surface(pts.astype(np.float64)) if len(pts) \
        else (None, np.array([]), None)
    samp = gt.sample(30000)
    d_gt2cloud = cKDTree(pts.astype(np.float64)).query(samp)[0] if len(pts) else np.full(30000, np.inf)
r = np.linalg.norm(pts, axis=1)
res = dict(
    obj=args.obj, has_gt=has_gt, n_pts=int(len(pts)), n_depth=int(n_depth), n_rayne=int(n_rayne),
    cloud2gt_p50=float(np.median(d_cloud2gt)) if (has_gt and len(pts)) else None,
    cloud2gt_mean=float(np.mean(d_cloud2gt)) if (has_gt and len(pts)) else None,
    gt2cloud_mean=float(np.mean(d_gt2cloud)) if has_gt else None,
    coverage_01=float(np.mean(d_gt2cloud < 0.1)) if has_gt else None,
    coverage_02=float(np.mean(d_gt2cloud < 0.2)) if has_gt else None,
    chamfer_mean=float((np.mean(d_cloud2gt) + np.mean(d_gt2cloud)) / 2) if (has_gt and len(pts)) else None,
    radius_p50=float(np.percentile(r, 50)) if len(pts) else None,
    radius_p90=float(np.percentile(r, 90)) if len(pts) else None,
)
os.makedirs(args.dumps, exist_ok=True)
np.save(os.path.join(args.dumps, f"{args.obj}.npy"),
        np.hstack([pts, d_cloud2gt[:, None]]) if (has_gt and len(pts)) else pts)
with open(os.path.join(args.dumps, f"{args.obj}.res.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, ensure_ascii=False))
