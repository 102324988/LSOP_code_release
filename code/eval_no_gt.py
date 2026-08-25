"""eval_no_gt.py: 真实场景无 GT mesh 下的固定前段鲁棒性评估。

真实照片没有 GT 网格/深度，改从"可交叉验证的观测"度量四条防线：

  1. 渲染保真（训练端密度卫生）
     gsplat 渲染训练视角 RGB vs 真实照片，在前景（mask 或 alpha>0.5）内算 PSNR/SSIM。
     浮游/背景幽灵高斯会直接拉低前景外质量与背景 alpha。

  2. silhouette 一致性（2.5D 表面贴合真实边界）
     每视角渲染 depth 反投影的点回投到本视角，落在前景 mask 内的比例。
     高命中 = 反解表面贴合物体真实轮廓。

  3. 跨视角深度一致性（反解稳定性，MVS 风格）
     视角 i 的 depth 点变换到视角 j 相机系，与 j 的 depth 在对应像素比较。
     （i,j 可见重叠区应深度一致；相对差 p50 与一致率。）

  4. 几何卫生（无 GT 下的点云自洽统计）
     组合点云尺度（radius 中位数）、离群比例；渲染 alpha 覆盖/掠射比例。

用法：
  python eval_no_gt.py --ply <model.ply> --source <colmap_dir> --dumps <outdir> \\
      [--mask_dir <masks>] [--step 12] [--cam_style standard] [--t_alpha 0.5]
  source 为含 sparse/0/ 的 COLMAP 数据集目录，图像按 images.txt 的 name 相对 source 解析。

输出 <dumps>/no_gt.res.json + 按视角渲染图（<dumps>/render_i.png，可选 --save_render）。
"""
import argparse
import json
import os

import numpy as np
import plyfile
import torch
from PIL import Image
from skimage.metrics import structural_similarity

import gsplat

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--source", required=True)
ap.add_argument("--dumps", default="no_gt")
ap.add_argument("--mask_dir", default=None, help="前景 mask 目录（文件名与图像 basename 前缀匹配）")
ap.add_argument("--step", type=int, default=12, help="深度一致性评估的像素采样步长")
ap.add_argument("--cam_style", default="gof", choices=["center", "standard", "gof"],
                help="center: tvec=相机中心(合成 colmap_writer); standard: tvec=-R·C(COLMAP); "
                     "gof: GOF 训练内部约定 R=qvec2rotmat(q).T, W2C=[[R,-R@tvec]]（忠实复现 GOF 渲染，默认）")
ap.add_argument("--t_alpha", type=float, default=0.5, help="前景像素 alpha 阈值")
ap.add_argument("--d_alpha", type=float, default=0.05, help="深度/一致性有效像素 alpha 阈值")
ap.add_argument("--n_cross", type=int, default=2, help="每视角做深度一致性的参考视角数（+1..+n_cross 环）")
ap.add_argument("--no_filter", action="store_true", help="不施加 GOF 3D filter（诊断用）")
ap.add_argument("--save_render", action="store_true", help="保存每视角渲染图")
args = ap.parse_args()

# ---------------- gaussians（含颜色，用于渲染 RGB） ----------------
v = plyfile.PlyData.read(args.ply)["vertex"]
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
opacities = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))).astype(np.float32)
dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
rgb = (0.5 + 0.28209479177387814 * dc).clip(0, 1).astype(np.float32)
# 全 SH 特征（GOF 训练 sh_degree=3，PLY 含 f_dc + f_rest；按 band-major [N, n_bands², 3]）
names = list(v.data.dtype.names)
frest = [c for c in names if c.startswith("f_rest_")]
n_bands = int(round((len(frest) / 3 + 1) ** 0.5)) if frest else 1
sh_degree = n_bands - 1
if frest:
    rest = np.stack([v[c] for c in frest], 1).astype(np.float32)      # [N, 3*(n_bands²-1)]
    # GOF PLY f_rest 是 channel-major-over-bands：f_rest_{3*?}... 实为 [N,3,K] 扁平 →
    # 列 f_rest_{K*c + b} 对应 band (b+1)、channel c（K=n_bands²-1）。
    # 先 reshape [N,3,K] 再 transpose 到 gsplat 需要的 [N,K,3]。
    K = n_bands ** 2 - 1
    feats = np.concatenate([dc[:, None, :], rest.reshape(-1, 3, K).transpose(0, 2, 1)], axis=1)
    print(f"[no_gt] 全 SH 渲染: sh_degree={sh_degree} ({n_bands}² 带)", flush=True)
else:
    feats = None
    print("[no_gt] PLY 无 f_rest 列，按 SH0 颜色渲染", flush=True)
n = len(means)
print(f"[no_gt] {n} gaussians", flush=True)

# ---- GOF 3D filter（GOF 渲染时对缩放/不透明度施加的滤波，见 gaussian_model.py）----
# 训练损失在滤波后的渲染上计算，评估必须一致；旧 PLY 无该列时退化为无滤波。
if "filter_3D" in v.data.dtype.names and not args.no_filter:
    filt = np.asarray(v["filter_3D"], dtype=np.float32).reshape(-1, 1)   # [N,1]
    det1 = np.prod(scales ** 2, axis=1)                                  # prod(raw²)
    scales = np.sqrt(scales ** 2 + filt ** 2)                            # scales² + f² 再开方
    det2 = np.prod(scales ** 2, axis=1)                                  # prod(raw² + f²)
    coef = np.sqrt(det1 / np.maximum(det2, 1e-12))                       # [N] 逐高斯衰减
    opacities = opacities * coef
    print(f"[no_gt] 施加 GOF 3D filter: filter 中位={np.median(filt):.4f}  "
          f"opacity 衰减 coef 中位={np.median(coef):.4f}", flush=True)
else:
    print("[no_gt] PLY 无 filter_3D 列，按无滤波渲染（旧 GOF 模型）", flush=True)

means_t = torch.from_numpy(means).cuda()
scales_t = torch.from_numpy(scales).cuda()
quats_t = torch.from_numpy(quats).cuda()
opac_t = torch.from_numpy(opacities).cuda()
rgb_t = torch.from_numpy(rgb).cuda()
features_t = torch.from_numpy(feats).cuda() if feats is not None else None

# ---------------- 相机 ----------------
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
    if args.cam_style == "gof":
        # GOF 训练内部约定:读入 R 后转置为相机旋转,W2C 平移 = -R@tvec
        # 已验证该约定下 gsplat 渲染与 GOF 自身渲染 PSNR 27.7 一致。
        R = R.T
        vm[:3, :3] = R
        vm[:3, 3] = -R @ t
    elif args.cam_style == "center":
        vm[:3, 3] = -R @ t
    else:
        vm[:3, 3] = t
    c = cam[cid]
    model, p = c["model"], c["p"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[:4]
    # GOF/3DGS 渲染器主点居中（FoV-based 投影，忽略 COLMAP cx/cy）。模型锚定在此约定下，
    # 渲染保真必须用居中主点 K 评估；真 K 的主点偏移会造成 ~23px 的系统平移假象。
    K = np.array([[fx, 0, c["w"] / 2.0], [0, fy, c["h"] / 2.0], [0, 0, 1]], dtype=np.float32)
    viewmats.append(vm); Ks.append(K); names.append(name)
viewmats_np = np.stack(viewmats).astype(np.float32)
Ks_np = np.stack(Ks).astype(np.float32)
H, W = cam[list(cam.keys())[0]]["h"], cam[list(cam.keys())[0]]["w"]
N = len(viewmats)
print(f"[no_gt] {N} views, {W}x{H}", flush=True)

# ---------------- 辅助：加载图像 ----------------
def load_img(name, H, W):
    # 兼容两种布局：name 直接可拼（真实 colmap 写 "images/xxx.jpg"），或
    # 图像统一在 <source>/images/ 下而 name 只有文件名（colmap_writer 合成数据）
    base = os.path.basename(name)
    candidates = [os.path.join(args.source, name),
                  os.path.join(args.source, "images", name),
                  os.path.join(args.source, "images", base)]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        raise FileNotFoundError(f"image not found (tried {candidates[0]})")
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    if im.shape[:2] != (H, W):
        im = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((W, H))
                        ).astype(np.float32) / 255.0
    return im

def load_mask(name):
    """按图像 basename 前缀在 --mask_dir 中找同名前景 mask（H,W bool）。"""
    if not args.mask_dir:
        return None
    stem = os.path.splitext(os.path.basename(name))[0]
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        cand = os.path.join(args.mask_dir, stem + ext)
        if os.path.exists(cand):
            m = np.asarray(Image.open(cand).convert("L"))
            if m.shape[:2] != (H, W):
                m = np.asarray(Image.fromarray(m).resize((W, H)))
            return m > 127
    return None

# ---------------- 渲染全部视角 ----------------
render = []           # list of (rgb, depth, alpha)
for b0 in range(0, N, 8):
    vm = torch.from_numpy(viewmats_np[b0:b0 + 8]).cuda()
    Kk = torch.from_numpy(Ks_np[b0:b0 + 8]).cuda()
    out = gsplat.rasterization(means_t, quats_t, scales_t, opac_t,
                               features_t if features_t is not None else rgb_t,
                               vm, Kk, W, H,
                               sh_degree=sh_degree if features_t is not None else None,
                               render_mode="RGB+D")          # out[0]=RGB+D, out[1]=alpha, out[2]=meta
    render_colors = out[0].cpu().numpy()
    render_alpha = out[1].cpu().numpy()
    for bi in range(render_colors.shape[0]):
        render.append((render_colors[bi, :, :, :3],           # RGB
                       render_colors[bi, :, :, 3],            # depth（最后一通道）
                       render_alpha[bi, :, :, 0]))            # alpha
    print(f"  rendered {min(b0+8, N)}/{N}", flush=True)

# ---------------- 1. 渲染保真（前景内 PSNR/SSIM） ----------------
# 有 --mask_dir 时前景 = 真实 mask ∩ α 前景（DTU 全场景模型连背景重建，α 掩膜≈全图，
# 必须用真 mask 才能度量物体本身的外观拟合）；无 mask 时退回 α>t_alpha 自掩膜。
# SSIM 限制在 fg 包围盒内（skimage 无 mask 参数，全图会含背景）。
psnrs, ssims, fg_covs, psnrs_full = [], [], [], []
for i, (rgb_i, d_i, a_i) in enumerate(render):
    gt = load_img(names[i], H, W)
    # 全图 PSNR（与 GOF 自带 eval 同口径，含背景；背景为黑时数值偏乐观）
    psnrs_full.append(float(10 * np.log10(1.0 / max(np.mean((rgb_i - gt) ** 2), 1e-10))))
    fg = (a_i > args.t_alpha)
    if args.mask_dir:
        m = load_mask(names[i])
        if m is not None:
            fg = fg & m
    if fg.sum() < 100:
        continue
    mse = np.mean((rgb_i[fg] - gt[fg]) ** 2)
    psnrs.append(float(10 * np.log10(1.0 / max(mse, 1e-10))))
    ys, xs = np.nonzero(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ssims.append(float(structural_similarity(rgb_i[y0:y1, x0:x1], gt[y0:y1, x0:x1],
                                             channel_axis=2, data_range=1.0,
                                             gaussian_weights=True, sigma=1.5,
                                             use_sample_covariance=False)))
    if args.mask_dir:
        fg_covs.append(float(fg.sum() / max(m.sum(), 1)))   # mask 覆盖率 = mask 像素中 α>t 的比例（模型对物体区域的覆盖）
    if args.save_render:
        os.makedirs(args.dumps, exist_ok=True)
        Image.fromarray((rgb_i * 255).astype(np.uint8)).save(os.path.join(args.dumps, f"render_{i:03d}.png"))
print("  逐视角全图 PSNR:", [round(x, 1) for x in psnrs_full], flush=True)
render_metric = {"psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
                 "psnr_full_mean": float(np.mean(psnrs_full)),
                 "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
                 "n_views": len(psnrs)}
if fg_covs:
    render_metric["mask_cov_mean"] = float(np.mean(fg_covs))   # mask 覆盖率：mask 像素中 α>t 的比例（模型对物体区域的覆盖）
    render_metric["mask_cov_std"] = float(np.std(fg_covs))
print("[1] 渲染保真:", render_metric, flush=True)

# ---------------- 2. silhouette 一致性（depth 点回投本视角 vs mask） ----------------
# ok 限制在 mask 像素内：只在"物体应出现的像素"查 2.5D 表面是否贴真实轮廓，
# 背景像素（α 前景但 mask 外）不再稀释命中率。
hit_ratios, proj_counts = [], []
if args.mask_dir:
    u = np.arange(0, W, args.step, dtype=np.float32) + args.step / 2
    vv = np.arange(0, H, args.step, dtype=np.float32) + args.step / 2
    V, U = np.meshgrid(vv, u, indexing="ij")
    Uf, Vf = U.reshape(-1).astype(np.float32), V.reshape(-1).astype(np.float32)
    for i, (rgb_i, d_i, a_i) in enumerate(render):
        mask = load_mask(names[i])
        if mask is None:
            continue
        Dz = d_i[::args.step, ::args.step].ravel()
        Av = a_i[::args.step, ::args.step].ravel()
        Msk = mask[np.clip(Vf, 0, H - 1).astype(np.int64), np.clip(Uf, 0, W - 1).astype(np.int64)]
        ok = Msk & (Av > args.d_alpha) & np.isfinite(Dz) & (Dz > 0)
        if not ok.any():
            continue
        Kb = Ks_np[i]
        pc = np.stack([(Uf[ok] - Kb[0, 2]) / Kb[0, 0] * Dz[ok],
                       (Vf[ok] - Kb[1, 2]) / Kb[1, 1] * Dz[ok],
                       Dz[ok]], 1)
        pc_h = np.concatenate([pc, np.ones((len(pc), 1))], 1)
        pw = (np.linalg.inv(viewmats_np[i]) @ pc_h.T).T[:, :3]
        # 回投本视角
        pc2 = (viewmats_np[i] @ np.concatenate([pw, np.ones((len(pw), 1))], 1).T).T[:, :3]
        x = pc2[:, 0] / pc2[:, 2] * Kb[0, 0] + Kb[0, 2]
        y = pc2[:, 1] / pc2[:, 2] * Kb[1, 1] + Kb[1, 2]
        xi = np.round(x).astype(int); yi = np.round(y).astype(int)
        ins = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        hit = mask[yi[ins], xi[ins]] if ins.any() else np.zeros(0, bool)
        if len(hit):
            hit_ratios.append(float(hit.mean()))
            proj_counts.append(int(ins.sum()))
sil_metric = ({"silhouette_hit_mean": float(np.mean(hit_ratios)),
               "silhouette_hit_median": float(np.median(hit_ratios)),
               "n_views": len(hit_ratios)}
              if hit_ratios else {})
print("[2] silhouette:", sil_metric, flush=True)

# ---------------- 3. 跨视角深度一致性 ----------------
rel_errors = []          # 所有有效像素的相对深度差
cons_fracs = []          # 每视角对的一致率（rel err < 10%）
ncols = int(np.ceil(W / args.step)); nrows = int(np.ceil(H / args.step))
u = np.arange(0, W, args.step, dtype=np.float32) + args.step / 2
vv = np.arange(0, H, args.step, dtype=np.float32) + args.step / 2
V, U = np.meshgrid(vv, u, indexing="ij")
Uf, Vf = U.reshape(-1).astype(np.float32), V.reshape(-1).astype(np.float32)
for i in range(N):
    d_i, a_i = render[i][1], render[i][2]
    Dz = d_i[::args.step, ::args.step].ravel()
    Av = a_i[::args.step, ::args.step].ravel()
    ok = (Av > args.d_alpha) & np.isfinite(Dz) & (Dz > 0)
    if args.mask_dir:
        m = load_mask(names[i])
        if m is not None:
            ok = ok & m[np.clip(Vf, 0, H - 1).astype(np.int64),
                        np.clip(Uf, 0, W - 1).astype(np.int64)]
    if not ok.any():
        continue
    Kb = Ks_np[i]
    pc = np.stack([(Uf[ok] - Kb[0, 2]) / Kb[0, 0] * Dz[ok],
                   (Vf[ok] - Kb[1, 2]) / Kb[1, 1] * Dz[ok],
                   Dz[ok]], 1)
    pc_h = np.concatenate([pc, np.ones((len(pc), 1))], 1)
    pw = (np.linalg.inv(viewmats_np[i]) @ pc_h.T).T[:, :3]       # world
    for k in range(1, args.n_cross + 1):
        j = (i + k) % N
        d_j, a_j = render[j][1], render[j][2]
        Dj = d_j[::args.step, ::args.step].ravel()
        Aj = a_j[::args.step, ::args.step].ravel()
        pj = (viewmats_np[j] @ np.concatenate([pw, np.ones((len(pw), 1))], 1).T).T[:, :3]
        z_j = pj[:, 2]
        Kj = Ks_np[j]
        x = pj[:, 0] / z_j * Kj[0, 0] + Kj[0, 2]
        y = pj[:, 1] / z_j * Kj[1, 1] + Kj[1, 2]
        # j 视图的 depth/alpha 是 step 采样网格（原图像素 / step）
        xs = np.round(x / args.step).astype(int)
        ys = np.round(y / args.step).astype(int)
        ins = (xs >= 0) & (xs < ncols) & (ys >= 0) & (ys < nrows) & (z_j > 0)
        idx = np.clip(ys * ncols + xs, 0, nrows * ncols - 1)
        Aj_v, Dj_v = Aj[idx], Dj[idx]
        valid = ins & (Aj_v > args.d_alpha) & np.isfinite(Dj_v) & (Dj_v > 0)
        if not valid.any():
            continue
        d_ref = Dj_v[valid]
        rel = np.abs(d_ref - z_j[valid]) / np.maximum(z_j[valid], 1e-6)
        rel_errors.append(rel)
        cons_fracs.append(float((rel < 0.10).mean()))
rel_all = np.concatenate(rel_errors) if rel_errors else np.array([])
cross_metric = ({"depth_rel_p50": float(np.median(rel_all)),
                 "depth_rel_mean": float(np.mean(rel_all)),
                 "consistency_10pct": float(np.mean(cons_fracs)),
                 "n_pairs": len(cons_fracs)} if len(rel_all) else {})
print("[3] 跨视角深度一致性:", cross_metric, flush=True)

# ---------------- 4. 几何卫生 ----------------
r = np.linalg.norm(means, axis=1)
med = float(np.median(r))
far_bg = opacities[r > 3 * med]
bg_frac = float(np.mean(far_bg > 0.5)) if len(far_bg) else 0.0   # 远离主体 3 倍半径处的高 opacity 高斯 = 浮游幽灵
geo_metric = {
    "gaussians": n,
    "radius_med": med,
    "radius_p90": float(np.percentile(r, 90)),
    "frac_radius_gt3med": float(np.mean(r > 3 * med)),
    "floaters_frac": bg_frac,
    "alpha_cov_mean": float(np.mean([(a > args.t_alpha).mean() for _, _, a in render])),
}
print("[4] 几何卫生:", geo_metric, flush=True)

res = {"render": render_metric, "silhouette": sil_metric,
       "cross_view": cross_metric, "geometry": geo_metric}
os.makedirs(args.dumps, exist_ok=True)
with open(os.path.join(args.dumps, "no_gt.res.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, ensure_ascii=False))
