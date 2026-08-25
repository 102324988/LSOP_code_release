#!/usr/bin/env python3
"""diag_gof_render.py: 用 GOF 自身 render 管线渲染训练视角，与我们 gsplat 渲染逐像素对比。

目的：定位 eval_no_gt PSNR(~15) vs GOF 自带(26.89) 的差异来源。
用法: python diag_gof_render.py <model_path> <source_path> [n_views]
"""
import sys

import numpy as np
import torch
from argparse import ArgumentParser

from scene import Scene
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from utils.image_utils import psnr

model_path, source_path = sys.argv[1], sys.argv[2]
n_views = int(sys.argv[3]) if len(sys.argv) > 3 else 5

parser = ArgumentParser()
mp = ModelParams(parser)
pp = PipelineParams(parser)
args = parser.parse_args([])
dataset = mp.extract(args)
dataset.source_path = source_path
dataset.model_path = model_path
pipeline = pp.extract(args)
dataset.sh_degree = 3

gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=20000)
bg = torch.zeros(3, dtype=torch.float32, device="cuda")

gof_psnrs, our_psnrs = [], []
cams = scene.getTrainCameras()
for ci in range(min(n_views, len(cams))):
    cam = cams[ci]
    # GOF 自身渲染
    with torch.no_grad():
        out = render(cam, gaussians, pipeline, bg, kernel_size=dataset.kernel_size)
    gof_img = out["render"][:3].clamp(0, 1)
    gt = cam.original_image.clamp(0, 1).cuda()
    p_gof = psnr(gof_img, gt).mean().item()
    # 我们 gsplat：用同一 viewmatrix/proj 与高斯参数
    import gsplat
    means = gaussians.get_xyz.detach().float()
    quats = gaussians.get_rotation.detach().float()
    scales = gaussians.get_scaling_with_3D_filter.detach().float()
    opac = gaussians.get_opacity_with_3D_filter.detach().float()
    features = gaussians.get_features.detach().float()          # [N, 48]
    # GOF Camera 无 cx/cy：3DGS 渲染器默认主点居中（FoV-based 投影矩阵）。
    # gsplat 用相同约定：K = [[fx,0,W/2],[0,fy,H/2],[0,0,1]]。
    K = torch.tensor([[cam.focal_x, 0, cam.image_width/2],
                      [0, cam.focal_y, cam.image_height/2],
                      [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
    vm = cam.world_view_transform.detach().float()               # 与 GOF 完全一致的 viewmat
    # gsplat 用 W2C + K
    out2 = gsplat.rasterization(means, quats, scales, opac,
                                features.reshape(-1, 16, 3), vm[None], K,
                                cam.image_width, cam.image_height, sh_degree=3,
                                render_mode="RGB+D")
    our_img = out2[0][0, :, :, :3].clamp(0, 1).permute(2, 0, 1)
    p_our = psnr(our_img, gt).mean().item()
    gof_psnrs.append(p_gof); our_psnrs.append(p_our)
    print(f"view {ci} ({cam.image_name}): GOF_PSNR={p_gof:.2f}  our_gsplat={p_our:.2f}", flush=True)
    # 对比两张渲染本身
    diff = (gof_img - our_img).abs().mean().item()
    print(f"   render|render| mean abs diff = {diff:.4f}", flush=True)

print(f"[diag] 均值: GOF={np.mean(gof_psnrs):.2f}  our={np.mean(our_psnrs):.2f}")
