"""E0 diagnostic: render the trained GOF model at a given iteration and compare
to the ground-truth input images, separating object-mask vs background PSNR.

Usage: python diag_render.py -s data/<obj> -m output/<obj>_<run> --iteration 15000
"""
import argparse
import os

import numpy as np
import torch

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render


def main():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=15000)
    parser.add_argument("--max_cams", type=int, default=8)
    args = parser.parse_args()

    dataset = lp.extract(args)
    pipeline = pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cams = scene.getTrainCameras()

    psnr_all, psnr_obj, psnr_bg = [], [], []
    for cam in cams[:args.max_cams]:
        out = render(cam, gaussians, pipeline, bg, kernel_size=dataset.kernel_size)
        rend = out["render"].detach().permute(1, 2, 0).cpu().numpy()[:, :, :3]
        gt = cam.original_image.permute(1, 2, 0).cpu().numpy()
        msk = gt.max(axis=-1) > 0.04
        mse_all = np.mean((rend - gt) ** 2)
        mse_obj = np.mean((rend[msk] - gt[msk]) ** 2) if msk.sum() else float("nan")
        mse_bg = np.mean((rend[~msk] - gt[~msk]) ** 2) if (~msk).sum() else float("nan")
        psnr_all.append(10 * np.log10(1.0 / (mse_all + 1e-12)))
        psnr_obj.append(10 * np.log10(1.0 / (mse_obj + 1e-12)))
        psnr_bg.append(10 * np.log10(1.0 / (mse_bg + 1e-12)))

    print(f"iteration {args.iteration}:")
    print(f"  PSNR all ={np.mean(psnr_all):6.2f}  object={np.mean(psnr_obj):6.2f}  bg={np.mean(psnr_bg):6.2f}")
    print(f"  bg pixel frac ={float((~msk).mean()):.3f}")

    # object-region color stats: is the render too dark / saturated?
    rend = rend.mean(axis=2)  # last camera
    gt = gt.mean(axis=2)
    print(f"  last cam: render lum mean={rend.mean():.3f} std={rend.std():.3f}"
          f" | gt lum mean={gt.mean():.3f} std={gt.std():.3f}")


if __name__ == "__main__":
    main()
