"""E0 diagnostic 2: per-channel stats of the rasterizer render output.
"""
import argparse
import os

import numpy as np
import torch

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=15000)
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    ds = lp.extract(args)
    pipe = pp.extract(args)
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cam = scene.getTrainCameras()[args.cam]

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
    r = out["render"].detach().cpu().numpy()  # (C,H,W)
    print("render shape:", r.shape)
    for c in range(r.shape[0]):
        print(f"ch {c}: mean={r[c].mean():.4f} std={r[c].std():.4f} "
              f"min={r[c].min():.4f} max={r[c].max():.4f}")

    gt = cam.original_image.detach().cpu().numpy()
    print("GT shape:", gt.shape, "mean:", gt.mean(), "std:", gt.std())

    # also check Gaussian stats as seen by the rasterizer (filtered getters)
    xyz = g.get_xyz.detach().cpu().numpy()
    op = g.get_opacity_with_3D_filter.detach().cpu().numpy()
    sc = g.get_scaling_with_3D_filter.detach().cpu().numpy()
    print(f"N={xyz.shape[0]} opacity[min/mean/max]={op.min():.3f}/{op.mean():.3f}/{op.max():.3f}")
    print(f"scaling[min/mean/max]={sc.min():.3f}/{sc.mean():.3f}/{sc.max():.3f}")


if __name__ == "__main__":
    main()
