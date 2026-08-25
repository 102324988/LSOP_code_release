"""Decisive: render a handful of trained Gaussians individually, get their true
pixels, then test which projection formula reproduces them."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import getProjectionMatrix


def ndc2Pix(v, W):
    return 0.5 * (v + 1.0) * W - 0.5


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=4000)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cam = scene.getTrainCameras()[0]
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    W, H = cam.image_width, cam.image_height
    xyz = g.get_xyz  # (N,3)
    N = xyz.shape[0]
    idxs = list(range(0, N, max(1, N // args.n)))[:args.n]

    # candidate projection matrices
    R = torch.tensor(cam.R, dtype=torch.float32, device="cuda")       # (3,3) stored
    T = torch.tensor(cam.T, dtype=torch.float32, device="cuda")       # (3,)
    P = getProjectionMatrix(cam.znear, cam.zfar, cam.FoVx, cam.FoVy).to("cuda")  # (4,4)
    I4 = torch.eye(4, dtype=torch.float32, device="cuda")

    W2C_proper = I4.clone()
    W2C_proper[:3, :3] = R
    W2C_proper[:3, 3] = -R @ T
    W2C_shift = I4.clone()
    W2C_shift[:3, :3] = R

    FPT = cam.full_proj_transform  # (4,4)
    candidates = {
        "FPT_value": FPT,
        "FPT^T": FPT.T,
        "P@W2C_proper": P @ W2C_proper,
        "P@W2C_shift": P @ W2C_shift,
    }

    print(f"{'gauss':>6s} {'xyz':^22s} {'true_pix':^12s}", end="")
    for name in candidates:
        print(f" {name[:12]:>14s}", end="")
    print()

    base_op = g.get_opacity.clone()
    base_scale = g.get_scaling.clone()
    for i in idxs:
        # zero all opacities except gaussian i
        op = torch.full_like(base_op, -20.0)
        op[i] = base_op[i]
        g._opacity.data.copy_(op)
        g._scaling.data.copy_(base_scale * 1.0)
        with torch.no_grad():
            out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
        rgb = out["render"].detach().cpu().permute(1, 2, 0).numpy()[:, :, :3]
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if gray.max() < 0.01:
            tpix = "(-,-)"
        else:
            yy, xx = np.unravel_index(gray.argmax(), gray.shape)
            tpix = f"({xx},{yy})"
        g._opacity.data.copy_(base_op)

        p = torch.cat([xyz[i], torch.ones(1, device="cuda")])
        row = f"{i:6d} {xyz[i].detach().cpu().numpy().__str__():^22s} {tpix:^12s}"
        for name, M in candidates.items():
            ph = (M @ p).detach()
            px = ndc2Pix(ph[0] / ph[3], W).item()
            py = ndc2Pix(ph[1] / ph[3], H).item()
            row += f" {f'({px:.0f},{py:.0f})':>14s}"
        print(row)


if __name__ == "__main__":
    main()
