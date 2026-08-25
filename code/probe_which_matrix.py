"""Decisive: which matrix convention projects trained Gaussians onto the render?
Render cam0 (known correct), project all centers via candidates, correlate."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene


def ndc2Pix(v, W):
    return 0.5 * (v + 1.0) * W - 0.5


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=4000)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cam = scene.getTrainCameras()[0]
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    with torch.no_grad():
        out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
    rgb = out["render"].detach().cpu().permute(1, 2, 0).numpy()[:, :, :3]
    obj = rgb.max(axis=-1) < 0.94  # object mask (white bg)
    print(f"render obj mask: {obj.sum()} px")

    W, H = cam.image_width, cam.image_height
    xyz = g.get_xyz  # (N,3)
    N = xyz.shape[0]
    p = torch.cat([xyz, torch.ones(N, 1, device="cuda")], dim=1)  # (N,4)

    FPT = cam.full_proj_transform  # (4,4) value
    candidates = {
        "FPT (row-major value)": FPT,
        "FPT^T": FPT.T,
        "FPT stored-as-view @ PM": None,
    }
    for name, M in candidates.items():
        if M is None:
            continue
        ph = (M.detach() @ p.T).detach().T  # (N,4)
        pw = ph[:, 3].detach().clamp(min=1e-6)
        px = ndc2Pix(ph[:, 0] / pw, W).cpu().numpy()
        py = ndc2Pix(ph[:, 1] / pw, H).cpu().numpy()
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (pw.cpu().numpy() > 0)
        n_inside = inside.sum()
        if n_inside == 0:
            print(f"  {name}: {n_inside}/{N} centers inside image (nothing to compare)")
            continue
        xin = np.clip(px[inside].astype(int), 0, W - 1)
        yin = np.clip(py[inside].astype(int), 0, H - 1)
        hit = obj[yin, xin].mean()
        print(f"  {name}: {n_inside}/{N} inside, frac on object mask={hit:.3f}")


if __name__ == "__main__":
    main()
