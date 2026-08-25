"""E0 diagnostic 3: is the black render due to SH colors or to geometry/alpha?

- override_color = bright red for all Gaussians -> if torus shows, alpha ok, SH broken.
- convert_SHs_python=True  -> same check via the python SH path.
"""
import argparse

import numpy as np
import torch

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render


def stats(a, label):
    print(f"{label}: mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}")

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

    # (1) override_color = red
    n = g.get_xyz.shape[0]
    red = torch.full((n, 3), 0.9, device="cuda")
    red[:, 1] = 0.05
    red[:, 2] = 0.05
    out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size, override_color=red)
    r = out["render"].detach().cpu().numpy()[:3]
    stats(r[0], "override-red R")
    stats(r[1], "override-red G")
    stats(r[2], "override-red B")

    # (2) convert_SHs_python
    pipe2 = pp.extract(args)
    pipe2.convert_SHs_python = True
    out2 = render(cam, g, pipe2, bg, kernel_size=ds.kernel_size)
    r2 = out2["render"].detach().cpu().numpy()[:3]
    stats(r2[0], "python-SH R")
    stats(r2[1], "python-SH G")
    stats(r2[2], "python-SH B")

    # (3) pixel coverage: fraction of pixels where any channel > 0.02
    msk = (r2.max(axis=0) > 0.02)
    gt = cam.original_image.detach().cpu().numpy()
    gmsk = gt.max(axis=0) > 0.04
    print(f"GT object pixel frac: {gmsk.mean():.3f}  | python-SH bright pixel frac: {msk.mean():.3f}")


if __name__ == "__main__":
    main()
