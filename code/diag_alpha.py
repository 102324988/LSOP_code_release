"""E0: render alpha image (from 3DGS rasterizer) and sample probe pixels.
Distinguishes 'model really has opacity in empty region' (geometry problem)
vs 'integrate kernel inconsistent with render' (kernel problem).
"""
import argparse
import os
import sys

import numpy as np
import torch

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cam = scene.getTrainCameras()[args.cam]
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    with torch.no_grad():
        out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
    print("keys:", list(out.keys()))
    for k, t in out.items():
        if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[0] >= 1:
            a = t[0].detach().cpu().numpy()
            print(f"{k}: shape={tuple(t.shape)} min={a.min():.3f} "
                  f"mean={a.mean():.3f} max={a.max():.3f}")
            # far_empty (0,1.2,0)->(812,300); outer_surf->(400,389);
            # tube_center->(400,358); top_surf->(400,291)
            for name, (px, py) in {"far_empty": (812, 300), "outer_surf": (400, 389),
                                   "tube_center": (400, 358), "top_surf": (400, 291)}.items():
                if py < a.shape[0] and px < a.shape[1]:
                    print(f"   {name}({px},{py})={a[py, px]:.3f}")


if __name__ == "__main__":
    main()
