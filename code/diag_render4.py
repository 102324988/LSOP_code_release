"""E0 diagnostic 5: definitively isolate whether the black render comes from
the SH features, the eval path, or the rasterizer.

1. compute colors_precomp via the SAME code as render's python-SH branch
2. pass them as override_color -> if bright, rasterizer is fine, and the
   python-SH branch inside render must be misbehaving
3. also try the GPU SH path (convert_SHs_python=False) on the same cam
"""
import argparse

import numpy as np
import torch

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.sh_utils import eval_sh


def mean3(r, label):
    r = r[:3]
    print(f"{label}: R={r[0].mean():.4f} G={r[1].mean():.4f} B={r[2].mean():.4f} "
          f"max={r.max():.4f}")


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

    # --- reproduce render's python-SH colors exactly ---
    feats = g.get_features  # (N,16,3)
    shs_view = feats.transpose(1, 2).view(-1, 3, (g.max_sh_degree + 1) ** 2)
    dir_pp = g.get_xyz - cam.camera_center.repeat(g.get_xyz.shape[0], 1)
    dir_pp_n = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(g.active_sh_degree, shs_view, dir_pp_n)
    colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
    print(f"manual colors_precomp: mean={colors.mean():.3f} min={colors.min():.3f} max={colors.max():.3f}")

    # (a) override_color = these colors
    out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size, override_color=colors)
    mean3(out["render"], "render(manual colors)")

    # (b) GPU SH path
    out2 = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
    mean3(out2["render"], "render(GPU SH, default)")

    # (c) python-SH path via pipe flag
    pipe2 = pp.extract(args)
    pipe2.convert_SHs_python = True
    out3 = render(cam, g, pipe2, bg, kernel_size=ds.kernel_size)
    mean3(out3["render"], "render(pipe convert_SHs_python=True)")


if __name__ == "__main__":
    main()
