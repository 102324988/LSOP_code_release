"""E0 diagnostic 4: inspect the loaded model's SH features and eval_sh output."""
import argparse

import numpy as np
import torch

from arguments import ModelParams
from scene import Scene, GaussianModel
from utils.sh_utils import eval_sh
from utils.sh_utils import C0 as SH_C0


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=15000)
    args = parser.parse_args()

    ds = lp.extract(args)
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cam = scene.getTrainCameras()[0]

    feats = g.get_features  # (N, 3, (deg+1)^2)
    print("features shape:", tuple(feats.shape))
    dc = feats[:, :, 0]
    rest = feats[:, :, 1:]
    print(f"active_sh_degree: {g.active_sh_degree}  max_sh_degree: {g.max_sh_degree}")
    print(f"DC: mean={dc.mean():.4f} std={dc.std():.4f} min={dc.min():.4f} max={dc.max():.4f}")
    print(f"REST: absmax={rest.abs().max():.4f} mean_abs={rest.abs().mean():.4f}")

    # manual eval_sh like the render path
    shs_view = feats.transpose(1, 2).view(-1, 3, (g.max_sh_degree + 1) ** 2)
    dir_pp = (g.get_xyz - cam.camera_center.repeat(g.get_xyz.shape[0], 1))
    dir_pp_n = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(g.active_sh_degree, shs_view, dir_pp_n)
    color = torch.clamp_min(sh2rgb + 0.5, 0.0)
    print(f"eval_sh color: mean={color.mean():.4f} std={color.std():.4f} "
          f"min={color.min():.4f} max={color.max():.4f}")
    print("SH_C0 =", SH_C0)


if __name__ == "__main__":
    main()
