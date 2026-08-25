"""E0: global geometry stats of the trained 3DGS cloud.

Quantify how much of the opacity lives away from the true torus surface
(= floating gaussians), which corrupt GOF field extraction.
GT torus: major R=0.55, minor r=0.23, in XY plane (implicit
(sqrt(x^2+y^2)-0.55)^2 + z^2 - 0.23^2 = 0).
"""
import argparse
import os
import sys

import numpy as np
import torch

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)

from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)

    xyz = g.get_xyz.detach().cpu().numpy()          # (N,3)
    op = torch.sigmoid(g.get_opacity).detach().cpu().numpy().squeeze()
    scale = torch.exp(g.get_scaling).detach().cpu().numpy()   # (N,3)

    N = xyz.shape[0]
    print(f"gaussians: {N}")
    print(f"xyz bbox min={xyz.min(0)} max={xyz.max(0)}")
    r_xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    surf_dist = np.abs(np.sqrt((r_xy - 0.55) ** 2 + xyz[:, 2] ** 2) - 0.23)

    for thr in [0.0, 0.05, 0.1, 0.15, 0.25, 0.5]:
        m = surf_dist > thr
        print(f"  dist_to_surface > {thr:4.2f}: {m.sum():6d} ({100*m.mean():5.1f}%) "
              f"opacity-weighted {100*(op[m].sum()/op.sum()):5.1f}%")

    # floating: high-opacity gaussians far from surface
    far = (surf_dist > 0.15)
    print(f"\nfloating (d>0.15): {far.sum()} gaussians; opacity distribution there:")
    if far.sum():
        for q in [0.1, 0.5, 0.9]:
            print(f"    op quantile {q}: {np.quantile(op[far], q):.3f}")
        print(f"    max op: {op[far].max():.3f}")
        # where are they? bbox + a few samples
        print(f"    bbox: min={xyz[far].min(0)} max={xyz[far].max(0)}")
        idx = np.argsort(op[far])[-5:]
        big = np.where(far)[0][idx]
        for i in big:
            print(f"    op={op[i]:.2f} scale={scale[i].round(3)} "
                  f"pos={xyz[i].round(3)}")

    # tube density: how well populated is the true tube?
    tube = surf_dist < 0.1
    print(f"\ntube (d<0.1): {tube.sum()} gaussians, op mean {op[tube].mean():.3f} "
          f"op-weighted frac {100*op[tube].sum()/op.sum():.1f}%")

    # opacity-weighted center of mass
    com = (xyz * op[:, None]).sum(0) / op.sum()
    print(f"opacity-weighted COM: {com.round(3)}  (GT ~ origin)")


if __name__ == "__main__":
    main()
