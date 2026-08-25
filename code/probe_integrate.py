"""Probe integrate() per-view alpha_integrated for known query points."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import integrate
from scene import GaussianModel, Scene


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
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras()

    # query points: (world xyz, expected)
    query = {
        "above_torus(0,0,0.6)": (0.0, 0.0, 0.6),     # clearly empty, above XY-plane torus
        "outer_equator(0.95,0,0)": (0.95, 0.0, 0.0),  # on surface
        "hole_center(0,0,0)": (0.0, 0.0, 0.0),        # inside torus hole (empty but occluded?)
        "far_side(0,0.5,0.6)": (0.0, 0.5, 0.6),       # empty, above torus, off to the side
    }
    pts = np.array(list(query.values()), dtype=np.float32)
    p = torch.from_numpy(pts).cuda()

    with torch.no_grad():
        alphas = []
        for i, view in enumerate(views[:8]):
            ret = integrate(p, view, g, pipe, bg, kernel_size=ds.kernel_size)
            alphas.append(ret["alpha_integrated"].cpu().numpy())
    alphas = np.stack(alphas)  # (n_views, n_pts)

    print(f"{'point':>28s} " + " ".join(f"v{i:<6d}" for i in range(alphas.shape[0])) + "  final_o=1-min")
    for k, (name, _) in enumerate(query.items()):
        row = " ".join(f"{alphas[i, k]:.3f}" for i in range(alphas.shape[0]))
        print(f"{name:>28s} {row}  {1 - alphas[:, k].min():.3f}")


if __name__ == "__main__":
    main()
