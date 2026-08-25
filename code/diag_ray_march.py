"""E0: diagnose clean ray-march along a specific ray: where does alpha accumulate?"""
import argparse
import os
import sys

import numpy as np
import torch

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from clean_ray_profile import ray_march  # noqa: E402
from cull_cloud import cull_gaussians  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--cull_s_max", type=float, default=0.0)
    ap.add_argument("--cull_d_max", type=float, default=0.0)
    ap.add_argument("--target", default="0.78,0,0", help="ray target point")
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    if args.cull_s_max > 0:
        keep, st = cull_gaussians(g, args.cull_s_max, args.cull_d_max)
        print(f"[cull] keep {st['keep']}/{st['total']}")

    O = torch.tensor([3.007, 0.0, 1.094], device="cuda")  # cam0 position
    P = torch.tensor([float(v) for v in args.target.split(",")], device="cuda")
    d = P - O
    r = torch.linalg.norm(d)
    t, a, T, Pc = ray_march(O, d, g, 0.05, r, K=500)

    print(f"ray {O.tolist()} -> {P.tolist()}  r={r:.3f}")
    for i in range(0, len(t), 50):
        print(f"  t={t[i]:6.3f}  alpha={a[i]:8.4f}  T={T[i]:.4f}  "
              f"P={Pc[i]:.4f}")
    j = int(np.argmax(a))
    pt = O + d / r * t[j]
    print(f"max alpha {a[j]:.3f} at t={t[j]:.3f} -> point {pt.tolist()}")
    print(f"final 1-T = {1 - T[-1]:.4f}")

    # find nearest gaussians to the max-alpha point
    xyz = g._xyz.detach().cpu().numpy()
    scl = torch.exp(g._scaling.detach()).cpu().numpy()
    op = torch.sigmoid(g._opacity.detach()).cpu().numpy().squeeze()
    Q = pt.cpu().numpy()
    dd = np.linalg.norm(xyz - Q, axis=1)
    idx = np.argsort(dd)[:8]
    for i in idx:
        print(f"  near dist={dd[i]:.4f} scale={scl[i].round(3)} "
              f"op={op[i]:.3f} pos={xyz[i].round(3)}")


if __name__ == "__main__":
    main()
