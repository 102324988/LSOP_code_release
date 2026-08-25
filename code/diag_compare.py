"""E0: pinpoint ray_march vs hand-computed discrepancy at t=0.05."""
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
from clean_ray_profile import quat2mat, ray_march  # noqa: E402
from cull_cloud import cull_gaussians  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--cull_s_max", type=float, default=0.0)
    ap.add_argument("--cull_d_max", type=float, default=0.0)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    if args.cull_s_max > 0:
        keep, st = cull_gaussians(g, args.cull_s_max, args.cull_d_max)
        print(f"[cull] keep {st['keep']}/{st['total']}")

    O = torch.tensor([3.007, 0.0, 1.094], device="cuda")
    P = torch.tensor([0.78, 0.0, 0.0], device="cuda")
    d = P - O
    d = d / torch.linalg.norm(d)
    t0 = 0.05
    S = O + d * t0

    xyz = g._xyz.detach()
    scl_act = torch.exp(g._scaling.detach())
    op = torch.sigmoid(g._opacity.detach())
    Rm = quat2mat(g._rotation.detach())

    diff = (S[None, :] - xyz)[:, None, :]
    diff_r = torch.einsum("nij,njk->nik", diff, Rm).squeeze(1)
    maha = ((diff_r / scl_act) ** 2).sum(-1)
    contrib = op.squeeze() * torch.exp(-0.5 * maha)
    print(f"[hand] scale range {scl_act.min():.5f}..{scl_act.max():.3f} "
          f"sum={contrib.sum().item():.6f}")

    # same point via ray_march's first sample
    t, a, T, Pc = ray_march(O, P - O, g, 0.05, 2.5, K=500)
    print(f"[ray_march] a[0]={a[0]:.6f}  (hand sum should match)")


if __name__ == "__main__":
    main()
