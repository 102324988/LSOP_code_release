"""E0: minimal — what contributes alpha at the t=0.05 sample of the
cam0->outer_surf ray? Verify ray_march against a hand-computed maha."""
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
from clean_ray_profile import quat2mat  # noqa: E402


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
        from cull_cloud import cull_gaussians  # noqa: E402
        keep, st = cull_gaussians(g, args.cull_s_max, args.cull_d_max)
        print(f"[cull] keep {st['keep']}/{st['total']}")
    N = g._xyz.shape[0]
    smax = torch.exp(g._scaling.detach()).max(1).values
    print(f"gaussians={N} scale max={smax.max().item():.3f} "
          f"p99={smax.quantile(0.99).item():.3f}")

    O = torch.tensor([3.007, 0.0, 1.094], device="cuda")
    P = torch.tensor([0.78, 0.0, 0.0], device="cuda")
    dvec = P - O
    dvec = dvec / torch.linalg.norm(dvec)
    t0 = 0.05
    S = O + dvec * t0

    xyz = g._xyz.detach()
    scl = torch.exp(g._scaling.detach())
    op = torch.sigmoid(g._opacity.detach())
    Rm = quat2mat(g._rotation.detach())

    diff = (S[None, :] - xyz)[:, None, :]  # (N,1,3)
    diff_r = torch.einsum("nij,njk->nik", diff, Rm).squeeze(1)  # (N,3)
    maha = ((diff_r / scl) ** 2).sum(-1)
    contrib = (op.squeeze() * torch.exp(-0.5 * maha))
    print(f"sum of all contribs at t=0.05: {contrib.sum().item():.4f}")

    top = torch.argsort(contrib, descending=True)[:10]
    for i in top.tolist():
        print(f"  idx={i} contrib={contrib[i].item():.6f} "
              f"dist={torch.linalg.norm(xyz[i]-S).item():.4f} "
              f"scale={scl[i].cpu().numpy().round(4)} "
              f"op={op[i].item():.3f} pos={xyz[i].cpu().numpy().round(3)}")

    # hand-check: is maha tiny for the top contributor?
    i = top[0].item()
    print(f"\n[hand] top contrib idx={i}: maha={maha[i].item():.4f} "
          f"exp(-.5maha)={torch.exp(-0.5*maha[i]).item():.6f} op={op[i].item():.3f}")


if __name__ == "__main__":
    main()
