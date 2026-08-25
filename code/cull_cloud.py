"""E0: cull floating / giant gaussians in-memory.

Floating gaussians (far from the true surface, or with giant scale) are the
documented reason the GOF field and ray profiles collapse: a single gaussian
with scale~15 covers the whole scene bbox with opacity, making A_v ~ 1 along
every ray.  Provide cull_gaussians(g, s_max, d_max) that masks the model's
parameters in place, and a CLI to report stats.
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


def cull_gaussians(g, s_max=0.2, d_max=0.3, R=0.55, r=0.23, hole_cull=False):
    """Mask gaussians with max-scale > s_max OR dist to torus surface > d_max
    (optionally also those inside the torus hole cylinder, rxy<R-r, |z|<r).
    Returns (keep_indices, stats dict).  Mutates g in place."""
    with torch.no_grad():
        xyz = g._xyz.detach()
        scl = torch.exp(g._scaling.detach())
        op = g._opacity.detach()
        rot = g._rotation.detach()
        fdc = g._features_dc.detach()
        frst = g._features_rest.detach()

    smax = scl.max(dim=1).values.cpu().numpy()
    x = xyz.cpu().numpy()
    rxy = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    d = np.abs(np.sqrt((rxy - R) ** 2 + x[:, 2] ** 2) - r)

    keep = (smax <= s_max) & (d <= d_max)
    n_hole = 0
    if hole_cull:
        in_hole = (rxy < (R - r)) & (np.abs(x[:, 2]) < r)
        n_hole = int(in_hole.sum())
        keep &= ~in_hole
    opa = 1 / (1 + np.exp(-op.cpu().numpy().squeeze()))
    stats = {
        "total": len(x), "keep": int(keep.sum()),
        "keep_frac": float(keep.mean()),
        "opacity_kept": float(opa[keep].sum() / opa.sum()),
        "culled_by_scale": int((smax > s_max).sum()),
        "culled_by_dist": int((d > d_max).sum()),
        "culled_hole": n_hole,
    }

    k = torch.from_numpy(keep).to(xyz.device)
    g._xyz = torch.nn.Parameter(xyz[k].contiguous())
    g._scaling = torch.nn.Parameter(op.new_zeros((int(keep.sum()), 3)).copy_(torch.log(scl[k])))
    g._opacity = torch.nn.Parameter(op[k].contiguous())
    g._rotation = torch.nn.Parameter(rot[k].contiguous())
    g._features_dc = torch.nn.Parameter(fdc[k].contiguous())
    g._features_rest = torch.nn.Parameter(frst[k].contiguous())
    # invalidate cached per-gaussian filter buffers (get_*_with_3D_filter)
    if hasattr(g, "filter_3D") and g.filter_3D is not None and g.filter_3D.numel() != int(keep.sum()):
        g.filter_3D = g.filter_3D[k].contiguous()
    return keep, stats


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--s_max", type=float, default=0.2)
    ap.add_argument("--d_max", type=float, default=0.3)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    keep, st = cull_gaussians(g, args.s_max, args.d_max)
    print(f"total={st['total']} keep={st['keep']} ({100*st['keep_frac']:.1f}%) "
          f"opacity-kept={100*st['opacity_kept']:.1f}%")
    print(f"  culled by scale>{args.s_max}: {st['culled_by_scale']}  "
          f"culled by dist>{args.d_max}: {st['culled_by_dist']}")


if __name__ == "__main__":
    main()
