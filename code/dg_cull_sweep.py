"""E0: sweep cull thresholds (s_max, d_max) -> keep count, hole-ray 1-T, PSNR.
Hole ray B: eye(el70)->center, expected T~1 (empty hole).
Also add thin-disk (min-axis scale) criterion as an alternative ghost killer.
"""
import argparse, os, sys
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from clean_ray_profile import ray_march
from cull_cloud import cull_gaussians

ap = argparse.ArgumentParser()
lp = ModelParams(ap); pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)
g0 = GaussianModel(ds.sh_degree)
scene = Scene(ds, g0, load_iteration=args.iteration, shuffle=False)

O = torch.tensor([1.094,0,3.007], device="cuda")   # el70 eye
Tgt = torch.tensor([0,0,0], device="cuda")

def hole_transmittance(g):
    t, o, Tt, P = ray_march(O, Tgt - O, g, 0.05, 8.0, K=1000)
    return Tt[-1].item(), o.max().item()

import copy
for s_max, d_max in [(0.2,0.3),(0.2,0.2),(0.2,0.15),(0.15,0.1),(0.1,0.05)]:
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    keep, st = cull_gaussians(g, s_max, d_max)
    Tf, omax = hole_transmittance(g)
    print(f"s_max={s_max:.2f} d_max={d_max:.2f}: keep={st['keep']}/{st['total']} "
          f"({100*st['keep_frac']:.1f}%) op_kept={100*st['opacity_kept']:.1f}%  "
          f"hole-ray: 1-T={1-Tf:.3f} o_max={omax:.3f}")
