"""E0: hole_cull effect on hole-ray + A-ray + render PSNR."""
import argparse, os, sys
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from clean_ray_profile import ray_march, torus_hit_ranges
from cull_cloud import cull_gaussians

ap = argparse.ArgumentParser()
lp = ModelParams(ap); pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)

O_hole = torch.tensor([1.094,0,3.007], device="cuda")
O_tube = torch.tensor([3.007,0,1.094], device="cuda")

for hole_cull in [False, True]:
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    keep, st = cull_gaussians(g, 0.2, 0.3, hole_cull=hole_cull)
    k, t = st["keep"], st["total"]
    print(f"--- hole_cull={hole_cull}: keep={k}/{t} op={100*st['opacity_kept']:.1f}% "
          f"hole_culled={st['culled_hole']} ---")
    # hole ray B
    tB, oB, TB, PB = ray_march(O_hole, -O_hole, g, 0.05, 8.0, K=1000)
    print(f"  hole-ray: 1-T={1-TB[-1]:.4f} o_max={oB.max():.4f}")
    # tube ray A
    tA, oA, TA, PA = ray_march(O_tube, -O_tube, g, 0.05, 8.0, K=1000)
    rA = torus_hit_ranges(np.array([3.007,0,1.094]), np.array([0,0,0])-np.array([3.007,0,1.094]), Rm=0.55, rm=0.23)
    inside = 0.0
    for (r1,r2) in rA:
        inside += oA[(tA>=r1)&(tA<=r2)].sum()
    print(f"  tube-ray: o_in_gt={100*inside/oA.sum():.1f}%  o_max={oA.max():.4f} "
          f"1-T={1-TA[-1]:.4f} P_max_r={tA[int(np.argmax(PA))]:.3f}")

# render PSNR with and without hole cull
import subprocess
from diag_render_view import main as rendermain
