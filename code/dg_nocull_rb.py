"""E0: random-bg model WITHOUT any culling — is it natively clean?
Check: hole-ray 1-T (should be ~0), tube-ray o-in-GT, ghost gaussians count,
max scale (should have no giant 15.58 floaters).
"""
import argparse, os, sys
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from clean_ray_profile import ray_march, torus_hit_ranges

ap = argparse.ArgumentParser()
lp = ModelParams(ap); pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)
g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
N = g._xyz.shape[0]
scl = torch.exp(g._scaling.detach())
smax = scl.max(1).values
print(f"[model] gaussians={N}  scale max={smax.max().item():.3f} p99={smax.quantile(0.99).item():.3f}")

# ghost count: inside hole cylinder (rxy<0.32, |z|<0.23)
pos = g._xyz.detach().cpu().numpy()
rxy = np.sqrt(pos[:,0]**2+pos[:,1]**2)
in_hole = (rxy < 0.32) & (np.abs(pos[:,2]) < 0.23)
print(f"[ghost] gaussians inside hole cylinder: {in_hole.sum()}/{N}")

# hole-ray (el70 -> center): expect 1-T ~ 0
O = torch.tensor([1.094,0,3.007], device="cuda")
T0 = torch.tensor([0,0,0], device="cuda")
t, o, Tt, P = ray_march(O, T0-O, g, 0.05, 8.0, K=1500)
print(f"[hole-ray] 1-T={1-Tt[-1]:.4f}  o_max={o.max():.4f}")

# tube-ray (el20 -> center): o concentrated in GT range
O2 = torch.tensor([3.007,0,1.094], device="cuda")
t2, o2, T2, P2 = ray_march(O2, T0-O2, g, 0.05, 8.0, K=1500)
rA = torus_hit_ranges([3.007,0,1.094], np.array([0,0,0])-np.array([3.007,0,1.094]), Rm=0.55, rm=0.23)
inside = sum(o2[(t2>=r1)&(t2<=r2)].sum() for (r1,r2) in rA)
print(f"[tube-ray] o_in_gt={100*inside/o2.sum():.1f}%  1-T={1-T2[-1]:.4f}")
