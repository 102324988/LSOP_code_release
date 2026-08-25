"""E0: numeric check of fixed ray profiles on culled torus model.
Expected:
  ray through tube (A)  -> P has two peaks inside GT tube r-range
  ray through hole (B)  -> o ~ 0 everywhere, T ~ 1, P ~ 0
  ray through tube top (C) -> peaks near GT tube r-range
"""
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
g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
keep, st = cull_gaussians(g, 0.2, 0.3)
print(f"[cull] keep {st['keep']}/{st['total']}")

rays = [
    ("A tube(el20->center)", np.array([3.007,0,1.094]), np.array([0,0,0])),
    ("B hole(el70->center)", np.array([1.094,0,3.007]), np.array([0,0,0])),
    ("C tubetop(el70->top)", np.array([1.094,0,3.007]), np.array([0.55,0,0])),
]
for name, O, T in rays:
    dvec = T - O
    t, o, Tt, P = ray_march(torch.tensor(O,dtype=torch.float32,device="cuda"),
                            torch.tensor(dvec,dtype=torch.float32,device="cuda"),
                            g, 0.05, 8.0, K=1000)
    ranges = torus_hit_ranges(O, dvec, Rm=0.55, rm=0.23)
    print(f"\n=== {name} ===")
    print(f"  GT tube r-range: {[(round(a,3),round(b,3)) for a,b in ranges]}")
    print(f"  o: max={o.max():.4f} sum={o.sum():.4f}  nonzero frac={np.mean(o>1e-3):.3f}")
    print(f"  T final={Tt[-1]:.4f}  1-T={1-Tt[-1]:.4f}")
    # peaks of P
    j = np.argmax(P)
    print(f"  P max={P.max():.4f} at r={t[j]:.3f}")
    # where does o concentrate? top-k samples
    idx = np.argsort(o)[-8:][::-1]
    print("  top-o at r:", [round(float(t[i]),3) for i in idx])
    # fraction of o inside GT tube range
    if ranges:
        tot = o.sum()
        inside = 0.0
        for (r1,r2) in ranges:
            inside += o[(t>=r1)&(t<=r2)].sum()
        print(f"  o inside GT tube: {100*inside/tot:.1f}%")
