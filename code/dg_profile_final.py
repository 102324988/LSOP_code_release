"""E0: final profile validation on culled+hole-culled model.
Rays: A tube(el20->center), B hole(el70->center), C tube-top(el70->top),
      D hole-edge(el20->through-hole), E miss(el20->beside object).
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
keep, st = cull_gaussians(g, 0.2, 0.3, hole_cull=True)
print(f"[cull] keep {st['keep']}/{st['total']} hole_culled={st['culled_hole']}")

rays = [
    ("A tube(enter/exit)",   np.array([3.007,0,1.094]), np.array([0,0,0])),
    ("B hole(should be T~1)",np.array([1.094,0,3.007]), np.array([0,0,0])),
    ("C tube-top(el70)",     np.array([1.094,0,3.007]), np.array([0.55,0,0])),
    ("D hole-rim(el20)",     np.array([3.007,0,1.094]), np.array([0.55,0,0.23])),
    ("E miss (empty)",       np.array([3.007,0,1.094]), np.array([1.2,0,0])),
]
for name, O, Tgt in rays:
    dvec = Tgt - O
    t, o, Tt, P = ray_march(torch.tensor(O,dtype=torch.float32,device="cuda"),
                            torch.tensor(dvec,dtype=torch.float32,device="cuda"),
                            g, 0.05, 8.0, K=1500)
    ranges = torus_hit_ranges(O, dvec, Rm=0.55, rm=0.23)
    j = int(np.argmax(P))
    inside = 0.0
    for (r1,r2) in ranges:
        inside += o[(t>=r1)&(t<=r2)].sum()
    frac = 100*inside/o.sum() if o.sum()>0 else float("nan")
    print(f"\n=== {name} ===")
    print(f"  GT tube range: {[(round(a,3),round(b,3)) for a,b in ranges]}")
    print(f"  o_max={o.max():.4f} o_sum={o.sum():.4f}  in-GT={frac:.1f}%")
    print(f"  T_final={Tt[-1]:.4f} 1-T={1-Tt[-1]:.4f}  P_max={P[j]:.4f}@r={t[j]:.3f}")
    # peak count inside GT (double-peak check for A)
    peaks = [t[i] for i in range(1,len(P)-1) if P[i]>P[i-1] and P[i]>=P[i+1] and P[i]>0.05]
    print(f"  P peaks(r>0.05): {[round(float(r),3) for r in peaks]}")
