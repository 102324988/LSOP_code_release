"""E0: diagnose sphere inversion failure — is the 3DGS opacity field
hollow (density on the shell) or solid (density filling the ball)?

Hypothesis: a sphere is a solid disk from every view, so 3DGS has no
inside/outside constraint and packs gaussians throughout the interior;
ray-march profiles then show spurious interior peaks and local-max
inversion hits interior points -> large GT-surface error + many hits.

Prints:
  1. radial hit distribution for the eval ray grid (like eval_matrix2)
  2. one center-crossing ray profile: peak locations vs shell radius
"""
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

ap = argparse.ArgumentParser()
lp = ModelParams(ap)
pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
args = ap.parse_args()
ds = lp.extract(args)
pipe = pp.extract(args)

g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
pos = g._xyz.detach().cpu().numpy()
r_g = np.linalg.norm(pos, axis=1)
print(f"gaussians={len(r_g)}  gaussian radius p50={np.median(r_g):.3f} "
      f"p90={np.percentile(r_g,90):.3f} max={r_g.max():.3f}")

els = [20, 45, 70]
azs = np.linspace(0, 360, 6, endpoint=False)
Rcam = 3.2
tgt_r = np.linspace(0.0, 0.78, 13)
tgt_z = np.linspace(-0.30, 0.30, 13)
rr = []
for el_deg in els:
    el = np.deg2rad(el_deg)
    for az_deg in azs:
        az = np.deg2rad(az_deg)
        cam = np.array([Rcam*np.cos(el)*np.cos(az),
                        Rcam*np.cos(el)*np.sin(az), Rcam*np.sin(el)])
        ct, stt = np.cos(az), np.sin(az)
        for tr in tgt_r:
            for tz in tgt_z:
                tgtr = np.array([tr*ct, tr*stt, tz])
                dvec = tgtr - cam
                t, o, Tt, P = ray_march(torch.tensor(cam, dtype=torch.float32, device="cuda"),
                                        torch.tensor(dvec, dtype=torch.float32, device="cuda"),
                                        g, 0.02, 7.0, K=1200)
                thr = max(0.01, 0.05*o.max())
                for i in range(1, len(o)-1):
                    if o[i] > o[i-1] and o[i] >= o[i+1] and o[i] > thr:
                        pt = cam + (dvec/np.linalg.norm(dvec))*t[i]
                        if np.abs(pt).max() < 1.2:
                            rr.append(np.linalg.norm(pt))
rr = np.array(rr)
print(f"hits={len(rr)}  hit-radius (from center) p10={np.percentile(rr,10):.3f} "
      f"p50={np.percentile(rr,50):.3f} p90={np.percentile(rr,90):.3f} "
      f"(GT shell radius=0.95)")
frac_inner = np.mean(rr < 0.7)
print(f"fraction of hits with radius<0.7 (interior, not on shell): {frac_inner:.3f}")

# one center-crossing ray from el=20,az=0 camera
cam = np.array([Rcam*np.cos(np.deg2rad(20)), 0.0, Rcam*np.sin(np.deg2rad(20))])
dvec = -cam
t, o, Tt, P = ray_march(torch.tensor(cam, dtype=torch.float32, device="cuda"),
                        torch.tensor(dvec, dtype=torch.float32, device="cuda"),
                        g, 0.02, 7.0, K=1200)
peaks = [(t[i], o[i]) for i in range(1, len(o)-1)
         if o[i] > o[i-1] and o[i] >= o[i+1] and o[i] > max(0.01, 0.05*o.max())]
print("\ncenter ray peaks (t, o):")
for tt, oo in peaks:
    pt = cam + (-cam/np.linalg.norm(cam))*tt
    print(f"  t={tt:.3f} o={oo:.3f} hit-dist-from-center={np.linalg.norm(pt):.3f} (shell=0.95)")
