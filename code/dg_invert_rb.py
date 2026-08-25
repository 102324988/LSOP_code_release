"""E0: inversion on random-bg model, NO cull. Compare vs white-bg (culled 0.076 median)."""
import argparse, os, sys
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from clean_ray_profile import ray_march

ap = argparse.ArgumentParser()
lp = ModelParams(ap); pp = PipelineParams(ap)
ap.add_argument("--iteration", type=int, default=6000)
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)
g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)

def dist_torus(p, R=0.55, r=0.23):
    rxy = np.sqrt(p[...,0]**2 + p[...,1]**2)
    return np.abs(np.sqrt((rxy-R)**2 + p[...,2]**2) - r)

els = [20, 45, 70]
azs = np.linspace(0, 360, 6, endpoint=False)
Rcam = 3.2
tgt_r = np.linspace(0.0, 0.78, 13)
tgt_z = np.linspace(-0.30, 0.30, 13)

errs = []
hits = 0
for el_deg in els:
    el = np.deg2rad(el_deg)
    for az_deg in azs:
        az = np.deg2rad(az_deg)
        cam = np.array([Rcam*np.cos(el)*np.cos(az), Rcam*np.cos(el)*np.sin(az), Rcam*np.sin(el)])
        ct, stt = np.cos(az), np.sin(az)
        for tr in tgt_r:
            for tz in tgt_z:
                tgtr = np.array([tr*ct, tr*stt, tz])
                dvec = tgtr - cam
                t, o, Tt, P = ray_march(torch.tensor(cam,dtype=torch.float32,device="cuda"),
                                        torch.tensor(dvec,dtype=torch.float32,device="cuda"),
                                        g, 0.02, 7.0, K=1200)
                thr = max(0.01, 0.05*o.max())
                for i in range(1, len(o)-1):
                    if o[i] > o[i-1] and o[i] >= o[i+1] and o[i] > thr:
                        pt = cam + (dvec/np.linalg.norm(dvec))*t[i]
                        if np.abs(pt).max() < 1.2:
                            errs.append(dist_torus(pt)); hits += 1
errs = np.array(errs)
print(f"rays=3042 hits={hits}")
print(f"dist: p50={np.median(errs):.4f} p90={np.percentile(errs,90):.4f} "
      f"mean={errs.mean():.4f} frac<0.1={np.mean(errs<0.1):.3f} frac<0.2={np.mean(errs<0.2):.3f}")
