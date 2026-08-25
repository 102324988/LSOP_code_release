"""E0: invert ray profiles back to surface hits, compare vs GT torus.
For each ray: reconstruct o(r); surface hits = local maxima of o.
Map back to 3D points; error = |dist_to_torus_surface(point) - 0|.
This is Path A's core claim: profiles are invertible to geometry."""
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
ap.add_argument("--nrays", type=int, default=200)
args = ap.parse_args()
ds = lp.extract(args); pipe = pp.extract(args)
g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
keep, st = cull_gaussians(g, 0.2, 0.3, hole_cull=True)
print(f"[cull] keep {st['keep']}/{st['total']}")

def dist_torus(p, R=0.55, r=0.23):
    rxy = np.sqrt(p[...,0]**2 + p[...,1]**2)
    return np.abs(np.sqrt((rxy-R)**2 + p[...,2]**2) - r)

# generate rays from 3 elevation rings aimed at a grid of targets near the torus
tgt_r = np.linspace(0.0, 0.78, 13)          # radial targets
tgt_z = np.linspace(-0.30, 0.30, 13)        # heights
# camera positions: el 20/45/70, 6 azimuths
els = [20, 45, 70]
azs = np.linspace(0, 360, 6, endpoint=False)
Rcam = 3.2

errs = []          # |profile-recon surface - GT torus surface|
hits_total = 0
for el_deg in els:
    el = np.deg2rad(el_deg)
    for az_deg in azs:
        az = np.deg2rad(az_deg)
        cam = np.array([Rcam*np.cos(el)*np.cos(az), Rcam*np.cos(el)*np.sin(az), Rcam*np.sin(el)])
        for tr in tgt_r:
            for tz in tgt_z:
                tgt = np.array([tr, 0.0, tz])  # note: target in XZ-plane slice
                # rotate target by az so we probe a spread of points
                ct, stt = np.cos(az), np.sin(az)
                tgtr = np.array([tr*ct, tr*stt, tz])
                dvec = tgtr - cam
                t, o, Tt, P = ray_march(torch.tensor(cam,dtype=torch.float32,device="cuda"),
                                        torch.tensor(dvec,dtype=torch.float32,device="cuda"),
                                        g, 0.02, 7.0, K=1200)
                # invert: surface hits = o local maxima above threshold
                thr = max(0.01, 0.05*o.max())
                pts = []
                for i in range(1, len(o)-1):
                    if o[i] > o[i-1] and o[i] >= o[i+1] and o[i] > thr:
                        pt = cam + (dvec/np.linalg.norm(dvec))*t[i]
                        # only count if inside scene extent and not in hole
                        if np.abs(pt).max() < 1.2:
                            pts.append(pt)
                for pt in pts:
                    errs.append(dist_torus(pt))
                    hits_total += 1

errs = np.array(errs)
print(f"rays={len(els)*len(azs)*len(tgt_r)*len(tgt_z)}, surface-hits={hits_total}")
if hits_total > 0:
    print(f"hit-to-GT-surface dist:  p50={np.median(errs):.4f}  "
          f"p90={np.percentile(errs,90):.4f}  mean={errs.mean():.4f}  "
          f"frac<0.1={np.mean(errs<0.1):.3f}  frac<0.2={np.mean(errs<0.2):.3f}")
