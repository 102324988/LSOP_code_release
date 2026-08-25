"""E0: are the hole-ghost gaussians white (== bg)? Check SH color of
gaussians inside the hole cylinder vs on the tube wall."""
import argparse, os, sys
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
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

pos = g._xyz.detach().cpu().numpy()
scl = torch.exp(g._scaling.detach()).cpu().numpy()
op  = torch.sigmoid(g._opacity.detach()).cpu().numpy().squeeze()
fdc = g._features_dc.detach().cpu().numpy().squeeze()  # (N,3) SH DC, 0.5*color/255 offset

def sh2rgb(f):
    # features_dc stored as (SH_C0 * color/255 - 0.5); 3DGS: rgb = 0.5 + 0.28209479 * f
    return np.clip(0.5 + 0.28209479 * f, 0, 1)

rxy = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
dtube = np.abs(np.sqrt((rxy-0.55)**2 + pos[:,2]**2) - 0.23)
in_hole = (rxy < 0.32) & (np.abs(pos[:,2]) < 0.4) & (dtube > 0.15)
on_wall = (dtube < 0.05) & (rxy > 0.3)

hole_rgb = sh2rgb(fdc[in_hole])
wall_rgb = sh2rgb(fdc[on_wall])
print(f"hole gaussians: {in_hole.sum()}   wall gaussians: {on_wall.sum()}")
print(f"HOLE mean rgb: {hole_rgb.mean(axis=0).round(3)}  (white=1,1,1)")
print(f"WALL mean rgb: {wall_rgb.mean(axis=0).round(3)}")
print(f"hole op: p50={np.median(op[in_hole]):.3f} mean={op[in_hole].mean():.3f}")
print(f"wall op: p50={np.median(op[on_wall]):.3f} mean={op[on_wall].mean():.3f}")
print(f"hole scale: p50={np.median(scl[in_hole],axis=0).round(4)}")
# fraction of hole gaussians with near-white color
near_white = (hole_rgb > 0.85).all(axis=1).mean()
print(f"hole frac near-white(>0.85): {near_white:.2f}   wall frac: {(wall_rgb>0.85).all(axis=1).mean():.2f}")
