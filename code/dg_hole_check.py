"""E0: why is the hole ray not empty after cull?
Locate gaussians contributing alpha near r~3.24 on hole ray (B).
GT torus: major R=0.55, minor r=0.23. Hole = dist-to-z-axis < 0.32.
"""
import argparse, os, sys
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from clean_ray_profile import quat2mat
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

O = torch.tensor([1.094,0,3.007], device="cuda")
Tgt = torch.tensor([0,0,0], device="cuda")
d = Tgt - O; d = d / torch.linalg.norm(d)

xyz = g._xyz.detach()
scl = torch.exp(g._scaling.detach())
op  = torch.sigmoid(g._opacity.detach())
Rm  = quat2mat(g.get_rotation)

# samples near the o peak at r=3.24
for r in [3.10, 3.24, 3.30, 3.45]:
    S = O + d * r
    diff = (S[None,:] - xyz)
    diff_r = torch.einsum("nij,njk->nik", diff[:,None,:], Rm).squeeze(1)
    maha = ((diff_r / scl)**2).sum(-1)
    contrib = op.squeeze() * torch.exp(-0.5*maha)
    top = torch.argsort(contrib, descending=True)[:5]
    print(f"\n=== sample r={r:.2f} point={S.cpu().numpy().round(3)} ===")
    print(f"  contrib sum={contrib.sum().item():.4f}")
    for i in top.tolist():
        pos = xyz[i].cpu().numpy()
        rxy = np.sqrt(pos[0]**2+pos[1]**2)
        dtube = np.sqrt((rxy-0.55)**2+pos[2]**2)
        loc = "IN-tube" if dtube<0.23 else ("hole(center)" if rxy<0.32 else "outside")
        print(f"    idx={i} c={contrib[i].item():.4f} op={op[i].item():.3f} "
              f"scale={scl[i].cpu().numpy().round(4)} dist={np.linalg.norm(pos-S.cpu().numpy()):.3f} "
              f"pos={pos.round(3)} rxy={rxy:.3f} tube-d={dtube:.3f} [{loc}]")

# how many gaussians are inside the hole (rxy<0.32, |z|<0.4) at all?
pos = xyz.cpu().numpy()
rxy = np.sqrt(pos[:,0]**2+pos[:,1]**2)
in_hole = (rxy < 0.32) & (np.abs(pos[:,2]) < 0.4)
print(f"\n[cloud] gaussians inside hole cylinder: {in_hole.sum()}/{len(pos)}")
if in_hole.any():
    for i in np.where(in_hole)[0][:10]:
        print(f"    pos={pos[i].round(3)} scale={scl[i].cpu().numpy().round(4)} op={op[i].item():.3f}")
