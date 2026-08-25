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

O = torch.tensor([3.007,0.0,1.094], device="cuda")
P = torch.tensor([0.78,0.0,0.0], device="cuda")
d = (P - O); d = d / torch.linalg.norm(d)
S = O + d * 0.05
xyz = g._xyz.detach()                       # (N,3)
scl = torch.exp(g._scaling.detach())        # (N,3) log-scale -> scale
op  = torch.sigmoid(g._opacity.detach())    # (N,1)

Rm_raw = quat2mat(g._rotation.detach())     # raw quat (NOT normalized)
Rm_act = quat2mat(g.get_rotation)           # normalized quat
print("R raw vs act identical:", bool((Rm_raw == Rm_act).all().item()))
print("|diff| between raw/act R:", (Rm_raw-Rm_act).abs().max().item())

diff = (S[None,:] - xyz)                    # (N,3)
print("diff norm range:", diff.norm(dim=1).min().item(), "..", diff.norm(dim=1).max().item())

def contrib_via(diff_r):
    maha = ((diff_r / scl)**2).sum(-1)
    return op.squeeze() * torch.exp(-0.5*maha)

da = torch.einsum("nij,njk->nik", diff[:,None,:], Rm_act).squeeze(1)
ma = ((da / scl)**2).sum(-1)
print(f"[einsum nik] act: maha min={ma.min().item():.3f} p50={ma.median().item():.3f} max={ma.max().item():.3f} contrib sum={contrib_via(da).sum().item():.6f}")

db = torch.einsum("knj,njm->knm", diff[None,:,:], Rm_act).squeeze(0)
mb = ((db / scl)**2).sum(-1)
print(f"[einsum knm] act: maha min={mb.min().item():.3f} p50={mb.median().item():.3f} max={mb.max().item():.3f} contrib sum={contrib_via(db).sum().item():.6f}")
print("   |nik - knm| maha:", (ma-mb).abs().max().item())

top = torch.argsort(op.squeeze()*torch.exp(-0.5*ma), descending=True)[:5]
for i in top.tolist():
    print(f"  top idx={i} maha={ma[i].item():.3f} scale={scl[i].cpu().numpy().round(4)} dist={diff[i].norm().item():.3f} op={op[i].item():.3f} pos={xyz[i].cpu().numpy().round(3)}")
