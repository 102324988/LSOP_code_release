"""E0: dissect ray_march vs hand-computed on the SAME culled model."""
import argparse, os, sys
import numpy as np, torch
GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from clean_ray_profile import quat2mat
from cull_cloud import cull_gaussians

def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap); pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--cull_s_max", type=float, default=0.2)
    ap.add_argument("--cull_d_max", type=float, default=0.3)
    args = ap.parse_args()
    ds = lp.extract(args); pipe = pp.extract(args)
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    keep, st = cull_gaussians(g, args.cull_s_max, args.cull_d_max)
    kp, tt = st["keep"], st["total"]
    print(f"[cull] keep {kp}/{tt}")

    print("xyz identical:", bool((g.get_xyz == g._xyz).all().item()))
    print("scaling identical:", bool((g.get_scaling == g._scaling).all().item()))
    print("opacity match:", bool((torch.sigmoid(g._opacity) == g.get_opacity).all().item()))
    qraw, qact = g._rotation.detach(), g.get_rotation
    print("rotation raw norm:", qraw.norm(dim=1).min().item(), "..", qraw.norm(dim=1).max().item())
    print("rotation raw==act:", bool((qraw == qact).all().item()))

    O = torch.tensor([3.007,0.0,1.094], device="cuda")
    P = torch.tensor([0.78,0.0,0.0], device="cuda")
    d = (P - O); d = d / torch.linalg.norm(d)
    S = O + d * 0.05
    xyz = g._xyz.detach()
    scl = torch.exp(g._scaling.detach())
    op  = torch.sigmoid(g._opacity.detach())
    Rm  = quat2mat(g._rotation.detach())
    diff = (S[None,:] - xyz)[:, None, :]
    diff_r = torch.einsum("nij,njk->nik", diff, Rm).squeeze(1)
    maha = ((diff_r / scl)**2).sum(-1)
    contrib = op.squeeze() * torch.exp(-0.5*maha)
    print(f"[raw]  sum={contrib.sum().item():.6f}")

    Rm2 = quat2mat(g.get_rotation)
    diff_r2 = torch.einsum("nij,njk->nik", diff, Rm2).squeeze(1)
    maha2 = ((diff_r2 / g.get_scaling.detach())**2).sum(-1)
    contrib2 = g.get_opacity.detach().squeeze() * torch.exp(-0.5*maha2)
    print(f"[get_] sum={contrib2.sum().item():.6f}")

    from clean_ray_profile import ray_march
    t, a, T, Pc = ray_march(O, P - O, g, 0.05, 2.5, K=500)
    print(f"[ray_march] a[0]={a[0]:.6f} t[0]={t[0]:.6f}")

    # per-chunk accumulation of ray_march path
    N = xyz.shape[0]; chunk = 20000
    alpha = torch.sigmoid(g.get_opacity); scale = torch.exp(g.get_scaling); Rr = quat2mat(g.get_rotation)
    acc = 0.0
    for i0 in range(0, N, chunk):
        sl = slice(i0, min(i0+chunk, N))
        df = S[None,None,:] - xyz[sl][None,:,:]  # (1,C,3)
        dfr = torch.einsum("knj,njm->knm", df, Rr[sl])
        mh = ((dfr / scale[sl][None,:,:])**2).sum(-1)
        s = (alpha[sl].T * torch.exp(-0.5*mh)).sum(-1).item()
        acc += s
        print(f"  chunk {i0}: sub-sum={s:.6f}")
    print(f"[manual chunked (get_)] a={acc:.6f}")

if __name__ == "__main__":
    main()
