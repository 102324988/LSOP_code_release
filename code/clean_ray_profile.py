"""E0: clean 3DGS ray-march profiles (independent of the GOF field kernel).

Path A core: from a trained 3DGS model, a ray through the volume should yield a
physically meaningful occupancy profile P(r) = T(r)*o(r):
  o(r)   = summed 3D gaussian density at point O + r*d
  T(r)   = exp(-integral o)
  P(r)   = T(r)*(1 - exp(-o(r)*dr))   (stopping density along the ray)

Volume density of a gaussian is normalized so its integrated opacity equals its
sigmoid(opacity):  o_j(s) = alpha_j * exp(-0.5*maha) / ( (2pi)^1.5 * prod(sigma) ).

Expected on the GT torus (major R=0.55, minor r=0.23, in XY plane):
  ray through the tube   -> P has two peaks (enter / exit)
  ray through the hole   -> o ~ 0 everywhere, T ~ 1
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)

from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from cull_cloud import cull_gaussians  # noqa: E402


def quat2mat(q):
    """(N,4) [w,x,y,z] -> (N,3,3) rotation matrices (3DGS convention)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros(N, 3, 3, dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@torch.no_grad()
def ray_march(origin, d, g, t_near, t_far, K=400, chunk=20000):
    """Return o, T, P along the ray (cuda).

    Matches 3DGS rasterization semantics (NOT volumetric density):
    each gaussian contributes alpha(s) = sigmoid(opacity) * exp(-0.5*maha)
    at sample s (maha in the gaussian's principal frame, exp(-0.5)<=1).
    Accumulated transmittance T *= (1 - alpha), P(r)=T*alpha.
    """
    d = d / torch.linalg.norm(d)
    t = torch.linspace(t_near, t_far, K, device=g.get_xyz.device)
    S = origin[None, :] + d[None, :] * t[:, None]          # (K,3)
    xyz = g.get_xyz
    N = xyz.shape[0]
    # NOTE: get_opacity / get_scaling already apply the activation
    # (sigmoid / exp); re-applying them double-activates and explodes.
    alpha = g.get_opacity                                   # (N,1) sigmoid(_opacity)
    scale = g.get_scaling                                   # (N,3) exp(_scaling)
    R = quat2mat(g.get_rotation)                            # (N,3,3) normalized quat
    a = torch.zeros(K, device=S.device)
    for i0 in range(0, N, chunk):
        sl = slice(i0, min(i0 + chunk, N))
        diff = S[:, None, :] - xyz[sl][None, :, :]          # (K,C,3)
        diff_r = torch.einsum("knj,njm->knm", diff, R[sl])  # rotate into principal frame
        maha = ((diff_r / scale[sl][None, :, :]) ** 2).sum(-1)  # (K,C)
        a += (alpha[sl].T * torch.exp(-0.5 * maha)).sum(-1)
    a = torch.clamp(a, max=0.99)
    T = torch.cumprod(1 - a, dim=0)                          # front-to-back
    P = T * a
    return t.cpu().numpy(), a.cpu().numpy(), T.cpu().numpy(), P.cpu().numpy()


def torus_hit_ranges(origin, d, Rm=0.67, rm=0.28, K=4000):
    """Numerical torus intersection along ray: return list of (r_in, r_out)."""
    d = np.asarray(d, dtype=np.float64)
    d = d / np.linalg.norm(d)
    rs = np.linspace(0.0, 8.0, K)
    pts = np.asarray(origin) + rs[:, None] * d[None, :]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    val = (np.sqrt(x * x + y * y) - Rm) ** 2 + z * z - rm * rm
    inside = val <= 0
    ranges = []
    prev = False
    r_in = None
    for i, ins in enumerate(inside):
        if ins and not prev:
            r_in = rs[i]
        elif (not ins) and prev and r_in is not None:
            ranges.append((r_in, rs[i]))
        prev = ins
    if prev and r_in is not None:
        ranges.append((r_in, rs[-1]))
    return ranges


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--out", default="/tmp/ray_profiles.png")
    ap.add_argument("--cull_s_max", type=float, default=0.0, help="0 = no cull")
    ap.add_argument("--cull_d_max", type=float, default=0.0)
    ap.add_argument("--hole_cull", action="store_true", help="drop gaussians in torus hole")
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    if args.cull_s_max > 0:
        keep, st = cull_gaussians(g, args.cull_s_max, args.cull_d_max,
                                  hole_cull=args.hole_cull)
        print(f"[cull] keep {st['keep']}/{st['total']} "
              f"({100*st['keep_frac']:.1f}%) opacity {100*st['opacity_kept']:.1f}%")
    print(f"[march] gaussians={g.get_xyz.shape[0]}")

    rays = {
        "A: eye(el20)->through tube": (np.array([3.007, 0.0, 1.094]),
                                       np.array([0.0, 0.0, 0.0]) - np.array([3.007, 0.0, 1.094])),
        "B: eye(el70)->through hole": (np.array([1.094, 0.0, 3.007]),
                                       np.array([0.0, 0.0, 0.0]) - np.array([1.094, 0.0, 3.007])),
        "C: eye(el70)->through tube top": (np.array([1.094, 0.0, 3.007]),
                                           np.array([0.55, 0.0, 0.0]) - np.array([1.094, 0.0, 3.007])),
    }

    fig, axes = plt.subplots(len(rays), 1, figsize=(11, 3 * len(rays)), sharex=False)
    for ax, (name, (O, dvec)) in zip(axes, rays.items()):
        t, o, T, P = ray_march(torch.tensor(O, dtype=torch.float32, device="cuda"),
                               torch.tensor(dvec, dtype=torch.float32, device="cuda"), g,
                               0.1, 6.0)
        ranges = torus_hit_ranges(O, dvec, Rm=0.55, rm=0.23)
        ax.plot(t, o, label="o(r) density", lw=1.2)
        ax.plot(t, T, label="T(r) transmittance", lw=1.2)
        ax.plot(t, P, label="P(r) stopping", lw=1.8)
        for (r1, r2) in ranges:
            ax.axvspan(r1, r2, color="tab:red", alpha=0.25, label="GT tube (in/out)" if r1 == ranges[0][0] else "")
            ax.axvline(r1, color="tab:red", ls=":", lw=0.8)
            ax.axvline(r2, color="tab:red", ls=":", lw=0.8)
        ax.set_title(f"{name}  |  GT tube r-ranges: {[(round(a,2), round(b,2)) for a, b in ranges]}")
        ax.set_ylabel("opacity / T")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("r along ray")
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"[march] figure -> {args.out}")


if __name__ == "__main__":
    main()
