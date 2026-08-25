"""E0: isolate "kernel integrate semantics" vs "model quality".

For probe points, compare:
  * kernel alpha_integrated (GOF integrate() output)
  * clean PyTorch ray-march accumulated opacity to the same point (1 - T)
A big gap means the kernel over-counts; a similar value means the 3DGS cloud
itself has density at the point (model issue).
"""
import argparse
import os
import sys

import numpy as np
import torch

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import integrate  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from clean_ray_profile import quat2mat, ray_march  # noqa: E402
from cull_cloud import cull_gaussians  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--cull_s_max", type=float, default=0.0, help="0 = no cull")
    ap.add_argument("--cull_d_max", type=float, default=0.0)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    if args.cull_s_max > 0:
        keep, st = cull_gaussians(g, args.cull_s_max, args.cull_d_max)
        print(f"[cull] keep {st['keep']}/{st['total']} "
              f"({100*st['keep_frac']:.1f}%) opacity {100*st['opacity_kept']:.1f}%")
    cam = scene.getTrainCameras()[args.cam]
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    O = torch.tensor(cam.camera_center, dtype=torch.float32, device="cuda")

    # GT torus: R=0.55 r=0.23, tube around z-axis
    probes = {
        "outer_surf(0.78,0,0)": np.array([0.78, 0.0, 0.0]),
        "top_surf(0.55,0,0.23)": np.array([0.55, 0.0, 0.23]),
        "tube_center(0.55,0,0)": np.array([0.55, 0.0, 0.0]),
        "hole_center(0,0,0)": np.array([0.0, 0.0, 0.0]),
        "above(0,0,0.6)": np.array([0.0, 0.0, 0.6]),
        "far_empty(0,1.2,0)": np.array([0.0, 1.2, 0.0]),
    }

    print(f"camera {args.cam} center={O.tolist()}")
    print(f"{'point':>22s} | {'kernel A_v':>10s} | {'clean 1-T':>10s} | {'r':>6s}")
    with torch.no_grad():
        for name, P in probes.items():
            p = torch.from_numpy(P).float().cuda()[None, :]
            ret = integrate(p, cam, g, pipe, bg, kernel_size=ds.kernel_size)
            kernel_av = ret["alpha_integrated"].item()
            r = float(torch.linalg.norm(p[0] - O))
            dvec = p[0] - O
            t, o, T, Pc = ray_march(O, dvec, g, 0.05, r, K=300)
            clean_av = float(1 - T[-1])
            print(f"{name:>22s} | {kernel_av:10.4f} | {clean_av:10.4f} | {r:6.2f}")


if __name__ == "__main__":
    main()
