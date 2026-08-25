"""E0: pixel-level diagnosis of integrate() at specific probe points.

For a probe point seen from a camera: where does it project, what does the
rendered alpha image say at that pixel, which gaussians cover that pixel and
at what depth vs the point depth. Separates kernel semantics from model
geometry (floating gaussians).
"""
import argparse
import os
import sys

import numpy as np
import torch

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render, integrate  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cam = scene.getTrainCameras()[args.cam]
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    W, H = cam.image_width, cam.image_height

    xyz = g.get_xyz  # (N,3)
    opacity = torch.sigmoid(g.get_opacity).detach().cpu().numpy().squeeze()
    scale = torch.exp(g.get_scaling).detach().cpu().numpy()

    probes = {
        "outer_surf(0.78,0,0)": np.array([0.78, 0.0, 0.0]),
        "top_surf(0.55,0,0.23)": np.array([0.55, 0.0, 0.23]),
        "tube_center(0.55,0,0)": np.array([0.55, 0.0, 0.0]),
        "hole_center(0,0,0)": np.array([0.0, 0.0, 0.0]),
        "far_empty(0,1.2,0)": np.array([0.0, 1.2, 0.0]),
    }

    # world->pixel projection using kernel-effective full proj matrix (value^T)
    FPT = cam.full_proj_transform  # value = effective^T ; effective = value^T
    for name, P in probes.items():
        p4 = torch.tensor([P[0], P[1], P[2], 1.0], dtype=torch.float32, device="cuda")
        ph = FPT.T @ p4  # effective proj
        px = (0.5 * ph[0] / ph[3] + 0.5) * W - 0.5
        py = (0.5 * ph[1] / ph[3] + 0.5) * H - 0.5
        # kernel uses focal projection for query points; match it
        R = torch.tensor(cam.R, dtype=torch.float32, device="cuda")
        T_ = torch.tensor(cam.T, dtype=torch.float32, device="cuda")
        pv = R @ torch.tensor(P, dtype=torch.float32, device="cuda") - R @ T_
        kx = cam.focal_x * pv[0] / (pv[2] + 1e-7) + W / 2
        ky = cam.focal_y * pv[1] / (pv[2] + 1e-7) + H / 2
        px2, py2 = kx.item(), ky.item()

        with torch.no_grad():
            p = torch.from_numpy(P).float().cuda()[None, :]
            ret = integrate(p, cam, g, pipe, bg, kernel_size=ds.kernel_size)
        kernel_av = ret["alpha_integrated"].item()

        # nearest gaussians to the point in 3D
        dist = torch.linalg.norm(xyz - torch.from_numpy(P).float().cuda(), dim=1)
        near_idx = torch.argsort(dist)[:5].cpu().numpy()
        near = [(xyz[i].cpu().tolist(), float(dist[i]), float(opacity[i]))
                for i in near_idx]
        # gaussians whose 2D pixel center is within the probe's pixel
        pixR = torch.tensor(cam.R, dtype=torch.float32, device="cuda")
        pixT = torch.tensor(cam.T, dtype=torch.float32, device="cuda")
        pv_all = pixR @ xyz.T - (pixR @ pixT)[:, None]     # (3,N) cam coords
        zc = pv_all[2]
        u = cam.focal_x * pv_all[0] / (zc + 1e-7) + W / 2
        v = cam.focal_y * pv_all[1] / (zc + 1e-7) + H / 2
        in_pix = ((u.cpu() - px2).abs() < 1.5) & ((v.cpu() - py2).abs() < 1.5)
        pix_n = int(in_pix.sum())
        rd = float(pv[2].item())
        # gaussians covering this pixel and their camera depth vs ray_depth
        if pix_n > 0:
            zs = zc[in_pix].detach().cpu().numpy()
            ops = opacity[in_pix.numpy()]
            ahead = float((zs < rd).sum())
            print(f"{name}: pix_kernel=({px2:.0f},{py2:.0f}) ray_depth={rd:.2f} "
                  f"kernel_A_v={kernel_av:.3f} "
                  f"pix_gaussians={pix_n} (ahead={ahead}, behind={pix_n - int(ahead)})")
        else:
            print(f"{name}: pix=({px2:.0f},{py2:.0f}) ray_depth={rd:.2f} "
                  f"kernel_A_v={kernel_av:.3f} NO gaussian on pixel")
        print(f"    nearest 5: {near}")


if __name__ == "__main__":
    main()
