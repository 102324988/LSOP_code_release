"""Replicate GOF integrate()'s preprocess_points culling in numpy using the
Camera's actual matrices, to find why all query points are culled."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from arguments import ModelParams
from scene import GaussianModel, Scene


def transform4x3(p, M):
    """M (4,4) row-major numpy; returns M @ [p,1] then first 3."""
    return M[:3, :3] @ p + M[:3, 3]


def transform4x4(p, M):
    return M @ np.concatenate([p, [1.0]])


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    ap.add_argument("--iteration", type=int, default=4000)
    args = ap.parse_args()
    ds = lp.extract(args)
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cam = scene.getTrainCameras()[0]

    W, H = cam.image_width, cam.image_height
    WVT = cam.world_view_transform.cpu().numpy()     # (4,4)
    FPT = cam.full_proj_transform.cpu().numpy()      # (4,4)
    print(f"W={W} H={H} fx={cam.focal_x:.1f} fy={cam.focal_y:.1f}")
    print("world_view_transform =\n", np.round(WVT, 4))
    print("full_proj_transform =\n", np.round(FPT, 4))

    # Reconstruct getWorld2View2(R,T) numpy result for comparison
    Rt = np.zeros((4, 4), dtype=np.float32)
    R, T = np.asarray(cam.R), np.asarray(cam.T)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    print("numpy Rt[:3,:3]=R.T, Rt[:3,3]=T == transposed WVT? ",
          np.allclose(WVT.T, Rt, atol=1e-4))

    pts = {
        "above_torus(0,0,0.6)": [0.0, 0.0, 0.6],
        "outer_equator(0.95,0,0)": [0.95, 0.0, 0.0],
        "origin(0,0,0)": [0.0, 0.0, 0.0],
    }
    focal_x, focal_y = cam.focal_x, cam.focal_y
    for name, p in pts.items():
        p = np.array(p, dtype=np.float64)
        p_view = transform4x3(p, WVT)                  # like preprocess_points kernel
        p_hom = transform4x4(p, FPT)                   # like in_frustum
        p_proj = p_hom[:3] / (p_hom[3] + 1e-7)
        u = focal_x * p_view[0] / (p_view[2] + 1e-7) + W / 2
        v = focal_y * p_view[1] / (p_view[2] + 1e-7) + H / 2
        culls = []
        if p_view[2] <= 0.2:
            culls.append(f"near(p_view.z={p_view[2]:.3f})")
        if not (-1 <= p_proj[0] <= 1 and -1 <= p_proj[1] <= 1 and p_proj[2] <= 1):
            culls.append(f"frustum(p_proj={np.round(p_proj,3)})")
        if not (0 <= u < W and 0 <= v < H):
            culls.append(f"image(pix=({u:.1f},{v:.1f}))")
        print(f"\n  {name:>22s}: p_view=({p_view[0]:.3f},{p_view[1]:.3f},{p_view[2]:.3f}) "
              f"pix=({u:.1f},{v:.1f}) p_proj=({p_proj[0]:.3f},{p_proj[1]:.3f},{p_proj[2]:.3f})")
        print(f"    -> culled by: {culls if culls else 'NOT CULLED'}")


if __name__ == "__main__":
    main()
