"""Manual projection of query points with GOF Camera matrices (cam 0)."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from arguments import ModelParams
from scene import GaussianModel, Scene


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
    print(f"cam0: W={W} H={H} fx={cam.focal_x:.1f} fy={cam.focal_y:.1f}")
    print(f"cam0 FoVx={cam.FoVx*180/3.14159:.2f} FoVy={cam.FoVy*180/3.14159:.2f}")
    print(f"cam0 camera_center={cam.camera_center.cpu().numpy()}")
    print(f"cam0 R={cam.R}")  # world->cam
    print(f"cam0 T={cam.T}")

    pts = {
        "above_torus(0,0,0.6)": [0.0, 0.0, 0.6],
        "outer_equator(0.95,0,0)": [0.95, 0.0, 0.0],
        "cam_center_itself": cam.camera_center.cpu().numpy().tolist(),
        "origin(0,0,0)": [0.0, 0.0, 0.0],
    }
    for name, p in pts.items():
        p = np.array(p, dtype=np.float64)
        # camera-space: pc = R @ (p - T)  (GOF: R is world->cam, T is cam center)
        pc = cam.R @ (p - np.asarray(cam.T))
        z = pc[2]
        u = cam.focal_x * pc[0] / z + W / 2
        v = cam.focal_y * pc[1] / z + H / 2
        inside = 0 <= u < W and 0 <= v < H and z > 0.2
        print(f"  {name:>28s}: camz={z:7.3f} pix=({u:7.1f},{v:7.1f}) inside={inside}")


if __name__ == "__main__":
    main()
