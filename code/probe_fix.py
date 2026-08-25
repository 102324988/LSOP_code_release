"""Verify corrected W2C matrices: single Gaussian at origin should render at image
center (400,300) with a Camera whose matrices are overridden."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import getProjectionMatrix


def correct_matrices(cam):
    """Build proper W2C + P@W2C for this camera; return contiguous tensors
    whose kernel-effective value equals the intended matrix."""
    R = torch.tensor(cam.R, dtype=torch.float32)          # world->cam
    T = torch.tensor(cam.T, dtype=torch.float32)          # camera center (world)
    W2C = torch.zeros(4, 4, dtype=torch.float32)
    W2C[:3, :3] = R
    W2C[:3, 3] = -R @ T
    W2C[3, 3] = 1.0
    P = getProjectionMatrix(cam.znear, cam.zfar, cam.FoVx, cam.FoVy)
    wvt = W2C.transpose(0, 1).contiguous().cuda()
    fpt = (P @ W2C).transpose(0, 1).contiguous().cuda()
    return wvt, fpt


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)
    g = GaussianModel(3)
    scene = Scene(ds, g, load_iteration=4000, shuffle=False)
    cam = scene.getTrainCameras()[0]

    # synthetic single Gaussian at origin
    g._xyz = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    g._features_dc = torch.ones(1, 1, 3, dtype=torch.float32, device="cuda") * 0.5
    g._features_rest = torch.zeros(1, 15, 3, dtype=torch.float32, device="cuda")
    g._opacity = torch.tensor([[20.0]], dtype=torch.float32, device="cuda")
    g._scaling = torch.tensor([[np.log(0.08)] * 3], dtype=torch.float32, device="cuda")
    g._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    W, H = cam.image_width, cam.image_height
    for label in ["BROKEN(original)", "CORRECTED"]:
        if label == "CORRECTED":
            wvt, fpt = correct_matrices(cam)
            cam.world_view_transform = wvt
            cam.full_proj_transform = fpt
        with torch.no_grad():
            out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
        rgb = out["render"].detach().cpu().permute(1, 2, 0).numpy()[:, :, :3]
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if gray.max() < 0.01:
            print(f"{label}: Gaussian not visible")
            continue
        yy, xx = np.unravel_index(gray.argmax(), gray.shape)
        print(f"{label}: origin Gaussian at pixel ({xx},{yy})  (expect ~400,300)")


if __name__ == "__main__":
    main()
