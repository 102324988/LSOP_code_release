"""Decisive: render ONE Gaussian at origin with a real Camera; find its pixel.
Then test which matrix convention (FPT or FPT^T) projects origin to that pixel."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene


def ndc2Pix(x, W):
    return 0.5 * (x + 1.0) * W - 0.5


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    args = ap.parse_args()
    ds = lp.extract(args)
    pipe = pp.extract(args)

    # load real camera geometry from the trained scene (model replaced below)
    g = GaussianModel(3)
    scene = Scene(ds, g, load_iteration=4000, shuffle=False)
    cam = scene.getTrainCameras()[0]

    # Build a synthetic model: ONE Gaussian at origin
    g._xyz = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    g._features_dc = torch.ones(1, 1, 3, dtype=torch.float32, device="cuda") * 0.5
    g._features_rest = torch.zeros(1, 15, 3, dtype=torch.float32, device="cuda")
    g._opacity = torch.tensor([[20.0]], dtype=torch.float32, device="cuda")  # sigmoid~1
    g._scaling = torch.tensor([[np.log(0.03)] * 3], dtype=torch.float32, device="cuda")
    g._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    with torch.no_grad():
        out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
    rgb = out["render"].detach().cpu().permute(1, 2, 0).numpy()[:, :, :3]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    y, x = np.unravel_index(gray.argmax(), gray.shape)
    print(f"single Gaussian at origin rendered brightest at pixel ({x},{y})")
    print(f"image size {cam.image_width}x{cam.image_height}")

    W, H = cam.image_width, cam.image_height
    p = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda")
    FPT = cam.full_proj_transform
    # candidate A: kernel reads tensor row-major -> effective = FPT
    for name, M in [("FPT", FPT), ("FPT^T", FPT.T)]:
        ph = M @ p
        pp3 = ph[:3] / ph[3]
        px = ndc2Pix(pp3[0].item(), W)
        py = ndc2Pix(pp3[1].item(), H)
        print(f"  candidate {name}: proj=({pp3[0]:.3f},{pp3[1]:.3f}) pix=({px:.1f},{py:.1f})")

    # also check a point known on-screen: what pixel does the GT put origin? center.
    print("GT expectation: origin -> image center ({},{})".format(W // 2, H // 2))


if __name__ == "__main__":
    main()
