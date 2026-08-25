"""Render cam0 with trained model; compare object position to GT view_0000.png."""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene


def bbox_of_bright(rgb, thr=0.2):
    """Where are bright (object) pixels? GT bg is dark, object bright."""
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    m = gray > thr
    if m.sum() == 0:
        return None, None, 0
    ys, xs = np.nonzero(m)
    return ((xs.min(), xs.max()), (ys.min(), ys.max())), (xs.mean(), ys.mean()), m.sum()


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=4000)
    ap.add_argument("--gt", default="data/torus_w/images/view_0000.png")
    args = ap.parse_args()

    ds = lp.extract(args)
    pipe = pp.extract(args)
    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cam = scene.getTrainCameras()[0]
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    with torch.no_grad():
        out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
    rgb = out["render"].detach().cpu().permute(1, 2, 0).numpy()

    gt = cv2.imread(args.gt)[..., ::-1].astype(np.float32) / 255.0

    print(f"render shape {rgb.shape}  value range {rgb.min():.3f}..{rgb.max():.3f}")
    for name, img in [("RENDER", rgb), ("GT", gt)]:
        bbox, cent, n = bbox_of_bright(img)
        print(f"{name}: bright bbox={bbox} centroid=({cent[0]:.0f},{cent[1]:.0f}) npx={n}")
    # where does render put brightest pixel
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    y, x = np.unravel_index(gray.argmax(), gray.shape)
    print(f"render brightest pixel at ({x},{y})  image center (400,300)")


if __name__ == "__main__":
    main()
