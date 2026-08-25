"""E0: render a trained view and compare against GT image pixels."""
import argparse
import os
import sys

import numpy as np
import cv2

import torch

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
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

    with torch.no_grad():
        out = render(cam, g, pipe, bg, kernel_size=ds.kernel_size)
    img = out["render"].permute(1, 2, 0).detach().cpu().numpy()[:, :, :3]
    img = np.clip(img, 0.0, 1.0)

    # GT image
    gt = cv2.imread(os.path.join(ds.source_path, "images", f"view_{args.cam:04d}.png"))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    H, W = gt.shape[:2]
    print(f"view {args.cam}: render {img.shape} GT {gt.shape}")
    print(f"render pixel range: [{img.min():.3f},{img.max():.3f}] "
          f"mean={img.mean():.3f}")
    print(f"GT     pixel range: [{gt.min():.3f},{gt.max():.3f}] "
          f"mean={gt.mean():.3f}")
    mse = ((img - gt) ** 2).mean()
    print(f"MSE={mse:.5f} PSNR={10*np.log10(1.0/mse):.2f} dB")
    # fraction of pixels where render is "white" vs GT
    print(f"render >0.99: {(img.max(2)>0.99).mean()*100:.1f}%   "
          f"GT >0.99: {(gt.max(2)>0.99).mean()*100:.1f}%")
    # center-pixel check: torus tube should project around image center
    print(f"center patch render={img[H//2-2:H//2+3, W//2-2:W//2+3].mean():.3f} "
          f"GT={gt[H//2-2:H//2+3, W//2-2:W//2+3].mean():.3f}")


if __name__ == "__main__":
    main()
