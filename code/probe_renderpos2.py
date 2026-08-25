"""Compare render RGB vs GT image pixel-by-pixel for cam0."""
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
    rgb = out["render"].detach().cpu().permute(1, 2, 0).numpy()[:, :, :3]
    gt = cv2.imread(args.gt)[..., ::-1].astype(np.float32) / 255.0

    print(f"render RGB range {rgb.min():.3f}..{rgb.max():.3f}  GT range {gt.min():.3f}..{gt.max():.3f}")
    print(f"n channels in out['render']: {out['render'].shape[0]}")

    # object = pixels where GT differs from bg 0.95
    obj = gt.max(axis=-1) < 0.94
    print(f"GT object mask: {obj.sum()} px ({(obj.sum()/obj.size*100):.1f}%)")
    # same pixels in render
    robj = rgb.max(axis=-1) < 0.94
    print(f"render object-ish mask: {robj.sum()} px")

    # distance between centroids of the object mask
    if obj.sum() > 0 and robj.sum() > 0:
        ys, xs = np.nonzero(obj); rc = (xs.mean(), ys.mean())
        ys, xs = np.nonzero(robj); rr = (xs.mean(), ys.mean())
        print(f"GT obj centroid=({rc[0]:.0f},{rc[1]:.0f})  render obj centroid=({rr[0]:.0f},{rr[1]:.0f})")

    # mean abs diff inside object region
    diff = np.abs(rgb - gt)
    print(f"mean|diff| overall={diff.mean():.4f}  inside GT-object={diff[obj].mean():.4f}")
    # correlation
    a, b = rgb[..., :3].reshape(-1, 3), gt.reshape(-1, 3)
    print(f"corr: {np.corrcoef(a[:,0], b[:,0])[0,1]:.3f} "
          f"{np.corrcoef(a[:,1], b[:,1])[0,1]:.3f} {np.corrcoef(a[:,2], b[:,2])[0,1]:.3f}")


if __name__ == "__main__":
    main()
