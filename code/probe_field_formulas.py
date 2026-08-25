"""E0 decisive probe: collect per-(point, view) alpha_integrated on a grid, then
compare candidate opacity-field formulas offline.

The GOF kernel outputs alpha_integrated = accumulated front-to-back opacity
A_v(x) (NOT transmittance).  The official formula uses
    alpha = 1 - min_v A_v(x)
which mathematically equals max_v T_v(x) = max transmittance.  That looks like
the *inverted* field (empty->1, occupied->0).  Here we test every plausible
formula and score each against known-good points on the torus.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import integrate  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=6000)
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--views_every", type=int, default=4, help="use every k-th train view")
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--bbox", type=str, default="-1.4,1.4,-1.4,1.4,-1.4,1.4")
    ap.add_argument("--out", required=True, help="dir to save Av matrix + json")
    args = ap.parse_args()

    ds = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(ds.sh_degree)
    scene = Scene(ds, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras()[:: args.views_every]
    print(f"[field] views={len(views)} gaussians={gaussians.get_xyz.shape[0]}")

    bbox = np.array([float(x) for x in args.bbox.split(",")], dtype=np.float32)
    res = args.res
    xs = np.linspace(bbox[0], bbox[1], res, dtype=np.float32)
    ys = np.linspace(bbox[2], bbox[3], res, dtype=np.float32)
    zs = np.linspace(bbox[4], bbox[5], res, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)
    n = pts.shape[0]
    nv = len(views)

    Av = np.zeros((n, nv), dtype=np.float32)
    with torch.no_grad():
        for ci in range(0, n, args.chunk):
            p = torch.from_numpy(pts[ci:ci + args.chunk]).cuda()
            for vi, view in enumerate(views):
                ret = integrate(p, view, gaussians, pipe, bg, kernel_size=ds.kernel_size)
                Av[ci:ci + p.shape[0], vi] = ret["alpha_integrated"].detach().cpu().numpy()
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "Av.npy"), Av.astype(np.float16))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"bbox": bbox.tolist(), "res": res, "n_views": nv,
                   "model": ds.model_path, "iteration": args.iteration}, f, indent=1)
    print(f"[field] saved Av {Av.shape} -> {args.out}")
    print(f"        alpha_integrated stats: min={Av.min():.4f} "
          f"max={Av.max():.4f} mean={Av.mean():.4f}")


if __name__ == "__main__":
    main()
