"""E0: build a regular Cartesian opacity field grid from a trained GOF model.

opacity(x) = 1 - min_over_views transmittance(x)  -- GOF's view-independent
occupancy field (same evaluation as extract_mesh.py's evaluage_alpha, but on
a regular grid instead of tetra points, and without needing the tetra cpp).

Saves grid.npy (res**3 float32) + meta.json (bbox, res).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

GOF = os.path.expanduser("~/e0lab/gaussian-opacity-fields")
sys.path.insert(0, GOF)

from arguments import ModelParams, PipelineParams  # noqa: E402
from utils.general_utils import build_scaling_rotation  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--iteration", type=int, default=15000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", type=str, default="-1.3,1.3,-1.3,1.3,-1.3,1.3")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--views_every", type=int, default=4, help="use every k-th view for the min")
    ap.add_argument("--chunk", type=int, default=50000)
    args = ap.parse_args()

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    bbox = np.array([float(x) for x in args.bbox.split(",")], dtype=np.float32)
    assert bbox.shape == (6,)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()[:: args.views_every]
    print(f"[grid] {len(views)} views used for min-opacity field, "
          f"gaussians={gaussians.get_xyz.shape[0]}")

    res = args.res
    xs = np.linspace(bbox[0], bbox[1], res, dtype=np.float32)
    ys = np.linspace(bbox[2], bbox[3], res, dtype=np.float32)
    zs = np.linspace(bbox[4], bbox[5], res, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)

    # Direct view-independent Gaussian occupancy field.
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity_with_3D_filter.detach().squeeze(-1)
    scales = gaussians.get_scaling_with_3D_filter.detach()
    rot = gaussians.get_rotation.detach()
    L = build_scaling_rotation(scales, rot)
    inv_cov = torch.inverse(L @ L.transpose(1, 2))
    alpha = np.zeros(pts.shape[0], dtype=np.float32)
    gchunk = 512
    with torch.no_grad():
        for i in tqdm(range(0, pts.shape[0], args.chunk), desc="grid points"):
            p = torch.from_numpy(pts[i:i + args.chunk]).cuda()
            best = torch.zeros(p.shape[0], device="cuda")
            for j in range(0, xyz.shape[0], gchunk):
                d = p[:, None, :] - xyz[j:j + gchunk][None, :, :]
                q = torch.einsum("mni,nij,mnj->mn", d, inv_cov[j:j + gchunk], d)
                a = opacity[j:j + gchunk][None, :] * torch.exp(-0.5 * q)
                best = torch.maximum(best, a.max(dim=1).values)
            alpha[i:i + args.chunk] = best.clamp(0, 1).cpu().numpy()

    grid = alpha.reshape(res, res, res)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "grid.npy"), grid)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"bbox": bbox.tolist(), "res": res, "views_used": len(views),
                   "model": dataset.model_path, "iteration": args.iteration}, f, indent=1)
    print(f"[grid] occupancy in [0,1]: min={grid.min():.4f} max={grid.max():.4f} "
          f"frac>0.5={float((grid > 0.5).mean()):.4f} -> {args.out}")


if __name__ == "__main__":
    main()
