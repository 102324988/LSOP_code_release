#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q1 center-robustness (reviewer question):
Shift the 8 input views horizontally by dx pixels (simulating an error in the
object-center estimate of the spherical parameterization) and measure how the
D8 discriminative soft-depth error degrades. Reflect padding keeps the shift
in-image. All views shift by the same amount (a fixed center bias).

GT loading + soft-depth protocol follow m3a_gen_cond / d7_eval_fixed:
  m = cov >= 0.02 ; soft = soft-argmax(pred); gtn = peak/rmax;
  dmed_s = median |soft/(N_BINS-1) - gtn| over covered rays, then over objects.

Usage (server, /opt/conda base):
  python center_robust.py --subset 12          # smoke
  python center_robust.py                       # full val 90
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
D4 = os.path.join(ROOT, "output", "gso_d4")
N_BINS = 96
SEL = np.arange(0, 48, 6)

sys.path.insert(0, ROOT)
from d8_train_full import Model, ray_encodings, IMG_T  # noqa: E402
from m1_train_vae import load_object                 # noqa: E402  (GT profiles)


def load_imgs(name):
    rd = os.path.join(RENDERS, name)
    imgs, feats = [], []
    poses = json.load(open(os.path.join(rd, "poses.json")))
    views = poses["views"]
    for i in SEL:
        v = views[i]
        im = Image.open(os.path.join(rd, "images", "view_%04d.png" % i)).convert("RGB")
        imgs.append(IMG_T(im))
        az, el = float(v["az"]), float(v["el"])
        feats.append([np.sin(az), np.cos(az), np.sin(el), np.cos(el)])
    return torch.stack(imgs)[None], torch.tensor([feats], dtype=torch.float32)


def shift_x(imgs, dx):
    """Move content right by dx px (reflect-padded); dx<0 moves left.
    F.pad reflect supports 4D (k,c,h,w), not 5D, so flatten views."""
    if dx == 0:
        return imgs
    b, k, c, h, w = imgs.shape
    flat = imgs.view(b * k, c, h, w)
    pad = F.pad(flat, (abs(dx), abs(dx), 0, 0), mode="reflect")
    if dx > 0:
        out = pad[..., :w]
    else:
        out = pad[..., -w:]
    return out.view(b, k, c, h, w)


def eval_dx(model, device, ray_pe, cache, dx):
    errs, per = [], []
    with torch.no_grad():
        for n, (imgs, feats) in cache.items():
            imgs = shift_x(imgs.to(device), dx)
            pred = torch.sigmoid(model(imgs, feats.to(device), ray_pe))[0].cpu().numpy()
            _sh, cov, peak, rmax = load_object(n)
            m = cov.numpy() >= 0.02
            axis = np.arange(N_BINS)
            soft = (pred[m] * axis[None, :]).sum(-1) / (pred[m].sum(-1) + 1e-6)
            gtn = peak.numpy()[m] / rmax
            d = np.abs(soft / (N_BINS - 1) - gtn)
            e = float(np.median(d))
            errs.append(e)
            per.append({"name": n, "dmed_s": e})
    return float(np.median(errs)), float(np.mean(errs)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--subset", type=int, default=0, help="0 = full split")
    ap.add_argument("--dxs", default="0,2,4,9",
                    help="pixel shifts at 224px input (0/1%/2%/4% of width)")
    args = ap.parse_args()
    dxs = [int(x) for x in args.dxs.split(",")]

    meta = json.load(open(os.path.join(ROOT, "d8_mean_pool", "meta.json")))
    names = meta[args.split + "_names"]
    if args.subset:
        names = names[:args.subset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=%s | n=%d split=%s dxs=%s" % (device, len(names), args.split, dxs),
          flush=True)

    ray_pe, _ = ray_encodings()
    model = Model(fusion="mean_pool").to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, "d8_mean_pool", "model.pt"),
                                     map_location=device))
    model.eval()

    # preload images once, reuse across all shifts
    print("preloading %d objects ..." % len(names), flush=True)
    cache = {n: load_imgs(n) for n in names}
    print("preload done", flush=True)

    out = {"split": args.split, "n": len(names), "dmed_med": {}, "dmed_mean": {},
           "per_object_dx0": None}
    for dx in dxs:
        med, mean, per = eval_dx(model, device, ray_pe, cache, dx)
        out["dmed_med"][str(dx)] = med
        out["dmed_mean"][str(dx)] = mean
        if dx == 0:
            out["per_object_dx0"] = per
        print("dx=%3dpx (%4.1f%% of 224): dmed_s med=%.4f mean=%.4f"
              % (dx, 100.0 * dx / 224.0, med, mean), flush=True)

    outdir = os.path.join(ROOT, "center_robust")
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    print("saved -> %s/summary.json" % outdir, flush=True)


if __name__ == "__main__":
    main()
