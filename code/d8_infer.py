#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D8 inference: run the trained full-scale model on the val and test splits
with the FIXED inference views (0,6,...,42), saving the same reconstruction
artifacts as the D7/D6 side (hard/soft/prof/predpeak/cov, no gtpeak/gtprof so
d7_eval treats it as is_D7=False -- the D6-side A/B columns).

Artifacts go to d8_<fusion>_val/ and d8_<fusion>_test/ so that
  python d7_eval.py d8_cross_attn_test   -> clean independent-test numbers
  python d7_eval.py d8_cross_attn_val    -> val numbers incl. shared-12 A/B
Usage: python d8_infer.py --fusion cross_attn
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/e0lab/e0")
from d8_train_full import Model, ray_encodings, IMG_T  # noqa: E402

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
D4 = os.path.join(ROOT, "output", "gso_d4")
N_BINS = 96
SEL = np.arange(0, 48, 6)  # fixed inference views


def load_object(name):
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
    return (torch.stack(imgs)[None],
            torch.tensor([feats], dtype=torch.float32))


def run_split(model, device, ray_pe, names, outdir):
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        for n in names:
            imgs, feats = load_object(n)
            pred = torch.sigmoid(model(imgs.to(device), feats.to(device), ray_pe))[0].cpu()
            pr = pred.argmax(-1)
            axis = torch.arange(N_BINS)
            soft = (pred * axis[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
            prof = os.path.join(D4, n, "profiles")
            cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)
            np.save(os.path.join(outdir, n + ".npy"), pr.numpy().astype(np.float32))
            np.save(os.path.join(outdir, n + "_soft.npy"), soft.numpy().astype(np.float32))
            np.save(os.path.join(outdir, n + "_prof.npy"), pred.numpy().astype(np.float32))
            np.save(os.path.join(outdir, n + "_predpeak.npy"),
                    pred.max(-1).values.numpy().astype(np.float32))
            np.save(os.path.join(outdir, n + "_cov.npy"), cov)
            print(n, flush=True)
    print("done. artifacts for %d objects -> %s/" % (len(names), outdir), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", choices=["cross_attn", "mean_pool"], default="cross_attn")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(ROOT, "d8_%s" % args.fusion, "meta.json")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, _ = ray_encodings()
    model = Model(fusion=args.fusion).to(device)
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "d8_%s" % args.fusion, "model.pt"), map_location=device))
    model.eval()
    for sp in args.splits.split(","):
        names = meta[sp + "_names"]
        outdir = os.path.join(ROOT, "d8_%s_%s" % (args.fusion, sp))
        print("== split %s (%d objects) -> %s" % (sp, len(names), outdir), flush=True)
        run_split(model, device, ray_pe, names, outdir)


if __name__ == "__main__":
    main()
