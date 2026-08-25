#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D7c2 inference: cross-attn + v3 + w_var model (anti-collapse recipe).

Runs the trained D7c2 checkpoint over the val split with the FIXED
inference views (0,6,...,42) and saves the same reconstruction artifacts
as the D7 side. The soft-argmax uses the CORRECTED 2D formula:
    soft = (pred * axis[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
(no keepdim, 2D axis) so the saved _soft.npy is a per-ray scalar, NOT the
(R,R) outer-product bug that corrupted all pre-fix artifacts.

Artifacts go to d7c2_pred/ (clean dir, separate from checkpoints) so that
    python d7_eval_fixed.py d7c2_pred
gives the shared-12 A/B vs D6 v3 (0.0568 / 0.3499) and D7b (0.0824 / 0.3814).
Usage: python d7c2_infer.py
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/e0lab/e0")
from d7c2_train_ca import Model, ray_encodings, IMG_T  # noqa: E402

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
D4 = os.path.join(ROOT, "output", "gso_d4")
CKPT = os.path.join(ROOT, "d7c2_ca")
OUT = os.path.join(ROOT, "d7c2_pred")
N_BINS = 96


def load_object(name, sel):
    rd = os.path.join(RENDERS, name)
    imgs, feats = [], []
    poses = json.load(open(os.path.join(rd, "poses.json")))
    views = poses["views"]
    for i in sel:
        v = views[i]
        im = Image.open(os.path.join(rd, "images", "view_%04d.png" % i)).convert("RGB")
        imgs.append(IMG_T(im))
        az, el = float(v["az"]), float(v["el"])
        feats.append([np.sin(az), np.cos(az), np.sin(el), np.cos(el)])
    return (torch.stack(imgs)[None],
            torch.tensor([feats], dtype=torch.float32))


def main():
    meta = json.load(open(os.path.join(CKPT, "meta.json")))
    val_names = meta["val_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, _ = ray_encodings()
    model = Model().to(device)
    model.load_state_dict(torch.load(os.path.join(CKPT, "model.pt"), map_location=device))
    model.eval()
    sel = np.arange(0, 48, 6)
    os.makedirs(OUT, exist_ok=True)
    with torch.no_grad():
        for n in val_names:
            imgs, feats = load_object(n, sel)
            pred = torch.sigmoid(model(imgs.to(device), feats.to(device), ray_pe))[0].cpu()
            pr = pred.argmax(-1)
            axis = torch.arange(N_BINS)
            soft = (pred * axis[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)   # FIXED 2D
            prof = os.path.join(D4, n, "profiles")
            cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)
            np.save(os.path.join(OUT, n + ".npy"), pr.numpy().astype(np.float32))
            np.save(os.path.join(OUT, n + "_soft.npy"), soft.numpy().astype(np.float32))
            np.save(os.path.join(OUT, n + "_prof.npy"), pred.numpy().astype(np.float32))
            np.save(os.path.join(OUT, n + "_predpeak.npy"),
                    pred.max(-1).values.numpy().astype(np.float32))
            np.save(os.path.join(OUT, n + "_cov.npy"), cov)
            print(n, flush=True)
    print("done. artifacts for %d val objects -> d7c2_pred/" % len(val_names), flush=True)


if __name__ == "__main__":
    main()
