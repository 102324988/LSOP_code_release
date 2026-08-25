#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D6-v3 full-60-val inference for the honest A/B vs D7c2.

Regenerates D6 v3 (mean-pool + v3 loss) artifacts on the SAME 60 val objects
that D7c2 used (read from d7c2_ca/meta.json), with the FIXED inference views
(0,6,...,42) and the CORRECTED 2D soft-argmax:
    soft = (pred * axis[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
so d7_eval_fixed.py can compare the two models on the identical 60-object
population. Artifacts go to d6_pred_v3_full/.
Usage: python d6_v3_full_infer.py
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/e0lab/e0")
from d6_infer_v3 import Model, ray_encodings, IMG_T  # noqa: E402

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
D4 = os.path.join(ROOT, "output", "gso_d4")
CKPT = os.path.join(ROOT, "d6_pred_v3", "model.pt")
OUT = os.path.join(ROOT, "d6_pred_v3_full")
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
    meta = json.load(open(os.path.join(ROOT, "d7c2_ca", "meta.json")))
    val_names = meta["val_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, _ = ray_encodings()
    model = Model().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
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
    print("done. artifacts for %d val objects -> d6_pred_v3_full/" % len(val_names),
          flush=True)


if __name__ == "__main__":
    main()
