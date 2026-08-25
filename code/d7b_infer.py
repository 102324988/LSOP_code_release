#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D7b inference: same fixed-view artifacts as d7_infer.py, but for the
cross-attention model (d7b_ca). Imports Model/ray_encodings/IMG_T from
d7b_train_ca.py so the eval harness (d7_eval.py) works unchanged.
Usage: python d7b_infer.py
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/e0lab/e0")
from d7b_train_ca import Model, ray_encodings, IMG_T  # noqa: E402

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
D4 = os.path.join(ROOT, "output", "gso_d4")
OUT = os.path.join(ROOT, "d7b_ca")
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
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    S = float(meta["S"])
    val_names = meta["val_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, _ = ray_encodings()
    model = Model().to(device)
    model.load_state_dict(torch.load(os.path.join(OUT, "model.pt"), map_location=device))
    model.eval()
    sel = np.arange(0, 48, 6)
    with torch.no_grad():
        for n in val_names:
            imgs, feats = load_object(n, sel)
            pred = torch.sigmoid(model(imgs.to(device), feats.to(device), ray_pe))[0].cpu()
            pr = pred.argmax(-1)
            axis = torch.arange(N_BINS)
            soft = (pred * axis[None, None, :]).sum(-1) / (pred.sum(-1, keepdim=True) + 1e-6)
            prof = os.path.join(D4, n, "profiles")
            raw = np.load(os.path.join(prof, "profiles.npy")).astype(np.float32).reshape(-1, N_BINS)
            cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)
            gt_t = np.clip(raw / S, 0.0, 1.0)
            np.save(os.path.join(OUT, n + ".npy"), pr.numpy().astype(np.float32))
            np.save(os.path.join(OUT, n + "_soft.npy"), soft[0].numpy().astype(np.float32))
            np.save(os.path.join(OUT, n + "_prof.npy"), pred.numpy().astype(np.float32))
            np.save(os.path.join(OUT, n + "_predpeak.npy"),
                    pred.max(-1).values.numpy().astype(np.float32))
            np.save(os.path.join(OUT, n + "_cov.npy"), cov)
            np.save(os.path.join(OUT, n + "_gtprof.npy"), gt_t.astype(np.float32))
            np.save(os.path.join(OUT, n + "_gtpeak.npy"),
                    gt_t.max(-1).astype(np.float32))
            print(n, flush=True)
    print("done. artifacts for %d val objects -> d7b_ca/" % len(val_names), flush=True)


if __name__ == "__main__":
    main()
