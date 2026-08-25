#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose D7a collapse: does the trained model output object-independent
(mean-profile) predictions? Loads d7_pred/model.pt, runs 3 val objects at fixed
views, compares their predicted profiles pairwise.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/root/e0lab/e0")
from d7_train_perray import Model, ray_encodings, IMG_T  # reuse

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
OUT = os.path.join(ROOT, "d7_pred")


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
    names = meta["val_names"][:4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, _ = ray_encodings()
    model = Model().to(device)
    model.load_state_dict(torch.load(os.path.join(OUT, "model.pt"), map_location=device))
    model.eval()
    sel = np.arange(0, 48, 6)
    profs = {}
    with torch.no_grad():
        for n in names:
            imgs, feats = load_object(n, sel)
            pred = torch.sigmoid(model(imgs.to(device), feats.to(device), ray_pe))[0].cpu().numpy()
            profs[n] = pred
            print(n, "peak_p50=%.4f peak_p90=%.4f mean=%.5f" % (
                np.median(pred.max(-1)), np.percentile(pred.max(-1), 90), pred.mean()))
    # pairwise agreement
    a = list(profs.values())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.abs(a[i] - a[j]).mean()
            c = np.corrcoef(a[i].ravel(), a[j].ravel())[0, 1]
            print("pair %s vs %s  mean|diff|=%.5f  corr=%.4f"
                  % (names[i][:20], names[j][:20], d, c))


if __name__ == "__main__":
    main()
