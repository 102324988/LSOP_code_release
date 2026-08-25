#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect the FROZEN model's actual predictions: argmax-bin distribution,
predicted peak heights, cross-object agreement. Loads d7_pred/model.pt.
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
from d7_train_perray import Model, ray_encodings, IMG_T

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
OUT = os.path.join(ROOT, "d7_pred")
D4 = os.path.join(ROOT, "output", "gso_d4")
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
    names = meta["val_names"][:6]
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
            cov = np.load(os.path.join(D4, n, "profiles", "coverage.npy"))
            m = cov >= 0.02
            arg = pred.argmax(-1)
            frac_late = float((arg[m] > 0.6 * N_BINS).mean())
            print("%-30s peak p50=%.4f p90=%.4f | argmax bin med=%.1f frac_late(>58)=%.2f | prof mean=%.5f"
                  % (n[:30], np.median(pred.max(-1)), np.percentile(pred.max(-1), 90),
                     np.median(arg[m]), frac_late, pred.mean()))
    a = list(profs.values())
    print("\ncross-object agreement (should be ~1 if collapsed):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = np.corrcoef(a[i].ravel(), a[j].ravel())[0, 1]
            d = np.abs(a[i] - a[j]).mean()
            print("  %s vs %s  corr=%.4f mean|diff|=%.5f"
                  % (names[i][:20], names[j][:20], c, d))


if __name__ == "__main__":
    main()
