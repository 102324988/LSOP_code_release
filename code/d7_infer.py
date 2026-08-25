#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D7 inference: run the trained model at FIXED views (0,6,...,42) over all val
objects from d7_pred/meta.json and save full artifacts for d7_eval.py:
  <n>.npy        hard-argmax bin depth (96 bins)
  <n>_soft.npy   soft-argmax bin depth (center of mass)
  <n>_prof.npy   full 96-bin predicted profile (rescaled units, sigmoid out)
  <n>_predpeak.npy  predicted peak height per ray = confidence readout
  <n>_cov.npy    GT coverage
  <n>_gtprof.npy (GT raw profile / S).clamp(1)  (target space, for prof-L1)
  <n>_gtpeak.npy (GT raw peak / S).clamp(1)     (for calibration corr)
Usage: python d7_infer.py
"""
import json
import os

import numpy as np
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
D4 = os.path.join(ROOT, "output", "gso_d4")
OUT = os.path.join(ROOT, "d7_pred")
N_PHI, N_THETA, N_BINS = 128, 64, 96
PE_DIM = 21
IMG_T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def ray_encodings():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                    axis=-1).reshape(-1, 3).astype(np.float32)
    pe = [np.sin(dirs * f) for f in (1.0, 2.0, 4.0)] + \
         [np.cos(dirs * f) for f in (1.0, 2.0, 4.0)]
    pe = np.concatenate([dirs] + pe, axis=-1)
    return torch.from_numpy(pe), torch.from_numpy(dirs)


def load_object(name, n_views, sel):
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
    cov = np.load(os.path.join(D4, name, "profiles", "coverage.npy")).astype(np.float32)
    return (torch.stack(imgs)[None],
            torch.tensor([feats], dtype=torch.float32))


class FiLMBlock(nn.Module):
    def __init__(self, din, dout, gdim):
        super().__init__()
        self.fc = nn.Linear(din, dout)
        self.gamma = nn.Linear(gdim, dout)
        self.beta = nn.Linear(gdim, dout)

    def forward(self, x, g):
        h = torch.relu(self.fc(x))
        return h * self.gamma(g).unsqueeze(1) + self.beta(g).unsqueeze(1)


class Model(nn.Module):
    def __init__(self, gdim=512, nb=N_BINS):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.proj = nn.Linear(512 + 4, gdim)
        self.pe_proj = nn.Linear(PE_DIM, 64)
        self.decoder = nn.Sequential(
            FiLMBlock(64, 512, gdim),
            FiLMBlock(512, 512, gdim),
            FiLMBlock(512, 256, gdim),
        )
        self.head = nn.Linear(256, nb)

    def forward(self, imgs, feats, ray_pe, chunk=2048):
        B, K = imgs.shape[0], imgs.shape[1]
        x = self.backbone(imgs.flatten(0, 1)).flatten(1).view(B, K, -1)
        f = torch.cat([x, feats], dim=-1)
        g = self.proj(f).mean(dim=1)
        e = self.pe_proj(ray_pe.to(imgs.device))
        outs = []
        for r0 in range(0, e.shape[0], chunk):
            e_c = e[r0:r0 + chunk].to(imgs.device)
            C = e_c.shape[0]
            h = e_c[None, :, :].expand(B, -1, -1)
            for blk in self.decoder:
                h = blk(h, g)
            outs.append(self.head(h))
        return torch.cat(outs, dim=1)


def main():
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    S = float(meta["S"])
    val_names = meta["val_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, _ = ray_encodings()
    model = Model().to(device)
    model.load_state_dict(torch.load(os.path.join(OUT, "model.pt"), map_location=device))
    model.eval()
    sel = np.arange(0, 48, 6)  # fixed views, matching D6 comparisons
    with torch.no_grad():
        for n in val_names:
            imgs, feats = load_object(n, len(sel), sel)
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
            print(n, pred.shape, flush=True)
    print("done. artifacts for %d val objects -> d7_pred/" % len(val_names), flush=True)


if __name__ == "__main__":
    main()
