#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D5 feasibility: multi-view conditional -> spherical depth_peak field.
Minimal CNN baseline (ResNet18 encoder, mean-pooled view features -> MLP ->
64x128 depth_peak field). Compares against global-mean baseline on val.
Usage: python d5_train_baseline.py [--n_train 150] [--n_val 30] [--epochs 15]
                                    [--views 8] [--seed 42] [--device cuda]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
D4 = os.path.join(ROOT, "output", "gso_d4")
D5 = os.path.join(ROOT, "output", "gso_d5")
N_PHI, N_THETA = 128, 64
N_DIRS = N_PHI * N_THETA

IMG_T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_object(name, n_views, rng):
    """Return (imgs (n_views,3,224,224), feat (n_views,4), depth_peak (N_DIRS,),
       cov (N_DIRS,), rmax)."""
    rd = os.path.join(RENDERS, name)
    sel = np.sort(np.arange(0, 48, 48 // n_views).astype(int))
    imgs = []
    feats = []
    poses = json.load(open(os.path.join(rd, "poses.json")))
    views = poses["views"]
    for i in sel:
        v = views[i]
        im = Image.open(os.path.join(rd, "images", "view_%04d.png" % i)).convert("RGB")
        imgs.append(IMG_T(im))
        az, el = float(v["az"]), float(v["el"])
        feats.append([np.sin(az), np.cos(az), np.sin(el), np.cos(el)])
    prof = os.path.join(D4, name, "profiles")
    peak = np.load(os.path.join(prof, "depth_peak.npy")).astype(np.float32)
    cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)
    rmax = float(json.load(open(os.path.join(prof, "meta.json")))["rmax"])
    peak = peak / rmax  # normalize target to [0,1] (calibration fix)
    return (torch.stack(imgs), torch.tensor(feats, dtype=torch.float32),
            torch.from_numpy(peak), torch.from_numpy(cov), rmax)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.proj = nn.Linear(512 + 4, 512)
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, N_DIRS),
        )
        self.act = nn.Sigmoid()

    def forward(self, imgs, feats):
        B, K, C, H, W = imgs.shape
        x = self.backbone(imgs.flatten(0, 1)).flatten(1)  # (B*K,512)
        x = x.view(B, K, -1)
        f = torch.cat([x, feats], dim=-1)
        f = self.proj(f)
        g = f.mean(dim=1)  # (B,512)
        return self.act(self.mlp(g))  # (B,N_DIRS) in [0,1] (normalized depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    names = sorted(os.listdir(D5))
    idx = rng.permutation(len(names))
    train_names = [names[i] for i in idx[:args.n_train]]
    val_names = [names[i] for i in idx[args.n_train:args.n_train + args.n_val]]

    # preload (parallel, threads are fine: PIL releases the GIL)
    from multiprocessing.dummy import Pool

    def _load(n):
        return n, load_object(n, args.views, rng)
    with Pool(8) as pool:
        print("preloading train (parallel)...", flush=True)
        tr = dict(pool.map(_load, train_names, chunksize=4))
        print("preloading val (parallel)...", flush=True)
        va = dict(pool.map(_load, val_names, chunksize=4))
    print("train=%d val=%d" % (len(tr), len(va)), flush=True)

    # global-mean baseline (trained-set depth mean per direction)
    gmean = torch.stack([tr[n][2] for n in train_names]).mean(0)
    med_baseline = []
    for n in val_names:
        _, _, peak, cov, _ = va[n]
        mask = cov >= 0.02
        err = (gmean[mask] - peak[mask]).abs()
        med_baseline.append(float(err.median()))
    print("BASELINE global-mean val med-abs-depth=%.4f" % np.median(med_baseline), flush=True)

    model = Encoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.epochs // 2, gamma=0.3)
    lossf = nn.SmoothL1Loss(beta=0.05)
    n_batches = (len(train_names) + args.batch - 1) // args.batch

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_names))
        tot = 0.0
        for b0 in range(0, len(train_names), args.batch):
            bn = [train_names[i] for i in order[b0:b0 + args.batch]]
            imgs = torch.stack([tr[n][0] for n in bn]).to(device)
            feats = torch.stack([tr[n][1] for n in bn]).to(device)
            peak = torch.stack([tr[n][2] for n in bn]).to(device)
            cov = torch.stack([tr[n][3] for n in bn]).to(device)
            pred = model(imgs, feats)
            mask = cov >= 0.02
            loss = lossf(pred[mask], peak[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        # eval
        model.eval()
        meds = []
        with torch.no_grad():
            for n in val_names:
                imgs, feats, peak, cov, _ = va[n]
                pred = model(imgs[None].to(device), feats[None].to(device))[0].cpu()
                m = cov >= 0.02
                meds.append(float((pred[m] - peak[m]).abs().median()))
        print("epoch %2d loss=%.5f val med-abs=%.4f" % (ep + 1, tot / n_batches,
                                                        np.median(meds)), flush=True)

    # save predictions for a few val objects
    os.makedirs(os.path.join(ROOT, "d5_pred_v2split_norm"), exist_ok=True)
    model.eval()
    with torch.no_grad():
        for n in val_names[:8]:
            imgs, feats, peak, cov, rmax = va[n]
            pred = model(imgs[None].to(device), feats[None].to(device))[0].cpu().numpy()
            np.save(os.path.join(ROOT, "d5_pred_v2split_norm", n + ".npy"), pred)
    print("done. predictions -> d5_pred_v2split_norm/", flush=True)


if __name__ == "__main__":
    main()
