#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q7 external-baseline experiments on the SAME GSO split (819/90/90, seed 42):

  d8k1, d8k2 : D8 profile decoder trained with K=1 / K=2 views. D8 with K=8 is
               the paper's main model (val 0.0338 / test 0.0384); K=1 brackets
               single-view feed-forward methods (LRM/RayDF-class) and answers
               the reviewer's "fewer views" question.
  rw8,  rw1  : ray-wise distance-field baseline in the spirit of RayDF
               (image-cond + ray direction -> per-ray surface distance). No
               profile, no soft uncertainty: a direct contrast to the paper's
               per-ray occupancy-profile decoder.

All models share: ResNet18 encoder, 224x224, view fusion = mean-pool over K
views, Adam lr 3e-4 / batch 8 / 40 epochs / StepLR x0.3 at half epochs.
Training views: K random views per object (same protocol as the paper's D8).
Evaluation views: FIXED SEL[:K] = frames 0,6,...,6(K-1) (same protocol).
Metric: the paper's soft-depth, dmed_s = median |soft-argmax/(N_BINS-1) -
peak/rmax| over covered rays (cov>=0.02), then median over objects; for the
ray-wise baseline the prediction IS the depth, so the metric is median
|pred - peak/rmax|.

Checkpoints + meta saved under d8_mean_pool_vK / rw_dist_vK.
Usage: python exts_baseline.py --which d8k1|d8k2|rw8|rw1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from d8_train_full import Model as D8Model, IMG_T, N_PHI, N_THETA, N_BINS, PE_DIM
from d8_train_full import ray_encodings

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
RENDERS = "/root/gso/renders"
META = os.path.join(ROOT, "d8_mean_pool", "meta.json")
SEL = np.arange(0, 48, 6)          # fixed eval views (frames 0,6,...,42)


def load_views_imgs(name, idxs):
    rd = os.path.join(RENDERS, name)
    poses = json.load(open(os.path.join(rd, "poses.json")))
    views = poses["views"]
    imgs, feats = [], []
    for i in idxs:
        v = views[i]
        im = Image.open(os.path.join(rd, "images", "view_%04d.png" % i)).convert("RGB")
        imgs.append(IMG_T(im))
        az, el = float(v["az"]), float(v["el"])
        feats.append([np.sin(az), np.cos(az), np.sin(el), np.cos(el)])
    return torch.stack(imgs), torch.tensor(feats, dtype=torch.float32)


def load_profs(name):
    prof = os.path.join(D4, name, "profiles")
    p = np.load(os.path.join(prof, "profiles.npy")).astype(np.float32)
    p = p.reshape(-1, N_BINS)
    mx = p.max(-1, keepdims=True)
    sh = p / np.maximum(mx, 1e-6)
    cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)
    peak = np.load(os.path.join(prof, "depth_peak.npy")).astype(np.float32)
    rmax = float(json.load(open(os.path.join(prof, "meta.json")))["rmax"])
    return (torch.from_numpy(sh), torch.from_numpy(cov),
            torch.from_numpy(peak), rmax)


def load_fixed(name, K, rng):
    """Evaluation load: FIXED views SEL[:K] (paper protocol)."""
    imgs, feats = load_views_imgs(name, SEL[:K].tolist())
    sh, cov, peak, rmax = load_profs(name)
    return (name, imgs, feats, sh, cov, peak, rmax)


def load_train(name, K, rng):
    """Training load: K random views (paper protocol, same RNG draw)."""
    sel = np.sort(rng.choice(48, K, replace=False))
    imgs, feats = load_views_imgs(name, sel.tolist())
    sh, cov, peak, rmax = load_profs(name)
    return (name, imgs, feats, sh, cov, peak, rmax)


class FiLMBlock(nn.Module):
    def __init__(self, din, dout, gdim):
        super().__init__()
        self.fc = nn.Linear(din, dout)
        self.gamma = nn.Linear(gdim, dout)
        self.beta = nn.Linear(gdim, dout)

    def forward(self, x, g):
        h = torch.relu(self.fc(x))
        return h * self.gamma(g).unsqueeze(1) + self.beta(g).unsqueeze(1)


class Raywise(nn.Module):
    """Ray-wise distance field (RayDF-style): mean-pool view cond + per-ray
    direction -> per-ray surface distance, output in (0,1) via sigmoid."""

    def __init__(self, gdim=512):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.proj = nn.Linear(512 + 4, gdim)
        self.pe_proj = nn.Linear(PE_DIM, 64)
        self.decoder = nn.Sequential(
            FiLMBlock(64, 512, gdim),
            FiLMBlock(512, 512, gdim),
            FiLMBlock(512, 256, gdim))
        self.head = nn.Linear(256, 1)

    def forward(self, imgs, feats, ray_pe, chunk=2048):
        B, K = imgs.shape[0], imgs.shape[1]
        x = self.backbone(imgs.flatten(0, 1)).flatten(1).view(B, K, -1)
        f = torch.cat([x, feats], -1)
        g = self.proj(f).mean(dim=1)
        e = self.pe_proj(ray_pe.to(imgs.device)).unsqueeze(0).expand(B, -1, -1)
        outs = []
        for r0 in range(0, e.shape[1], chunk):
            h = e[:, r0:r0 + chunk]
            for blk in self.decoder:
                h = blk(h, g)
            outs.append(torch.sigmoid(self.head(h)))
        return torch.cat(outs, 1).squeeze(-1)     # (B,R) depth in (0,1)


def eval_split(model, device, ray_pe, names, K, is_d8):
    errs = []
    rng = np.random.RandomState(0)
    with torch.no_grad():
        for n in names:
            _, imgs, feats, sh, cov, peak, rmax = load_fixed(n, K, rng)
            out = model(imgs[None].to(device), feats[None].to(device),
                        ray_pe)[0].cpu()
            m = cov.numpy() >= 0.02
            gtn = peak.numpy()[m] / rmax
            if is_d8:
                pred = torch.sigmoid(out)          # (R,96)
                axis = np.arange(N_BINS)
                soft = (pred.numpy()[m] * axis[None, :]).sum(-1) / \
                       (pred.numpy()[m].sum(-1) + 1e-6)
                d = np.abs(soft / (N_BINS - 1) - gtn)
            else:
                d = np.abs(out.numpy()[m] - gtn)
            errs.append(float(np.median(d)))
    return float(np.median(errs)), float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["d8k1", "d8k2", "rw8", "rw1"],
                    required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    meta = json.load(open(META))
    tr_names, va_names, te_names = (meta["train_names"], meta["val_names"],
                                    meta["test_names"])
    K = {"d8k1": 1, "d8k2": 2, "rw8": 8, "rw1": 1}[args.which]
    is_d8 = args.which.startswith("d8")
    outdir = os.path.join(ROOT, ("d8_mean_pool_v%d" if is_d8 else "rw_dist_v%d") % K)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(42)
    rng = np.random.RandomState(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("%s | K=%d | train=%d val=%d test=%d | %s"
          % (args.which, K, len(tr_names), len(va_names), len(te_names), device),
          flush=True)

    ray_pe, _ = ray_encodings()
    model = (D8Model(fusion="mean_pool") if is_d8 else Raywise()).to(device)

    from multiprocessing.dummy import Pool
    with Pool(8) as pool:
        tr = dict((t[0], t[1:]) for t in
                  pool.map(lambda n: load_train(n, K, rng), tr_names, chunksize=8))
    print("train preload done (819 objects, K=%d)" % K, flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, args.epochs // 2), gamma=0.3)
    n_batches = (len(tr_names) + args.batch - 1) // args.batch

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_names))
        tot = 0.0
        for b0 in range(0, len(tr_names), args.batch):
            bn = [tr_names[i] for i in order[b0:b0 + args.batch]]
            imgs = torch.stack([tr[n][0] for n in bn]).to(device)
            feats = torch.stack([tr[n][1] for n in bn]).to(device)
            sh = torch.stack([tr[n][2] for n in bn]).to(device)
            cov = torch.stack([tr[n][3] for n in bn]).to(device)
            peak = torch.stack([tr[n][4] for n in bn]).to(device)
            rmax = torch.tensor([tr[n][5] for n in bn],
                                dtype=torch.float32, device=device)[:, None]
            if is_d8:
                pred = torch.sigmoid(model(imgs, feats, ray_pe))
                w = (cov + 0.05).clamp(max=1.0).unsqueeze(-1)
                loss = (w * (pred - sh).abs()).mean()
            else:
                pred = model(imgs, feats, ray_pe)   # (B,R) in (0,1)
                gtn = peak / rmax
                m = (cov >= 0.02).float()
                loss = (m * (pred - gtn).abs()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        model.eval()
        vmed, _ = eval_split(model, device, ray_pe, va_names, K, is_d8)
        print("epoch %2d loss=%.5f val dmed_s=%.4f" % (ep + 1, tot / n_batches,
                                                       vmed), flush=True)

    tmed, tmean = eval_split(model, device, ray_pe, te_names, K, is_d8)
    vmed, vmean = eval_split(model, device, ray_pe, va_names, K, is_d8)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    json.dump({"which": args.which, "K": K,
               "kind": "d8_profile" if is_d8 else "raywise_dist",
               "dmed_val": vmed, "dmed_test": tmed, "epochs": args.epochs},
              open(os.path.join(outdir, "meta.json"), "w"))
    print("FINAL val=%.4f (mean %.4f) test=%.4f (mean %.4f) -> %s/"
          % (vmed, vmean, tmed, tmean, outdir), flush=True)


if __name__ == "__main__":
    main()
