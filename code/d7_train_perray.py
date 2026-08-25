#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D7 (stage 1): raw (un-normalized) profile supervision for real confidence.

Isolated change vs D6-v3 (everything else identical: FiLM decoder, mean-pool
view fusion, 400/60 seed-42 split, 8 views, 40 epochs, same coverage-weighted
profile-L1 loss):
  - target = raw_profs / S, with S = 99.9th percentile of per-ray max over the
    training set (clipped to 1). Peak strength now = absolute occupancy
    intensity -> a usable confidence readout. D6's per-ray max-norm saturated
    every non-empty peak to 1, which is exactly why peak-height filtering
    failed there (D6 report sec 6.2).
  - depth/Chamfer readouts and confidence filtering are evaluated separately
    by d7_eval.py on the saved artifacts.

Usage: python d7_train_perray.py [--n_train N] [--n_val N] [--epochs E]
                                 [--views K] [--lr L] [--seed S]
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
OUT = os.path.join(ROOT, "d7_pred")
N_PHI, N_THETA, N_BINS = 128, 64, 96
N_DIRS = N_PHI * N_THETA
PE_DIM = 21  # dirs(3) + 3 freqs * sin/cos * 3 comps (18)
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
                    axis=-1).reshape(-1, 3).astype(np.float32)  # (R,3)
    pe = []
    for f in (1.0, 2.0, 4.0):
        pe.append(np.sin(dirs * f))
        pe.append(np.cos(dirs * f))
    pe = np.concatenate(pe, axis=-1)  # (R,18)
    pe = np.concatenate([dirs, pe], axis=-1)  # (R,21)
    return torch.from_numpy(pe), torch.from_numpy(dirs)


def load_object(name, n_views, rng):
    rd = os.path.join(RENDERS, name)
    sel = np.sort(rng.choice(48, n_views, replace=False))
    imgs, feats = [], []
    poses = json.load(open(os.path.join(rd, "poses.json")))
    views = poses["views"]
    for i in sel:
        v = views[i]
        im = Image.open(os.path.join(rd, "images", "view_%04d.png" % i)).convert("RGB")
        imgs.append(IMG_T(im))
        az, el = float(v["az"]), float(v["el"])
        feats.append([np.sin(az), np.cos(az), np.sin(el), np.cos(el)])
    prof = os.path.join(D4, name, "profiles")
    profs = np.load(os.path.join(prof, "profiles.npy")).astype(np.float32)  # (128,64,96)
    profs = profs.reshape(-1, N_BINS)  # (8192,96) row-major phi-major
    cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)  # (8192,)
    peak = np.load(os.path.join(prof, "depth_peak.npy")).astype(np.float32)
    rmax = float(json.load(open(os.path.join(prof, "meta.json")))["rmax"])
    # NOTE: raw (un-normalized) profiles; target scaling happens at loss time.
    return (torch.stack(imgs), torch.tensor(feats, dtype=torch.float32),
            torch.from_numpy(profs), torch.from_numpy(cov),
            torch.from_numpy(peak), rmax)


class FiLMBlock(nn.Module):
    """Linear -> ReLU, then FiLM-modulate with the global feature g."""
    def __init__(self, din, dout, gdim):
        super().__init__()
        self.fc = nn.Linear(din, dout)
        self.gamma = nn.Linear(gdim, dout)  # scale
        self.beta = nn.Linear(gdim, dout)   # shift

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
        g = self.proj(f).mean(dim=1)  # (B,512)
        e = self.pe_proj(ray_pe.to(imgs.device))  # (R,64)
        outs = []
        for r0 in range(0, e.shape[0], chunk):
            e_c = e[r0:r0 + chunk].to(imgs.device)   # (C,64)
            C = e_c.shape[0]
            h = e_c[None, :, :].expand(B, -1, -1)    # (B,C,64)
            for blk in self.decoder:
                h = blk(h, g)
            outs.append(self.head(h))
        return torch.cat(outs, dim=1)  # (B,R,96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--loss", choices=["plain", "var", "varp", "dec", "decp"], default="decp",
                    help="plain/var/varp: raw-profile L1 variants (collapse to a "
                         "mean-profile fixed point); dec: decomposed shape+peak "
                         "loss; decp: dec + cross-object variance weight (winner)")
    ap.add_argument("--lambda_pk", type=float, default=0.5,
                    help="weight of the peak (confidence) loss in the dec loss")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    names = sorted(os.listdir(D5))
    idx = rng.permutation(len(names))
    train_names = [names[i] for i in idx[:args.n_train]]
    val_names = [names[i] for i in idx[args.n_train:args.n_train + args.n_val]]

    ray_pe, dirs_t = ray_encodings()

    from multiprocessing.dummy import Pool

    def _load(n):
        return n, load_object(n, args.views, rng)
    with Pool(8) as pool:
        print("preload train...", flush=True)
        tr = dict(pool.map(_load, train_names, chunksize=4))
        print("preload val...", flush=True)
        va = dict(pool.map(_load, val_names, chunksize=4))
    print("train=%d val=%d" % (len(tr), len(va)), flush=True)

    # target scale S = 99.9th percentile of per-ray peak over the train set
    peaks = np.concatenate([tr[n][2].max(-1).values.numpy() for n in train_names])
    S = float(np.percentile(peaks, 99.9))
    S = max(S, 1e-4)
    print("target scale S = %.5f (99.9 pct per-ray peak over %d rays)"
          % (S, peaks.size), flush=True)

    def tgt(profs):
        return (profs / S).clamp(max=1.0)

    # variance-weight per ray: how much the target profile varies across train
    # objects (breaks the L1 mean-collapse fixed point -- rays where objects
    # agree get low weight, rays where conditioning matters get high weight).
    if args.loss in ("var", "varp", "decp"):
        s1 = torch.zeros(N_DIRS, N_BINS)
        s2 = torch.zeros(N_DIRS, N_BINS)
        for n in train_names:
            t = tgt(tr[n][2])
            s1 += t; s2 += t * t
        cnt = float(len(train_names))
        mean = s1 / cnt
        var = (s2 / cnt - mean * mean).clamp(min=0)
        w_var = var.mean(-1)
        w_var = w_var / (w_var.max() + 1e-6)  # (N_DIRS,) in [0,1]
        print("variance-weight: max=%.4f mean=%.4f frac_gt05=%.3f"
              % (float(w_var.max()), float(w_var.mean()),
                 float((w_var > 0.5).float().mean())), flush=True)

    # baseline: global mean raw-rescaled profile (hard-argmax depth)
    gs = torch.stack([tgt(tr[n][2]) for n in train_names]).mean(0)  # (R,96)
    base_l1, base_dep = [], []
    for n in val_names:
        raw, cov, peak, rmax = va[n][2], va[n][3], va[n][4], va[n][5]
        m = cov >= 0.02
        base_l1.append(float((gs[m] - tgt(raw)[m]).abs().mean()))
        gpk = gs[m].argmax(1) / (N_BINS - 1)      # normalized [0,1]
        gtn = peak[m] / rmax                      # normalized [0,1]
        base_dep.append(float((gpk - gtn).abs().median()))
    print("BASELINE global-mean(raw-rescaled): prof-L1=%.4f depth-med=%.4f"
          % (np.median(base_l1), np.median(base_dep)), flush=True)

    model = Model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 2), gamma=0.3)
    n_batches = (len(train_names) + args.batch - 1) // args.batch

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_names))
        tot = 0.0
        for b0 in range(0, len(train_names), args.batch):
            bn = [train_names[i] for i in order[b0:b0 + args.batch]]
            imgs = torch.stack([tr[n][0] for n in bn]).to(device)
            feats = torch.stack([tr[n][1] for n in bn]).to(device)
            raw = torch.stack([tr[n][2] for n in bn]).to(device)
            cov = torch.stack([tr[n][3] for n in bn]).to(device)
            pred = torch.sigmoid(model(imgs, feats, ray_pe))
            w = (cov + 0.05).clamp(max=1.0).unsqueeze(-1)
            # ---- DECOMPOSED LOSS (fixes the raw-profile mean-collapse) ----
            # The raw-profile L1 has a zero-gradient fixed point at the mean
            # profile (continuous small targets cancel signed errors), so the
            # model collapses to an object-independent output. Decouple the two
            # informations:
            #   shape part (WHERE the surface is): compare max-norm profiles,
            #     exactly the v3 loss that provably learns conditioning (spiky
            #     targets -> gradient does not vanish at the mean);
            #   peak part (HOW STRONG the surface is = confidence): L1 on the
            #     profile peak vs the rescaled raw peak. This is the confidence.
            # The shape comparison is scale-invariant, so the terms don't fight.
            if args.loss in ("dec", "decp"):
                wv = w_var.to(device)[None, :, None] if args.loss == "decp" else None
                shape_pred = pred / (pred.max(-1, keepdim=True).values + 1e-6)
                mx_t = raw.max(-1, keepdim=True).values
                shape_gt = raw / (mx_t + 1e-6)
                peak_pred = pred.max(-1).values
                peak_gt = (raw / S).clamp(max=1.0).max(-1).values
                w2 = (w * (wv + 0.10)).clamp(max=2.0) if wv is not None else w
                loss = (w2 * (shape_pred - shape_gt).abs()).mean() + \
                       args.lambda_pk * (w[..., 0] * (peak_pred - peak_gt).abs()).mean()
            else:  # plain/var/varp: cov-weighted raw-profile L1 (collapses; kept for A/B)
                if args.loss == "var":
                    w = w * (0.10 + w_var.to(device)[None, :, None])
                elif args.loss == "varp":
                    w = w * (0.10 + w_var.to(device)[None, :, None]) * \
                        (tgt(raw).max(-1, keepdim=True).values + 0.10)
                loss = (w * (pred - tgt(raw)).abs()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        # eval
        model.eval()
        pl1, dmed_h, dmed_s = [], [], []
        with torch.no_grad():
            for n in val_names:
                imgs, feats, raw, cov, peak, rmax = va[n]
                pred = torch.sigmoid(model(imgs[None].to(device),
                                           feats[None].to(device), ray_pe))[0].cpu()
                m = cov >= 0.02
                tgt_r = tgt(raw)
                pl1.append(float((pred[m] - tgt_r[m]).abs().mean()))
                axis = torch.arange(N_BINS)
                soft = (pred * axis[None, None, :]).sum(-1) / (pred.sum(-1, keepdim=True) + 1e-6)
                gtn = peak[m] / rmax
                dmed_h.append(float((pred[m].argmax(1) / (N_BINS - 1) - gtn).abs().median()))
                dmed_s.append(float((soft[0][m] / (N_BINS - 1) - gtn).abs().median()))
        print("epoch %2d loss=%.4f val prof-L1=%.4f depth-med hard=%.4f soft=%.4f"
              % (ep + 1, tot / n_batches, np.median(pl1),
                 np.median(dmed_h), np.median(dmed_s)), flush=True)

    # checkpoint + meta (names + scale, for d7_infer.py)
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump({"S": S, "n_train": args.n_train, "n_val": args.n_val,
                   "seed": args.seed, "epochs": args.epochs, "loss": args.loss,
                   "lambda_pk": args.lambda_pk,
                   "train_names": train_names, "val_names": val_names}, f)
    print("done. checkpoint+meta -> d7_pred/", flush=True)


if __name__ == "__main__":
    main()
