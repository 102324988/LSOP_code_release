#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D8: full-scale (999-object) profile decoder with the D7-winning recipe.

Runs the v3 loss (raw pred vs per-ray max-norm target, coverage-weighted
profile L1) on ALL 999 GSO objects, with an independent 90-object test split
that has NEVER been touched by D6/D7 (test = perm[460:550] of the same seed-42
permutation that produced the D7 400/60 split). View fusion is a flag:
  --fusion cross_attn  (per-ray cross-attention over views, D7b/D7c model)
  --fusion mean_pool   (cheaper, D6-v3 default)
The D7c verdict picks the winner; everything else is identical.

Split (seed-42 permutation p of all 999 D5 names; rng is consumed in the same
order as D7 so the 400 shared train objects get identical per-object view
selections as D7's train):
  train = p[:400] + p[580:999]    (819 = D7's 400 + 419 fresh)
  val   = p[400:460] + p[550:580] (90  = D7's 60 val incl. shared-12 + 30 fresh)
  test  = p[460:550]              (90  = brand new clean generalization test)
Total 819 + 90 + 90 = 999.

Reported eval artifacts use the FIXED inference views (0,6,...,42) saved by
d8_infer.py, independent of the random training-view selection, so D8 val
numbers stay apples-to-apples with the D6/D7 shared-12 table.

Usage: python d8_train_full.py --fusion cross_attn [--epochs 40] [--views 8]
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
N_PHI, N_THETA, N_BINS = 128, 64, 96
PE_DIM = 21
CA_DIM = 128
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
    pe = []
    for f in (1.0, 2.0, 4.0):
        pe.append(np.sin(dirs * f))
        pe.append(np.cos(dirs * f))
    pe = np.concatenate(pe, axis=-1)
    pe = np.concatenate([dirs, pe], axis=-1)
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
    profs = np.load(os.path.join(prof, "profiles.npy")).astype(np.float32)
    profs = profs.reshape(-1, N_BINS)
    mx = profs.max(-1, keepdims=True)          # max-norm (v3) target
    profs = profs / np.maximum(mx, 1e-6)
    cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)
    peak = np.load(os.path.join(prof, "depth_peak.npy")).astype(np.float32)
    rmax = float(json.load(open(os.path.join(prof, "meta.json")))["rmax"])
    return (torch.stack(imgs), torch.tensor(feats, dtype=torch.float32),
            torch.from_numpy(profs), torch.from_numpy(cov),
            torch.from_numpy(peak), rmax)


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
    def __init__(self, fusion="cross_attn", gdim=512, nb=N_BINS, ca=CA_DIM):
        super().__init__()
        self.fusion = fusion
        net = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.proj = nn.Linear(512 + 4, gdim)
        if fusion == "cross_attn":
            self.w_q = nn.Linear(PE_DIM, ca)
            self.w_k = nn.Linear(gdim, ca)
            self.w_v = nn.Linear(gdim, ca)
            self.in_proj = nn.Linear(64 + ca, 64)
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
        vf = self.proj(f)
        g = vf.mean(dim=1)
        rpe = ray_pe.to(imgs.device)
        if self.fusion == "cross_attn":
            q = self.w_q(rpe)
            k = self.w_k(vf)
            v = self.w_v(vf)
            att = torch.softmax(q[None] @ k.transpose(-1, -2) / (k.shape[-1] ** 0.5), dim=-1)
            c = att @ v
            e0 = self.pe_proj(rpe)
            e = torch.cat([e0[None].expand(B, -1, -1), c], dim=-1)
            e = torch.relu(self.in_proj(e))
        else:  # mean_pool (D6-v3): no per-ray context, just the global FiLM
            e = self.pe_proj(rpe).unsqueeze(0).expand(B, -1, -1)
        outs = []
        for r0 in range(0, e.shape[1], chunk):
            h = e[:, r0:r0 + chunk]
            for blk in self.decoder:
                h = blk(h, g)
            outs.append(self.head(h))
        return torch.cat(outs, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", choices=["cross_attn", "mean_pool"], default="cross_attn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wvar", action="store_true",
                    help="add decp cross-object variance weight (D7c2 recipe, "
                         "required for cross_attn to escape mean-absorption)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    names = sorted(os.listdir(D5))
    idx = rng.permutation(len(names))          # same 999-draw consumption as D7
    train_names = [names[i] for i in idx[:400]] + [names[i] for i in idx[580:999]]
    val_names = [names[i] for i in idx[400:460]] + [names[i] for i in idx[550:580]]
    test_names = [names[i] for i in idx[460:550]]
    assert len(train_names) == 819 and len(val_names) == 90 and len(test_names) == 90
    assert len(set(train_names) & set(test_names)) == 0
    assert len(set(val_names) & set(test_names)) == 0

    ray_pe, dirs_t = ray_encodings()

    from multiprocessing.dummy import Pool

    def _load(n):
        return n, load_object(n, args.views, rng)
    with Pool(8) as pool:
        print("preload train (819)...", flush=True)
        tr = dict(pool.map(_load, train_names, chunksize=8))
        print("preload val (90)...", flush=True)
        va = dict(pool.map(_load, val_names, chunksize=4))
    print("train=%d val=%d test=%d (disjoint)" % (len(tr), len(va), len(test_names)),
          flush=True)

    # D7c2 anti-collapse variance weight (only when --wvar): per-ray variance of
    # the max-norm target over train objects, mean over bins, normalized [0,1].
    w_var = None
    if args.wvar:
        s1 = torch.zeros(N_PHI * N_THETA, N_BINS)
        s2 = torch.zeros(N_PHI * N_THETA, N_BINS)
        for n in train_names:
            t = tr[n][2]
            s1 += t; s2 += t * t
        cnt = float(len(train_names))
        mean = s1 / cnt
        var = (s2 / cnt - mean * mean).clamp(min=0)
        w_var = var.mean(-1)
        w_var = w_var / (w_var.max() + 1e-6)
        print("variance-weight: max=%.4f mean=%.4f frac_gt05=%.3f"
              % (float(w_var.max()), float(w_var.mean()),
                 float((w_var > 0.5).float().mean())), flush=True)

    gs = torch.stack([tr[n][2] for n in train_names]).mean(0)
    base_l1, base_dep = [], []
    for n in val_names:
        sh, cov, peak, rmax = va[n][2], va[n][3], va[n][4], va[n][5]
        m = cov >= 0.02
        base_l1.append(float((gs[m] - sh[m]).abs().mean()))
        gpk = gs[m].argmax(1) / (N_BINS - 1)
        gtn = peak[m] / rmax
        base_dep.append(float((gpk - gtn).abs().median()))
    print("BASELINE global-mean(maxnorm): prof-L1=%.4f depth-med=%.4f"
          % (np.median(base_l1), np.median(base_dep)), flush=True)

    model = Model(fusion=args.fusion).to(device)
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
            sh = torch.stack([tr[n][2] for n in bn]).to(device)
            cov = torch.stack([tr[n][3] for n in bn]).to(device)
            pred = torch.sigmoid(model(imgs, feats, ray_pe))
            w = (cov + 0.05).clamp(max=1.0).unsqueeze(-1)
            if w_var is not None:  # D7c2 recipe
                w = w * (0.10 + w_var.to(device)[None, :, None]).clamp(max=2.0)
            # v3 EXACT loss: raw pred vs max-norm target. Do NOT max-norm the
            # prediction (zero gradient at the peak bin -> degenerate freeze).
            loss = (w * (pred - sh).abs()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        model.eval()
        pl1, dmed_h, dmed_s = [], [], []
        with torch.no_grad():
            for n in val_names:
                imgs, feats, sh, cov, peak, rmax = va[n]
                pred = torch.sigmoid(model(imgs[None].to(device),
                                           feats[None].to(device), ray_pe))[0].cpu()
                m = cov >= 0.02
                pl1.append(float((pred[m] - sh[m]).abs().mean()))
                axis = torch.arange(N_BINS)
                soft = (pred * axis[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
                gtn = peak[m] / rmax
                dmed_h.append(float((pred[m].argmax(1) / (N_BINS - 1) - gtn).abs().median()))
                dmed_s.append(float((soft[m] / (N_BINS - 1) - gtn).abs().median()))
        print("epoch %2d loss=%.4f val prof-L1=%.4f depth-med hard=%.4f soft=%.4f"
              % (ep + 1, tot / n_batches, np.median(pl1),
                 np.median(dmed_h), np.median(dmed_s)), flush=True)

    outdir = os.path.join(ROOT, "d8_%s" % args.fusion)
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"fusion": args.fusion, "n_train": len(train_names),
                   "n_val": len(val_names), "n_test": len(test_names),
                   "seed": args.seed, "epochs": args.epochs, "loss": "v3",
                   "wvar": bool(args.wvar), "target": "maxnorm",
                   "train_names": train_names,
                   "val_names": val_names, "test_names": test_names}, f)
    print("done. checkpoint+meta -> %s/" % outdir, flush=True)


if __name__ == "__main__":
    main()
