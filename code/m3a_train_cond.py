#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3a: image-conditioned latent diffusion over the frozen M1 latent space.

Unconditional M2 learns p(z); M3a learns p(z | c) where c is the frozen D8
mean-pool image encoder (ResNet18 over the object's 8 fixed views 0,6,...,42).
Sampling from p(z|c) -> frozen M1 decoder gives generative (multi-solution)
profile reconstruction, the M3a gate: val dmeds <= 0.040 (near the D8
discriminative 0.0338).

Everything downstream of the condition is the M2-final recipe verbatim:
  v-prediction, linear beta (1e-4->2e-2, T=1000), data = mu + 0.5*eps
  (data_std=0.5), whitening, cosine LR + grad clip, DDIM eta=0 50-step.

Cond dropout (--cond_drop 0.1 default) prevents the condition from memorizing
training objects (819 strong conditions). The denoiser input is
cat([z_t, c, time_emb]) -> MLP (M2 TimeMLP with the condition channel added).

Usage:
  python m3a_train_cond.py --epochs 400            # full
  python m3a_train_cond.py --subset 24 --epochs 10 # smoke
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from m1_train_vae import VAE, load_object, ray_encodings
from m2_train import time_embedding, get_beta_schedule
from d8_train_full import Model as D8Model, IMG_T

ROOT = "/root/e0lab/e0"
RENDERS = "/root/gso/renders"
M1_CKPT = "m1_vae_dg1024_b1e-4"
D8_CKPT = "d8_mean_pool"
OUT = os.path.join(ROOT, "m3a_cond")
FIXED_VIEWS = list(range(0, 48, 6))            # eval protocol views (8)
N_REPARAM = 8
SEED = 42


def load_views(name):
    """8 fixed-view images + az/el view features (D8 schema)."""
    rd = os.path.join(RENDERS, name)
    poses = json.load(open(os.path.join(rd, "poses.json")))
    views = poses["views"]
    imgs, feats = [], []
    for i in FIXED_VIEWS:
        v = views[i]
        im = Image.open(os.path.join(rd, "images", "view_%04d.png" % i)).convert("RGB")
        imgs.append(IMG_T(im))
        az, el = float(v["az"]), float(v["el"])
        feats.append([np.sin(az), np.cos(az), np.sin(el), np.cos(el)])
    return torch.stack(imgs), torch.tensor(feats, dtype=torch.float32)


@torch.no_grad()
def encode_cond(d8, imgs, feats):
    """Frozen D8 mean-pool: (B,K,3,224,224) -> (B,512) global condition."""
    B, K = imgs.shape[0], imgs.shape[1]
    x = d8.backbone(imgs.flatten(0, 1)).flatten(1).view(B, K, -1)
    f = torch.cat([x, feats], dim=-1)
    vf = d8.proj(f)
    return vf.mean(dim=1)


class CondTimeMLP(nn.Module):
    """z_t + condition + time -> predicted v. M2 TimeMLP + cond channel."""

    def __init__(self, dim_g=1024, cond=512, hidden=1024, t_dim=256):
        super().__init__()
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(dim_g + cond + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim_g))

    def forward(self, z, c, t):
        c_t = self.t_proj(time_embedding(t, self.t_proj[0].in_features))
        return self.net(torch.cat([z, c, c_t], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_rep", type=int, default=N_REPARAM)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--subset", type=int, default=0,
                    help="0 = full train (819); N = first N train objects")
    ap.add_argument("--data_std", type=float, default=0.5,
                    help="x0 = mu + data_std*eps (M2 final config 0.5)")
    ap.add_argument("--cond_drop", type=float, default=0.1,
                    help="probability to zero the condition during training "
                         "(robustness / CFG basis / anti-memorization)")
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = json.load(open(os.path.join(ROOT, M1_CKPT, "meta.json")))
    dim_g = meta["dim_g"]
    train_names = meta["train_names"]
    if args.subset:
        train_names = train_names[:args.subset]
    print("M1 ckpt %s: dim_g=%d, n_train=%d" % (M1_CKPT, dim_g, len(train_names)),
          flush=True)

    # frozen M1 encoder
    model = VAE(dim_g=dim_g)
    model.load_state_dict(torch.load(os.path.join(ROOT, M1_CKPT, "model.pt"),
                                     map_location="cpu"))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    # frozen D8 condition encoder
    d8 = D8Model(fusion="mean_pool")
    d8.load_state_dict(torch.load(os.path.join(ROOT, D8_CKPT, "model.pt"),
                                  map_location="cpu"))
    d8.eval().to(device)
    for p in d8.parameters():
        p.requires_grad_(False)
    print("D8 cond encoder loaded (backbone+proj frozen).", flush=True)

    from multiprocessing.dummy import Pool
    with Pool(8) as pool:
        data = dict(pool.map(lambda n: (n, load_object(n)), train_names,
                             chunksize=4))
    with Pool(8) as pool:
        views = dict(pool.map(lambda n: (n, load_views(n)), train_names,
                              chunksize=4))
    print("profiles + views loaded.", flush=True)

    # encode mu (+ reparam samples) and the per-object condition, batched.
    # NOTE: per-object serial encode+image-load was ~20min for 819; batched
    # GPU forward over chunks makes this a couple of minutes.
    latents, mu_all, conds = [], [], []
    ENC_BATCH = 32
    with torch.no_grad():
        for b0 in range(0, len(train_names), ENC_BATCH):
            bn = train_names[b0:b0 + ENC_BATCH]
            sh_b = torch.stack([data[n][0] for n in bn]).to(device)
            mu, _ = model.encode(sh_b)                       # (B, dim_g)
            noise_std = torch.full_like(mu[0], args.data_std)
            zs = [mu]
            for _ in range(args.n_rep):
                zs.append(mu + noise_std * torch.randn_like(mu))
            latents.append(torch.stack(zs, dim=1).reshape(-1, mu.shape[1]).cpu())
            mu_all.append(mu.cpu())
            imgs_b = torch.stack([views[n][0] for n in bn]).to(device)
            feats_b = torch.stack([views[n][1] for n in bn]).to(device)
            c = encode_cond(d8, imgs_b, feats_b)             # (B, 512)
            conds.append(c.repeat_interleave(args.n_rep + 1, dim=0).cpu())
    Z = torch.cat(latents, dim=0).numpy()                    # (N*(n_rep+1), dim_g)
    C = torch.cat(conds, dim=0).numpy().astype(np.float32)   # same rows as Z
    MU = torch.cat(mu_all, dim=0).numpy()                    # (N, dim_g)
    print("latent set %s, cond set %s" % (Z.shape, C.shape), flush=True)

    # whitening (M2 recipe, on the training latent set)
    z_mean = Z.mean(0, keepdims=True)
    z_std = Z.std(0, keepdims=True) + 1e-3
    Zw = ((Z - z_mean) / z_std).astype(np.float32)
    print("z: mean|.|_dim=%.3f std|.|_dim=%.3f; whitened std=%.3f"
          % (float(np.abs(z_mean).mean()), float(z_std.mean()), float(Zw.std())),
          flush=True)

    # DDPM (v-prediction, M2 final)
    betas, alphas, alpha_bar = get_beta_schedule(args.T)
    alpha_bar = alpha_bar.to(device)
    den = CondTimeMLP(dim_g, C.shape[1]).to(device)
    opt = torch.optim.Adam(den.parameters(), lr=args.lr, weight_decay=1e-6)
    print("cond denoiser params=%.2fM" % (sum(p.numel() for p in den.parameters()) / 1e6),
          flush=True)

    Zt = torch.from_numpy(Zw).to(device)
    Ct = torch.from_numpy(C).to(device)
    n = len(Zt)
    lr0 = args.lr
    lr_min = lr0 * 0.03
    for ep in range(args.epochs):
        den.train()
        tot, nb = 0.0, 0
        lr = lr_min + 0.5 * (lr0 - lr_min) * (1.0 + np.cos(np.pi * ep / args.epochs))
        for g in opt.param_groups:
            g["lr"] = lr
        order = torch.randperm(n)
        for b0 in range(0, n, args.batch):
            idx = order[b0:b0 + args.batch]
            x0 = Zt[idx]
            c = Ct[idx]
            if args.cond_drop > 0:
                mask = (torch.rand(len(idx), device=device) > args.cond_drop)
                c = c * mask[:, None]
            B = x0.shape[0]
            t = torch.rand(B, device=device)
            eps = torch.randn_like(x0)
            a_bar = alpha_bar[(t * (args.T - 1)).long()]
            x_t = torch.sqrt(a_bar[:, None]) * x0 + torch.sqrt(1.0 - a_bar[:, None]) * eps
            v = torch.sqrt(a_bar[:, None]) * eps - torch.sqrt(1.0 - a_bar[:, None]) * x0
            loss = ((den(x_t, c, t) - v) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        if ep == 0 or (ep + 1) % 5 == 0 or (ep + 1) == args.epochs:
            print("epoch %3d mse=%.5f lr=%.1e" % (ep + 1, tot / nb, lr), flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save(den.state_dict(), os.path.join(args.out, "denoiser.pt"))
    np.save(os.path.join(args.out, "z_mean.npy"), z_mean.astype(np.float32))
    np.save(os.path.join(args.out, "z_std.npy"), z_std.astype(np.float32))
    np.save(os.path.join(args.out, "mu_all.npy"), MU.astype(np.float32))
    np.save(os.path.join(args.out, "c_all.npy"), C.astype(np.float32))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"m1_ckpt": M1_CKPT, "d8_ckpt": D8_CKPT, "dim_g": dim_g,
                   "cond_dim": C.shape[1], "views": FIXED_VIEWS,
                   "n_train": len(train_names), "n_latent": n,
                   "n_rep": args.n_rep, "epochs": args.epochs, "T": args.T,
                   "lr": args.lr, "batch": args.batch, "schedule": "linear",
                   "b1": 1e-4, "bT": 2e-2, "whiten": True, "seed": args.seed,
                   "target": "v", "data_std": args.data_std,
                   "cond_drop": args.cond_drop, "train_names": train_names}, f)
    print("done. -> %s/" % args.out, flush=True)


if __name__ == "__main__":
    main()
