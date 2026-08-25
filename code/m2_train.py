#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M2: latent diffusion prior over the frozen M1 VAE latent space.

Trains a DDPM on posterior-sampled z vectors (1024-d) from the frozen
beta=1e-4 x dg1024 encoder, so unconditional sampling decodes to on-manifold
profiles instead of the Gaussian-prior (M1) average cloud.

  Data: for each of 819 train objects, mu + N_REPARAM reparameterized samples
        (encoder posterior q(z|x)) -> (819*(N_REPARAM+1), 1024) latent set,
        whitened to unit scale for a well-matched noise schedule.
  Model: time-conditional MLP denoiser, linear beta schedule, T=1000,
         eps-prediction (standard DDPM).
  Sampling: DDIM (eta=0) lives in m2_gen.py.

Usage:
  python m2_train.py --epochs 200          # full (1-3h on L20)
  python m2_train.py --epochs 5 --n_rep 1  # smoke
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from m1_train_vae import VAE, load_object, ray_encodings

ROOT = "/root/e0lab/e0"
M1_CKPT = "m1_vae_dg1024_b1e-4"
OUT = os.path.join(ROOT, "m2_latent_diff")
N_REPARAM = 8
SEED = 42


def get_beta_schedule(T=1000, b1=1e-4, bT=2e-2):
    betas = torch.linspace(b1, bT, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, 0)
    return betas, alphas, alpha_bar


def time_embedding(t, dim=256, max_period=10000.0):
    """Sinusoidal time embedding; t in [0,1] scaled to [0,1000] timesteps."""
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period, dtype=torch.float32))
                      * torch.arange(half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLP(nn.Module):
    """MLP denoiser: z_t (dim_g) + time embedding -> predicted noise (dim_g)."""

    def __init__(self, dim_g=1024, hidden=1024, t_dim=256):
        super().__init__()
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(dim_g + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim_g))

    def forward(self, z, t):
        c = self.t_proj(time_embedding(t, self.t_proj[0].in_features))
        return self.net(torch.cat([z, c], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_rep", type=int, default=N_REPARAM)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--subset", type=int, default=0,
                    help="0 = full train (819); N = first N train objects (smoke)")
    ap.add_argument("--out", type=str, default=OUT,
                    help="output dir (use a scratch dir for smoke runs)")
    ap.add_argument("--target", choices=["eps", "x0", "v"], default="v",
                    help="prediction target. eps: classic DDPM (numerically "
                         "unstable on near-Gaussian latent). x0: predict clean "
                         "latent, SNR-weighted (t->T collapses to constant -> "
                         "diversity loss). v: improved-DDPM v-param, keeps "
                         "signal at t->T and never divides by sqrt(a_bar). "
                         "Default v.")
    ap.add_argument("--data_std", type=float, default=0.5,
                    help="if set, training latent = mu + data_std*eps instead "
                         "of the true posterior sample (mu + posterior_std*eps). "
                         "Smaller data_std raises the mu-structure SNR the "
                         "diffusion has to learn (near-Gaussian posterior is "
                         "~99% background noise). 0.5 = final config (scan "
                         "0.1/0.3/0.5; gen FWHM==recon 7.0, diversity 0.65x GT).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m1 = json.load(open(os.path.join(ROOT, M1_CKPT, "meta.json")))
    dim_g = m1["dim_g"]
    train_names = m1["train_names"]
    if args.subset:
        train_names = train_names[:args.subset]
    print("M1 ckpt %s: dim_g=%d, n_train=%d" % (M1_CKPT, dim_g, len(train_names)),
          flush=True)

    # frozen M1 encoder
    model = VAE(dim_g=dim_g)
    model.load_state_dict(torch.load(os.path.join(ROOT, M1_CKPT, "model.pt"),
                                     map_location="cpu"))
    model.eval()
    model = model.to(device)

    from multiprocessing.dummy import Pool
    with Pool(8) as pool:
        data = dict(pool.map(lambda n: (n, load_object(n)), train_names,
                             chunksize=4))

    # 1. encode all train objects -> mu + reparam samples (the p(z) dataset)
    latents, mu_all, rmaxs = [], [], []
    with torch.no_grad():
        for n in train_names:
            sh, cov, peak, rmax = data[n]
            mu, lv = model.encode(sh[None].to(device))
            if args.data_std is None:
                noise_std = torch.exp(0.5 * lv[0])        # true posterior std
            else:
                noise_std = torch.full_like(mu[0], args.data_std)
            zs = [mu[0]]
            for _ in range(args.n_rep):
                zs.append(mu[0] + noise_std * torch.randn_like(mu[0]))
            latents.append(torch.stack(zs).cpu())
            mu_all.append(mu[0].cpu())
            rmaxs.append(rmax)
    Z = torch.cat(latents, dim=0).numpy()          # (N*(n_rep+1), dim_g)
    MU = torch.stack(mu_all).numpy()               # (N, dim_g) for diagnostics
    print("latent set: %s (mu subset %s)" % (Z.shape, MU.shape), flush=True)

    # 2. whitening
    z_mean = Z.mean(0, keepdims=True)
    z_std = Z.std(0, keepdims=True) + 1e-3
    Zw = ((Z - z_mean) / z_std).astype(np.float32)
    print("z: mean|.|_dim=%.3f std|.|_dim=%.3f; whitened std=%.3f"
          % (float(np.abs(z_mean).mean()), float(z_std.mean()), float(Zw.std())),
          flush=True)

    # 3. DDPM
    betas, alphas, alpha_bar = get_beta_schedule(args.T)
    alpha_bar = alpha_bar.to(device)
    den = TimeMLP(dim_g).to(device)
    opt = torch.optim.Adam(den.parameters(), lr=args.lr, weight_decay=1e-6)
    print("denoiser params=%.2fM" % (sum(p.numel() for p in den.parameters()) / 1e6),
          flush=True)

    Zt = torch.from_numpy(Zw).to(device)
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
            x0 = Zt[order[b0:b0 + args.batch]]
            B = x0.shape[0]
            t = torch.rand(B, device=device)
            eps = torch.randn_like(x0)
            a_bar = alpha_bar[(t * (args.T - 1)).long()]       # (B,)
            x_t = torch.sqrt(a_bar[:, None]) * x0 + torch.sqrt(1.0 - a_bar[:, None]) * eps
            if args.target == "eps":
                loss = ((den(x_t, t) - eps) ** 2).mean()
            elif args.target == "x0":
                loss = (a_bar[:, None] * (den(x_t, t) - x0) ** 2).mean()
            else:  # v-prediction (improved DDPM)
                v = torch.sqrt(a_bar[:, None]) * eps \
                    - torch.sqrt(1.0 - a_bar[:, None]) * x0
                loss = ((den(x_t, t) - v) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        if ep == 0 or (ep + 1) % 5 == 0 or (ep + 1) == args.epochs:
            print("epoch %3d mse=%.5f lr=%.1e" % (ep + 1, tot / nb, lr),
                  flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save(den.state_dict(), os.path.join(args.out, "denoiser.pt"))
    np.save(os.path.join(args.out, "z_mean.npy"), z_mean.astype(np.float32))
    np.save(os.path.join(args.out, "z_std.npy"), z_std.astype(np.float32))
    np.save(os.path.join(args.out, "mu_all.npy"), MU.astype(np.float32))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"m1_ckpt": M1_CKPT, "dim_g": dim_g, "n_train": len(train_names),
                   "n_latent": n, "n_rep": args.n_rep, "epochs": args.epochs,
                   "T": args.T, "lr": args.lr, "batch": args.batch,
                   "schedule": "linear", "b1": 1e-4, "bT": 2e-2,
                   "whiten": True, "seed": args.seed, "target": args.target,
                   "data_std": args.data_std,
                   "train_names": train_names}, f)
    print("done. -> %s/" % args.out, flush=True)


if __name__ == "__main__":
    main()
