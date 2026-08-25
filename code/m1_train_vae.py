#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1: spherical-profile VAE — compact latent prior over GSO profiles.

Learns a VAE over the per-ray soft-occupancy profile field (64x128 rays x 96
bins), per the design in workspace/m0/M1-剖面VAE实验设计.md:

  Encoder: per-ray bin-MLP(96->128->64) -> spherical feature (theta,phi,64)
           -> 3x spherical 2D-CNN (phi circular, theta zero pad, 64/128/256)
           -> global pool -> mu/logvar (dim_g)
  Latent:  z ~ N(0,1) reparameterized, prior N(0,1), KL with warmup
  Decoder: (reuses D8 structure) per-ray PE(21)->proj(64)->3xFiLMBlock->96-bin
           sigmoid, conditioned on z via FiLM
  Loss:    D8-v3 weighted profile L1 on raw pred vs max-norm target
           + beta*KL. NO L2/CE (would smear sharp profiles).

Split: reuses the D8 819/90/90 (seed-42) names from d8_mean_pool/meta.json so
every M1 number is per-object comparable with D8.

Usage:
  python m1_train_vae.py --dim_g 1024 --beta 1e-3 --epochs 60   # full
  python m1_train_vae.py --subset 24 --val_subset 12 --epochs 10 # smoke
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
N_PHI, N_THETA, N_BINS = 128, 64, 96
PE_DIM = 21


def ray_encodings():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                     np.cos(TH)], axis=-1).reshape(-1, 3).astype(np.float32)
    pe = []
    for f in (1.0, 2.0, 4.0):
        pe.append(np.sin(dirs * f))
        pe.append(np.cos(dirs * f))
    pe = np.concatenate(pe, axis=-1)
    pe = np.concatenate([dirs, pe], axis=-1)
    return torch.from_numpy(pe), torch.from_numpy(dirs)


class SphericalConv(nn.Module):
    """3x3 conv over (theta, phi); phi wraps circularly, theta zero-pads at
    the two pole ends."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=0)

    def forward(self, x):
        x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)   # phi circular
        x = torch.cat([x[:, :, :1], x, x[:, :, -1:]], dim=2)   # theta zero
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, dim_g=1024):
        super().__init__()
        self.bin_mlp = nn.Sequential(
            nn.Linear(N_BINS, 128), nn.ReLU(), nn.Linear(128, 64))
        self.sph = nn.Sequential(
            SphericalConv(64, 128), nn.ReLU(),
            SphericalConv(128, 256), nn.ReLU(),
            SphericalConv(256, 256), nn.ReLU())
        self.mu = nn.Linear(256, dim_g)
        self.logvar = nn.Linear(256, dim_g)

    def forward(self, prof, chunk=4096):
        # prof: (B, R, 96), R flattened phi-major (phi outer, theta inner)
        B, R = prof.shape[0], prof.shape[1]
        outs = []
        for r0 in range(0, R, chunk):
            outs.append(self.bin_mlp(prof[:, r0:r0 + chunk]))
        f = torch.cat(outs, dim=1)                     # (B, R, 64)
        # (B, R, 64) -> (B, PH, TH, 64) -> (B, ch, TH, PH)
        f = f.reshape(B, N_PHI, N_THETA, 64).permute(0, 3, 2, 1)
        h = self.sph(f).mean(dim=(2, 3))               # (B, 256)
        return self.mu(h), self.logvar(h)


class FiLMBlock(nn.Module):
    def __init__(self, din, dout, gdim):
        super().__init__()
        self.fc = nn.Linear(din, dout)
        self.gamma = nn.Linear(gdim, dout)
        self.beta = nn.Linear(gdim, dout)

    def forward(self, x, g):
        h = torch.relu(self.fc(x))
        return h * self.gamma(g).unsqueeze(1) + self.beta(g).unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, gdim):
        super().__init__()
        self.pe_proj = nn.Linear(PE_DIM, 64)
        self.decoder = nn.Sequential(
            FiLMBlock(64, 512, gdim),
            FiLMBlock(512, 512, gdim),
            FiLMBlock(512, 256, gdim),
        )
        self.head = nn.Linear(256, N_BINS)

    def forward(self, g, ray_pe, chunk=2048):
        e = self.pe_proj(ray_pe).unsqueeze(0).expand(g.shape[0], -1, -1)
        outs = []
        for r0 in range(0, e.shape[1], chunk):
            h = e[:, r0:r0 + chunk]
            for blk in self.decoder:
                h = blk(h, g)
            outs.append(self.head(h))
        return torch.cat(outs, dim=1)


class VAE(nn.Module):
    def __init__(self, dim_g=1024):
        super().__init__()
        self.encoder = Encoder(dim_g)
        self.decoder = Decoder(dim_g)

    def encode(self, prof):
        return self.encoder(prof)

    def reparam(self, prof):
        mu, lv = self.encoder(prof)
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def decode(self, z, ray_pe):
        return self.decoder(z, ray_pe)

    def forward(self, prof, ray_pe):
        mu, lv = self.encoder(prof)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        pred = torch.sigmoid(self.decoder(z, ray_pe))
        return pred, mu, lv


def load_object(name):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim_g", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0,
                    help="0 = full train (819); N = first N train objects (smoke)")
    ap.add_argument("--val_subset", type=int, default=0,
                    help="0 = full val (90); N = first N val objects (smoke)")
    ap.add_argument("--aux", action="store_true",
                    help="add differentiable morphology aux loss")
    ap.add_argument("--aux_lam", type=float, default=0.1,
                    help="weight of the morphology aux loss (default 0.1)")
    ap.add_argument("--aux_mode", choices=["std", "peak"], default="peak",
                    help="aux morphology term: 'std' = centroid+std diff "
                         "(weak; diag shows pred entropy ~ GT already), "
                         "'peak' = centroid + |1-pred.max| height (attacks "
                         "the measured defect: pred peak ~0.4 vs GT 1.0)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = json.load(open(os.path.join(ROOT, "d8_mean_pool", "meta.json")))
    train_names = meta["train_names"]
    val_names = meta["val_names"]
    if args.subset:
        train_names = train_names[:args.subset]
    if args.val_subset:
        val_names = val_names[:args.val_subset]

    from multiprocessing.dummy import Pool

    def _load(n):
        return n, load_object(n)
    with Pool(8) as pool:
        print("preload train (%d)..." % len(train_names), flush=True)
        tr = dict(pool.map(_load, train_names, chunksize=8))
        print("preload val (%d)..." % len(val_names), flush=True)
        va = dict(pool.map(_load, val_names, chunksize=4))

    # baseline: global-mean maxnorm profile (success criterion comparator)
    gs = torch.stack([tr[n][0] for n in train_names]).mean(0)
    base_l1, base_dep = [], []
    for n in val_names:
        sh, cov, peak, rmax = va[n]
        m = cov >= 0.02
        base_l1.append(float((gs[m] - sh[m]).abs().mean()))
        gpk = gs[m].argmax(1) / (N_BINS - 1)
        gtn = peak[m] / rmax
        base_dep.append(float((gpk - gtn).abs().median()))
    print("BASELINE global-mean(maxnorm): prof-L1=%.4f depth-med=%.4f"
          % (np.median(base_l1), np.median(base_dep)), flush=True)

    ray_pe, _ = ray_encodings()
    ray_pe = ray_pe.to(device)
    model = VAE(dim_g=args.dim_g).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, args.epochs // 2), gamma=0.3)
    warm_eps = 5
    print("model params=%.1fM" % (n_params / 1e6), flush=True)

    def eval_val():
        """Two readouts: reparam (training-time sampling) and mu (deterministic
        posterior mean). Both matter: mu-decode tells whether the encoder has
        learned a discriminative latent at all; reparam-decode adds sampling
        noise. Large mu-vs-reparam gap => posterior sigma too wide."""
        model.eval()
        outs = {"mu": [], "reparam": []}
        with torch.no_grad():
            for n in val_names:
                sh, cov, peak, rmax = va[n]
                mu, lv = model.encode(sh[None].to(device))
                axis = torch.arange(N_BINS)
                m = cov >= 0.02
                gtn = peak[m] / rmax
                for tag, z in (("mu", mu),
                               ("reparam", mu + torch.exp(0.5 * lv)
                                * torch.randn_like(mu))):
                    pred = torch.sigmoid(model.decode(z, ray_pe))[0].cpu()
                    soft = (pred * axis[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
                    outs[tag].append(float((pred[m] - sh[m]).abs().mean()))
                    outs[tag].append(float(
                        (soft[m] / (N_BINS - 1) - gtn).abs().median()))
        model.train()
        r = {}
        for tag, v in outs.items():
            r[tag + "_profL1"] = np.median(v[0::2])
            r[tag + "_dmeds"] = np.median(v[1::2])
        return r

    for ep in range(args.epochs):
        model.train()
        tot_rec, tot_kl, nb = 0.0, 0.0, 0
        beta = args.beta * min(1.0, (ep + 1) / warm_eps)
        order = torch.randperm(len(train_names))
        for b0 in range(0, len(train_names), args.batch):
            bn = [train_names[i] for i in order[b0:b0 + args.batch]]
            sh = torch.stack([tr[n][0] for n in bn]).to(device)
            cov = torch.stack([tr[n][1] for n in bn]).to(device)
            pred, mu, lv = model(sh, ray_pe)
            w = (cov + 0.05).clamp(max=1.0).unsqueeze(-1)
            rec = (w * (pred - sh).abs()).mean()
            kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(-1).mean()
            loss = rec + beta * kl
            if args.aux:
                # Differentiable morphology term. centroid (soft-argmax) gets
                # gradients on EVERY bin (no D7 peak-bin freeze). 'peak' mode
                # adds |1 - pred.max| height: the measured core defect is that
                # pred peaks sit ~0.4 vs GT maxnorm peaks at 1.0 (width ratio
                # and soft-depth error both stem from this flattening).
                bins = torch.arange(N_BINS, device=sh.device).float()
                sp = (pred * bins).sum(-1) / (pred.sum(-1) + 1e-6)
                sg = (sh * bins).sum(-1) / (sh.sum(-1) + 1e-6)
                wm = w.squeeze(-1)
                pos = (wm * (sp - sg).abs()).mean() / N_BINS
                if args.aux_mode == "peak":
                    aux = pos + (wm * (1.0 - pred.max(-1).values).abs()).mean()
                else:  # std (negative result; kept for ablation record)
                    ex2p = (pred * bins * bins).sum(-1) / (pred.sum(-1) + 1e-6)
                    ex2g = (sh * bins * bins).sum(-1) / (sh.sum(-1) + 1e-6)
                    sigp = (ex2p - sp * sp).clamp(min=0).sqrt()
                    sigg = (ex2g - sg * sg).clamp(min=0).sqrt()
                    aux = pos + (wm * (sigp - sigg).abs()).mean() / N_BINS
                loss = loss + args.aux_lam * aux
            opt.zero_grad(); loss.backward(); opt.step()
            tot_rec += rec.item(); tot_kl += kl.item(); nb += 1
        sched.step()
        r = eval_val()
        mu2 = 0.5 * (mu.pow(2)).sum(-1).mean().item()
        sig2 = 0.5 * (lv.exp() - 1 - lv).sum(-1).mean().item()
        print("epoch %2d rec=%.4f kl=%.4f beta=%.1e [mu2=%.3f sig2=%.3f] | "
              "val mu:   prof-L1=%.4f dmeds=%.4f | "
              "val reparam: prof-L1=%.4f dmeds=%.4f"
              % (ep + 1, tot_rec / nb, tot_kl / nb, beta, mu2, sig2,
                 r["mu_profL1"], r["mu_dmeds"],
                 r["reparam_profL1"], r["reparam_dmeds"]), flush=True)

    tag = "_aux%.1f_%s" % (args.aux_lam, args.aux_mode) if args.aux else ""
    outdir = os.path.join(ROOT, "m1_vae_dg%d_b%s%s"
                          % (args.dim_g, ("%.0e" % args.beta).replace("-0", "-"),
                             tag))
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"dim_g": args.dim_g, "beta": args.beta, "epochs": args.epochs,
                   "n_train": len(train_names), "n_val": len(val_names),
                   "loss": "v3-L1+KL", "target": "maxnorm",
                   "aux": args.aux, "aux_lam": args.aux_lam if args.aux else None,
                   "aux_mode": args.aux_mode if args.aux else None,
                   "train_names": train_names, "val_names": val_names}, f)
    print("done. checkpoint+meta -> %s/" % outdir, flush=True)


if __name__ == "__main__":
    main()
