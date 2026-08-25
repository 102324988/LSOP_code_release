#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3a diagnosis: where does the conditional-sampling dmed come from?

Hypothesis: M3a decodes through the frozen M1 decoder, so its precision
ceiling is M1's unconditional reconstruction (0.0419), NOT the D8
discriminative 0.0338. This script answers three questions on val:

  1. Multi-sample averaging: does the conditional posterior mean (average
     of N samples) approach the M1 0.0419 ceiling? Report both per-object
     best-of-N and averaged-profile dmed.
  2. Condition effectiveness: uncond sampling (c=0) vs conditional -- is the
     condition actually steering z, and by how much on dmed?
  3. DDIM steps: 50 vs 200 -- is 50-step sampling leaving the posterior
     unconverged?

Usage: python m3a_diag.py --m3a m3a_cond --out m3a_diag_val [--N 8]
"""
import argparse
import json
import os

import numpy as np
import torch

from m1_train_vae import VAE, load_object, ray_encodings
from m2_train import get_beta_schedule
from m3a_train_cond import CondTimeMLP, encode_cond, load_views

ROOT = "/root/e0lab/e0"


@torch.no_grad()
def ddim_sample(den, alpha_bar, c, dim_g, steps=50, eta=0.0, device="cpu"):
    """Conditional DDIM (v-prediction), batched over objects x N samples."""
    den.eval()
    T = len(alpha_bar)
    ts = np.linspace(T - 1, 0, steps).astype(np.int64)
    n = c.shape[0]
    x = torch.randn(n, dim_g, device=device)
    for i, t in enumerate(ts):
        t_prev = 0 if i == steps - 1 else int(ts[i + 1])
        a_t = alpha_bar[t].item()
        a_tp = alpha_bar[t_prev].item()
        t_frac = torch.full((n,), t / (T - 1), dtype=torch.float32, device=device)
        v = den(x, c, t_frac)
        x0 = (np.sqrt(a_t) * x - np.sqrt(1.0 - a_t) * v).clamp(-6.0, 6.0)
        eps = np.sqrt(1.0 - a_t) * x + np.sqrt(a_t) * v
        if i < steps - 1:
            sigma = eta * np.sqrt((1.0 - a_tp) / (1.0 - a_t)) * \
                    np.sqrt(max(0.0, 1.0 - a_t / a_tp))
            x = np.sqrt(a_tp) * x0 + np.sqrt(1.0 - a_tp - sigma ** 2) * eps \
                + sigma * torch.randn_like(x)
        else:
            x = x0
    return x


def dmed_of(pred, gt_sh, gt_cov, gt_peak, rmax):
    """soft-argmax depth med vs GT, single profile (8192,96)."""
    m = gt_cov.numpy() >= 0.02
    axis = np.arange(96)
    soft = (pred[m] * axis[None, :]).sum(-1) / (pred[m].sum(-1) + 1e-6)
    gtn = gt_peak.numpy()[m] / rmax
    return float(np.median(np.abs(soft / (96 - 1) - gtn)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3a", default="m3a_cond")
    ap.add_argument("--out", default="m3a_diag_val")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--N", type=int, default=8)
    args = ap.parse_args()

    m3a = json.load(open(os.path.join(ROOT, args.m3a, "meta.json")))
    m1 = json.load(open(os.path.join(ROOT, m3a["m1_ckpt"], "meta.json")))
    dim_g = m3a["dim_g"]
    T = m3a["T"]
    names = m1["test_names"] if args.split == "test" else m1["val_names"]
    print("split=%s n=%d N=%d" % (args.split, len(names), args.N), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, _ = ray_encodings()

    model = VAE(dim_g=dim_g)
    model.load_state_dict(torch.load(os.path.join(ROOT, m3a["m1_ckpt"], "model.pt"),
                                     map_location="cpu"))
    model.eval().to(device)
    ray_pe = ray_pe.to(device)

    from d8_train_full import Model as D8Model
    d8 = D8Model(fusion="mean_pool")
    d8.load_state_dict(torch.load(os.path.join(ROOT, m3a["d8_ckpt"], "model.pt"),
                                  map_location="cpu"))
    d8.eval().to(device)

    from multiprocessing.dummy import Pool
    with Pool(8) as pool:
        profs = dict(pool.map(lambda n: (n, load_object(n)), names, chunksize=4))
        views = dict(pool.map(lambda n: (n, load_views(n)), names, chunksize=4))
    gt, conds = {}, []
    with torch.no_grad():
        for n in names:
            imgs, feats = views[n]
            c = encode_cond(d8, imgs[None].to(device), feats[None].to(device))
            conds.append(c[0].cpu())
            gt[n] = profs[n]
    C = torch.stack(conds).to(device)                       # (n, 512)

    _, _, alpha_bar = get_beta_schedule(T)
    den = CondTimeMLP(dim_g, C.shape[1])
    den.load_state_dict(torch.load(os.path.join(ROOT, args.m3a, "denoiser.pt"),
                                   map_location="cpu"))
    den.to(device)
    z_mean = np.load(os.path.join(ROOT, args.m3a, "z_mean.npy")).astype(np.float32)
    z_std = np.load(os.path.join(ROOT, args.m3a, "z_std.npy")).astype(np.float32)
    CN = C.repeat_interleave(args.N, dim=0)                 # (n*N, 512)

    torch.manual_seed(7)
    axis = np.arange(96)
    results = {}
    for tag, cuse, steps in [("cond_50", CN, 50), ("cond_200", CN, 200),
                             ("uncond_50", torch.zeros_like(CN), 50)]:
        z_white = ddim_sample(den, alpha_bar, cuse, dim_g, steps, 0.0, device)
        z = (z_white.cpu().numpy() * z_std + z_mean).astype(np.float32)
        with torch.no_grad():
            pred = torch.sigmoid(model.decode(torch.from_numpy(z).to(device),
                                              ray_pe)).cpu().numpy()
        pred = pred.reshape(args.N, len(names), 8192, 96)
        # per-object stats over the N samples
        best, mean_dm = [], []
        for j, n in enumerate(names):
            sh, cov, peak, rmax = gt[n]
            dj = [dmed_of(pred[k, j], sh, cov, peak, rmax) for k in range(args.N)]
            best.append(float(np.min(dj)))
            avg = pred[:, j].mean(0)
            mean_dm.append(dmed_of(avg, sh, cov, peak, rmax))
        results[tag] = {
            "best_of_N_med": float(np.median(best)),
            "avg_profile_med": float(np.median(mean_dm)),
            "avg_profile_mean": float(np.mean(mean_dm)),
        }
        print("%-10s best_of_N med=%.4f  avg_profile med=%.4f mean=%.4f"
              % (tag, results[tag]["best_of_N_med"],
                 results[tag]["avg_profile_med"],
                 results[tag]["avg_profile_mean"]), flush=True)

    results["_refs"] = {"M1_uncond": 0.0419, "D8_discr": 0.0338,
                        "gate": 0.040, "N": args.N, "split": args.split}
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    json.dump(results, open(os.path.join(ROOT, args.out, "diag.json"), "w"),
              indent=1)
    print("saved -> %s/diag.json" % args.out, flush=True)


if __name__ == "__main__":
    main()
