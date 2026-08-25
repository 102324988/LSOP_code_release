#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3a: conditional sampling + generative-reconstruction evaluation.

For each val object: c = frozen D8 encoder (8 fixed views) -> DDIM sample
z ~ p(z|c) -> frozen M1 decoder -> profile -> soft-argmax depth. Reports the
authoritative depth-med vs GT (d7_eval_fixed protocol), compared against:
  D8 discriminative   val 0.0338 / test 0.0384
  M1 unconditional    val 0.0419
Gate: val dmeds <= 0.040.

Usage: python m3a_gen_cond.py --m3a m3a_cond --out m3a_cond_val
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
DDIM_STEPS = 50


@torch.no_grad()
def ddim_sample(den, alpha_bar, c, dim_g, steps=DDIM_STEPS, eta=0.0,
                device="cpu"):
    """Conditional DDIM (v-prediction, M2-final sampler + c channel)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3a", default="m3a_cond")
    ap.add_argument("--out", default="m3a_cond_val")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()

    m3a = json.load(open(os.path.join(ROOT, args.m3a, "meta.json")))
    m1 = json.load(open(os.path.join(ROOT, m3a["m1_ckpt"], "meta.json")))
    dim_g = m3a["dim_g"]
    T = m3a["T"]
    names = m1["test_names"] if args.split == "test" else m1["val_names"]
    print("split=%s n=%d" % (args.split, len(names)), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ray_pe, _ = ray_encodings()

    # frozen M1 decoder
    model = VAE(dim_g=dim_g)
    model.load_state_dict(torch.load(os.path.join(ROOT, m3a["m1_ckpt"], "model.pt"),
                                     map_location="cpu"))
    model.eval().to(device)
    ray_pe = ray_pe.to(device)

    # frozen D8 condition encoder
    from d8_train_full import Model as D8Model
    d8 = D8Model(fusion="mean_pool")
    d8.load_state_dict(torch.load(os.path.join(ROOT, m3a["d8_ckpt"], "model.pt"),
                                  map_location="cpu"))
    d8.eval().to(device)

    # condition encoder + GT for all objects
    gt, conds = {}, []
    from multiprocessing.dummy import Pool
    with Pool(8) as pool:
        profs = dict(pool.map(lambda n: (n, load_object(n)), names, chunksize=4))
        views = dict(pool.map(lambda n: (n, load_views(n)), names, chunksize=4))
    with torch.no_grad():
        for n in names:
            imgs, feats = views[n]
            c = encode_cond(d8, imgs[None].to(device), feats[None].to(device))
            conds.append(c[0].cpu())
            gt[n] = profs[n]  # (sh, cov, peak, rmax)
    C = torch.stack(conds).to(device)          # (n, 512)

    # conditional DDIM sampling
    _, _, alpha_bar = get_beta_schedule(T)
    den = CondTimeMLP(dim_g, C.shape[1])
    den.load_state_dict(torch.load(os.path.join(ROOT, args.m3a, "denoiser.pt"),
                                   map_location="cpu"))
    den.to(device)
    z_mean = np.load(os.path.join(ROOT, args.m3a, "z_mean.npy")).astype(np.float32)
    z_std = np.load(os.path.join(ROOT, args.m3a, "z_std.npy")).astype(np.float32)
    z_white = ddim_sample(den, alpha_bar, C, dim_g, DDIM_STEPS, 0.0, device)
    z = (z_white.cpu().numpy() * z_std + z_mean).astype(np.float32)
    print("sampled z: |z|_dim mean=%.3f std=%.3f"
          % (float(np.abs(z).mean()), float(z.std())), flush=True)

    # decode -> soft-argmax depth vs GT (d7_eval_fixed protocol)
    with torch.no_grad():
        pred = torch.sigmoid(model.decode(torch.from_numpy(z).to(device),
                                          ray_pe)).cpu().numpy()
    axis = np.arange(96)
    per, dmeds = [], []
    for i, n in enumerate(names):
        sh, cov, peak, rmax = gt[n]
        m = cov.numpy() >= 0.02
        soft = (pred[i, m] * axis[None, :]).sum(-1) / (pred[i, m].sum(-1) + 1e-6)
        gtn = peak.numpy()[m] / rmax
        d = np.abs(soft / (96 - 1) - gtn)
        per.append({"name": n, "dmed_s": float(np.median(d))})
        dmeds.append(float(np.median(d)))
    stats = {"split": args.split, "n": len(names),
             "dmed_s_med": float(np.median(dmeds)),
             "dmed_s_mean": float(np.mean(dmeds)),
             "per_object": per}
    print("M3a %s: dmed_s med=%.4f mean=%.4f (D8 0.0338/0.0384, M1 0.0419)"
          % (args.split, stats["dmed_s_med"], stats["dmed_s_mean"]), flush=True)
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    json.dump(stats, open(os.path.join(ROOT, args.out, "summary.json"), "w"),
              indent=1)
    np.save(os.path.join(ROOT, args.out, "z_sampled.npy"), z)
    print("saved -> %s/" % args.out, flush=True)


if __name__ == "__main__":
    main()
