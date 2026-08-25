#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M3a CFG diagnosis: can classifier-free guidance amplify the condition?

The m3a_diag result showed cond vs uncond sampling nearly identical
(best 0.039 vs 0.038, avg_profile 0.056 vs 0.055) -- the condition is NOT
steering the sampled z. CFG (v_uncond + w*(v_cond-v_uncond)) tests whether
the denoiser learned ANY usable condition signal: if it did, raising w
should pull the posterior mean toward the object and lower avg_profile dmed
toward the M1 0.0419 ceiling; if it did not, w just amplifies noise and avg
gets worse or stays flat.

Usage: python m3a_cfg.py --m3a m3a_cond --out m3a_cfg_val --N 8
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
def ddim_sample_cfg(den, alpha_bar, c, dim_g, w, steps=50, device="cpu"):
    den.eval()
    T = len(alpha_bar)
    ts = np.linspace(T - 1, 0, steps).astype(np.int64)
    n = c.shape[0]
    x = torch.randn(n, dim_g, device=device)
    zc = torch.zeros_like(c)
    for i, t in enumerate(ts):
        t_prev = 0 if i == steps - 1 else int(ts[i + 1])
        a_t = alpha_bar[t].item()
        a_tp = alpha_bar[t_prev].item()
        t_frac = torch.full((n,), t / (T - 1), dtype=torch.float32, device=device)
        v_c = den(x, c, t_frac)
        v_u = den(x, zc, t_frac)
        v = v_u + w * (v_c - v_u)
        x0 = (np.sqrt(a_t) * x - np.sqrt(1.0 - a_t) * v).clamp(-6.0, 6.0)
        eps = np.sqrt(1.0 - a_t) * x + np.sqrt(a_t) * v
        if i < steps - 1:
            x = np.sqrt(a_tp) * x0 + np.sqrt(1.0 - a_tp) * eps
        else:
            x = x0
    return x


def dmed_of(pred, gt_cov, gt_peak, rmax):
    m = gt_cov.numpy() >= 0.02
    axis = np.arange(96)
    soft = (pred[m] * axis[None, :]).sum(-1) / (pred[m].sum(-1) + 1e-6)
    gtn = gt_peak.numpy()[m] / rmax
    return float(np.median(np.abs(soft / (96 - 1) - gtn)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3a", default="m3a_cond")
    ap.add_argument("--out", default="m3a_cfg_val")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--ws", default="0,1,2,4,8")
    args = ap.parse_args()

    ws = [float(w) for w in args.ws.split(",")]
    m3a = json.load(open(os.path.join(ROOT, args.m3a, "meta.json")))
    m1 = json.load(open(os.path.join(ROOT, m3a["m1_ckpt"], "meta.json")))
    dim_g = m3a["dim_g"]
    T = m3a["T"]
    names = m1["test_names"] if args.split == "test" else m1["val_names"]
    print("split=%s n=%d N=%d ws=%s" % (args.split, len(names), args.N, ws),
          flush=True)

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
    C = torch.stack(conds).to(device)

    _, _, alpha_bar = get_beta_schedule(T)
    den = CondTimeMLP(dim_g, C.shape[1])
    den.load_state_dict(torch.load(os.path.join(ROOT, args.m3a, "denoiser.pt"),
                                   map_location="cpu"))
    den.to(device)
    z_mean = np.load(os.path.join(ROOT, args.m3a, "z_mean.npy")).astype(np.float32)
    z_std = np.load(os.path.join(ROOT, args.m3a, "z_std.npy")).astype(np.float32)
    CN = C.repeat_interleave(args.N, dim=0)

    torch.manual_seed(7)
    results = {}
    for w in ws:
        z_white = ddim_sample_cfg(den, alpha_bar, CN, dim_g, w, 50, device)
        z = (z_white.cpu().numpy() * z_std + z_mean).astype(np.float32)
        with torch.no_grad():
            pred = torch.sigmoid(model.decode(torch.from_numpy(z).to(device),
                                              ray_pe)).cpu().numpy()
        pred = pred.reshape(args.N, len(names), 8192, 96)
        best, avg = [], []
        for j, n in enumerate(names):
            sh, cov, peak, rmax = gt[n]
            dj = [dmed_of(pred[k, j], cov, peak, rmax) for k in range(args.N)]
            best.append(float(np.min(dj)))
            avg.append(dmed_of(pred[:, j].mean(0), cov, peak, rmax))
        results["w=%.1f" % w] = {
            "best_of_N_med": float(np.median(best)),
            "avg_profile_med": float(np.median(avg)),
            "avg_profile_mean": float(np.mean(avg)),
        }
        print("w=%.1f  best_of_N med=%.4f  avg_profile med=%.4f mean=%.4f"
              % (w, results["w=%.1f" % w]["best_of_N_med"],
                 results["w=%.1f" % w]["avg_profile_med"],
                 results["w=%.1f" % w]["avg_profile_mean"]), flush=True)

    results["_refs"] = {"M1_uncond": 0.0419, "D8_discr": 0.0338,
                        "gate": 0.040, "N": args.N, "split": args.split}
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    json.dump(results, open(os.path.join(ROOT, args.out, "cfg.json"), "w"),
              indent=1)
    print("saved -> %s/cfg.json" % args.out, flush=True)


if __name__ == "__main__":
    main()
