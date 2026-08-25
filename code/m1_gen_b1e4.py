#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 unconditional generation probe (M2 seed).

Samples z ~ N(0,1) (the VAE prior), decodes to a profile field, inverts to a
point cloud (soft-argmax depth x dirs), and checks:
  - DIVERSITY: pairwise Chamfer among generated point clouds (a collapsed
    generator produces ~identical clouds => near-zero distance)
  - AUTHENTICITY: FWHM distribution of generated profiles vs GT profiles
    (smoothed profiles => much wider than GT)

Usage: python m1_gen.py -> m1_gen/{summary.json,figs/*.png}
"""
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m1_train_vae import (D4, N_PHI, N_THETA, N_BINS, VAE, ray_encodings)

ROOT = "/root/e0lab/e0"
OUT = os.path.join(ROOT, "m1_gen_b1e4")
N_GEN = 10
SEED = 7
N_BINS = 96


def dirs_grid():
    th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                     np.cos(TH)], axis=-1).reshape(-1, 3).astype(np.float32)


def fwhm(prof):
    pk = prof.argmax(-1)
    hm = prof.max(-1) * 0.5
    w = np.zeros(len(prof), dtype=np.float32)
    for i in range(len(prof)):
        p = prof[i]
        lo, hi = pk[i], pk[i]
        while lo > 0 and p[lo - 1] >= hm[i]:
            lo -= 1
        while hi < N_BINS - 1 and p[hi + 1] >= hm[i]:
            hi += 1
        w[i] = hi - lo
    return w


def pairwise_chamfer(pcs, sample=3000):
    """Mean symmetric chamfer between every pair of (downsampled) clouds."""
    n = len(pcs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pi = pcs[i] if len(pcs[i]) <= sample else \
                pcs[i][np.random.choice(len(pcs[i]), sample, replace=False)]
            pj = pcs[j] if len(pcs[j]) <= sample else \
                pcs[j][np.random.choice(len(pcs[j]), sample, replace=False)]
            kd1, kd2 = cKDTree(pi), cKDTree(pj)
            d1, _ = kd2.query(pi)
            d2, _ = kd1.query(pj)
            D[i, j] = D[j, i] = float((d1.mean() + d2.mean()) / 2)
    return D


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    rng = np.random.RandomState(SEED)

    # reference scale rmax: median over training-object metas
    meta = json.load(open(os.path.join(ROOT, "m1_vae_dg1024_b1e-4", "meta.json")))
    rmaxs = []
    for n in meta["train_names"][::40]:
        m = json.load(open(os.path.join(D4, n, "profiles", "meta.json")))
        rmaxs.append(m["rmax"])
    rmax_ref = float(np.median(rmaxs))
    print("rmax_ref=%.3f (median over train)" % rmax_ref)

    ray_pe, _ = ray_encodings()
    model = VAE(dim_g=meta["dim_g"])
    model.load_state_dict(torch.load(os.path.join(ROOT, "m1_vae_dg1024_b1e-4",
                                                  "model.pt")))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ray_pe = model.to(device), ray_pe.to(device)

    with torch.no_grad():
        z = torch.from_numpy(rng.randn(N_GEN, meta["dim_g"])).float().to(device)
        pred = torch.sigmoid(model.decode(z, ray_pe)).cpu().numpy()  # (N,R,96)

    dirs = dirs_grid()
    pmax = pred.max(-1)
    print("generated pred.max(): med=%.3f p25=%.3f p75=%.3f"
          % (np.median(pmax), np.percentile(pmax, 25), np.percentile(pmax, 75)))
    soft = (pred * np.arange(N_BINS)[None, None, :]).sum(-1) / \
           (pred.sum(-1) + 1e-6)
    r = soft / (N_BINS - 1) * rmax_ref
    keep = pmax > 0.3
    pcs = []
    gen_fwhm = []
    for i in range(N_GEN):
        k = keep[i]
        if k.sum() < 100:
            k = np.ones(pred.shape[1], dtype=bool)  # degenerate -> all rays
        pc = (r[i, k, None] * dirs[k]).astype(np.float32)
        pcs.append(pc)
        gen_fwhm.append(fwhm(pred[i][k]))
        print("gen %2d: %5d rays kept, radius p50=%.3f p90=%.3f"
              % (i, k.sum(), np.median(r[i][k]), np.percentile(r[i][k], 90)))
    gen_fwhm = np.concatenate(gen_fwhm)

    # GT fwhm reference: sample val objects
    val_names = meta["val_names"][::9]
    gt_fwhm = []
    for n in val_names:
        gt = np.load(os.path.join(D4, n, "profiles", "profiles.npy")).astype(np.float32)
        gtn = gt.reshape(-1, N_BINS) / np.maximum(
            gt.reshape(-1, N_BINS).max(-1, keepdims=True), 1e-6)
        cov = np.load(os.path.join(D4, n, "profiles", "coverage.npy")).astype(np.float32)
        m = cov >= 0.02
        fw = fwhm(gtn[m])
        gt_fwhm.append(fw[fw >= 2])
    gt_fwhm = np.concatenate(gt_fwhm)

    D = pairwise_chamfer(pcs)
    offdiag = D[np.triu_indices(N_GEN, 1)]
    stats = {
        "rmax_ref": rmax_ref,
        "n_gen": N_GEN,
        "pairwise_chamfer_med": float(np.median(offdiag)),
        "pairwise_chamfer_min": float(offdiag.min()),
        "pairwise_chamfer_p25": float(np.percentile(offdiag, 25)),
        "gen_fwhm_med": float(np.median(gen_fwhm)),
        "gt_fwhm_med": float(np.median(gt_fwhm)),
        "gen_rays_kept_med": int(np.median([len(p) for p in pcs])),
    }
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    # figures
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    # 2D projections of 3 representative clouds
    for i in (0, 3, 7):
        pc = pcs[i]
        if len(pc) == 0:
            continue
        axs[0].scatter(pc[::4, 0], pc[::4, 2], s=1, alpha=0.4,
                       label="gen %d" % i)
    axs[0].set_aspect("equal")
    axs[0].set_title("generated point clouds (x-z projection)")
    axs[0].legend(fontsize=7, markerscale=3)
    # FWHM distribution
    axs[1].hist(gt_fwhm, bins=np.linspace(0, 40, 41), density=True, alpha=0.6,
                color="#0072B2", label="GT (val sample)")
    axs[1].hist(gen_fwhm, bins=np.linspace(0, 40, 41), density=True, alpha=0.5,
                color="#D55E00", label="generated")
    axs[1].set_xlabel("profile FWHM (bins)")
    axs[1].set_title("authenticity: width distribution")
    axs[1].legend(fontsize=8)
    # pairwise chamfer matrix
    im = axs[2].imshow(D, cmap="viridis")
    axs[2].set_title("pairwise Chamfer (diversity)")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig_gen.png"), dpi=150)
    plt.close(fig)
    print("saved -> %s/" % OUT)


if __name__ == "__main__":
    main()
