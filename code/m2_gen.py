#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M2 unconditional generation: DDIM-sample z -> frozen M1 decoder -> profiles.

Evaluates (same protocol as m1_gen.py so numbers are comparable):
  - AUTHENTICITY: generated-profile FWHM distribution vs GT vs M1-recon
    (gen~recon is the "sampling on-manifold" claim; M1 Gaussian-prior gen
    FWHM was 13 vs GT 4)
  - DIVERSITY: pairwise Chamfer among generated point clouds (collapsed
    generator => near-zero distance)
  - RADIUS: p50/p90 of generated clouds vs GT range
  - LATENT diagnostic: PCA of sampled z vs training mu (on-manifold check)

Usage: python m2_gen.py -> m2_gen/{summary.json,figs/*.png}
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
from m2_train import TimeMLP, get_beta_schedule

ROOT = "/root/e0lab/e0"
M1_CKPT = "m1_vae_dg1024_b1e-4"
M1_PRED_VAL = os.path.join(ROOT, "m1_vae_dg1024_b1e-4_val")
M2 = os.path.join(ROOT, "m2_latent_diff")
OUT = os.path.join(ROOT, "m2_gen")
N_GEN = 16
SEED = 7
DDIM_STEPS = 50
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


@torch.no_grad()
def ddim_sample(den, alpha_bar, n, dim_g, steps=DDIM_STEPS, target="x0",
                eta=0.0, device="cpu"):
    """DDIM over a linear timestep grid. 'x0' target never divides by
    sqrt(a_bar) (numerically stable on near-Gaussian latent); 'eps' target
    clips the reconstructed x0 instead."""
    den.eval()
    T = len(alpha_bar)
    ts = np.linspace(T - 1, 0, steps).astype(np.int64)
    x = torch.randn(n, dim_g, device=device)
    for i, t in enumerate(ts):
        t_prev = 0 if i == steps - 1 else int(ts[i + 1])
        a_t = alpha_bar[t].item()
        a_tp = alpha_bar[t_prev].item()
        t_frac = torch.full((n,), t / (T - 1), dtype=torch.float32, device=device)
        out = den(x, t_frac)
        if target == "x0":
            x0 = out.clamp(-6.0, 6.0)
            eps = (x - np.sqrt(a_t) * x0) / np.sqrt(1.0 - a_t)
        elif target == "eps":
            eps = out
            x0 = ((x - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)).clamp(-6.0, 6.0)
        else:  # v
            v = out
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2", default="m2_latent_diff",
                    help="dir holding denoiser.pt + meta.json + whiten stats")
    ap.add_argument("--out", default="m2_gen")
    args = ap.parse_args()
    M2 = os.path.join(ROOT, args.m2)
    OUT = os.path.join(ROOT, args.out)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    rng = np.random.RandomState(SEED)

    m2 = json.load(open(os.path.join(M2, "meta.json")))
    m1 = json.load(open(os.path.join(ROOT, m2["m1_ckpt"], "meta.json")))
    dim_g = m2["dim_g"]
    T = m2["T"]

    # rmax reference: median over training-object metas (same as m1_gen)
    rmaxs = []
    for n in m2["train_names"][::40]:
        rm = json.load(open(os.path.join(D4, n, "profiles", "meta.json")))
        rmaxs.append(rm["rmax"])
    rmax_ref = float(np.median(rmaxs))
    print("rmax_ref=%.3f (median over train)" % rmax_ref)

    # frozen M1 decoder
    ray_pe, _ = ray_encodings()
    model = VAE(dim_g=dim_g)
    model.load_state_dict(torch.load(os.path.join(ROOT, m2["m1_ckpt"], "model.pt"),
                                     map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ray_pe = model.to(device), ray_pe.to(device)

    # DDIM sample -> inverse whiten
    _, _, alpha_bar = get_beta_schedule(T)
    den = TimeMLP(dim_g)
    den.load_state_dict(torch.load(os.path.join(M2, "denoiser.pt"),
                                   map_location="cpu"))
    den.to(device)
    z_mean = np.load(os.path.join(M2, "z_mean.npy")).astype(np.float32)
    z_std = np.load(os.path.join(M2, "z_std.npy")).astype(np.float32)
    z_white = ddim_sample(den, alpha_bar, N_GEN, dim_g, DDIM_STEPS,
                          m2.get("target", "eps"), 0.0, device)
    z = (z_white.cpu().numpy() * z_std + z_mean).astype(np.float32)
    print("sampled z: |z|_dim mean=%.3f std=%.3f"
          % (float(np.abs(z).mean()), float(z.std())))

    with torch.no_grad():
        pred = torch.sigmoid(model.decode(
            torch.from_numpy(z).to(device), ray_pe)).cpu().numpy()  # (N,R,96)

    dirs = dirs_grid()
    pmax = pred.max(-1)
    print("generated pred.max(): med=%.3f p25=%.3f p75=%.3f"
          % (np.median(pmax), np.percentile(pmax, 25), np.percentile(pmax, 75)))
    soft = (pred * np.arange(N_BINS)[None, None, :]).sum(-1) / \
           (pred.sum(-1) + 1e-6)
    r = soft / (N_BINS - 1) * rmax_ref
    keep = pmax > 0.3
    pcs, gen_fwhm, rs_p50, rs_p90 = [], [], [], []
    for i in range(N_GEN):
        k = keep[i]
        if k.sum() < 100:
            k = np.ones(pred.shape[1], dtype=bool)
        pc = (r[i, k, None] * dirs[k]).astype(np.float32)
        pcs.append(pc)
        gen_fwhm.append(fwhm(pred[i][k]))
        rs_p50.append(np.median(r[i][k]))
        rs_p90.append(np.percentile(r[i][k], 90))
        print("gen %2d: %5d rays kept, radius p50=%.3f p90=%.3f"
              % (i, k.sum(), rs_p50[-1], rs_p90[-1]))
    gen_fwhm = np.concatenate(gen_fwhm)

    # GT fwhm + radius references (val sample, same as m1_gen)
    val_names = m1["val_names"][::9]
    gt_fwhm, gt_rad_p50, gt_rad_p90 = [], [], []
    bins_a = np.arange(N_BINS)
    for n in val_names:
        gt = np.load(os.path.join(D4, n, "profiles", "profiles.npy")).astype(np.float32)
        gtn = gt.reshape(-1, N_BINS) / np.maximum(
            gt.reshape(-1, N_BINS).max(-1, keepdims=True), 1e-6)
        cov = np.load(os.path.join(D4, n, "profiles", "coverage.npy")).astype(np.float32)
        m = cov >= 0.02
        fw = fwhm(gtn[m])
        gt_fwhm.append(fw[fw >= 2])
        soft = (gtn[m] * bins_a[None, :]).sum(-1) / (gtn[m].sum(-1) + 1e-6)
        r_gt = soft / (N_BINS - 1) * rmax_ref
        gt_rad_p50.append(np.median(r_gt))
        gt_rad_p90.append(np.percentile(r_gt, 90))
    gt_fwhm = np.concatenate(gt_fwhm)

    # GT diversity reference: pairwise chamfer among GT val point clouds
    # (the 0.408 from M1 Gaussian-prior was inflated by degenerate wide
    # profiles; what matters is whether M2 gen diversity reaches GT level)
    gt_pcs = []
    for n in m1["val_names"][:16]:
        gt = np.load(os.path.join(D4, n, "profiles", "profiles.npy")).astype(np.float32)
        gtn = gt.reshape(-1, N_BINS) / np.maximum(
            gt.reshape(-1, N_BINS).max(-1, keepdims=True), 1e-6)
        cov = np.load(os.path.join(D4, n, "profiles", "coverage.npy")).astype(np.float32)
        m = cov >= 0.02
        soft = (gtn[m] * bins_a[None, :]).sum(-1) / (gtn[m].sum(-1) + 1e-6)
        r_gt = soft / (N_BINS - 1) * rmax_ref
        gt_pcs.append((r_gt[:, None] * dirs[m]).astype(np.float32))
    Dg = pairwise_chamfer(gt_pcs)
    go = Dg[np.triu_indices(16, 1)]

    # M1-recon fwhm reference (a sample of val pred profiles): the level a
    # perfect p(z) should reach (decoder info-bottleneck bound).
    rnames = sorted({f[:-9] for f in os.listdir(M1_PRED_VAL)
                     if f.endswith("_prof.npy")})[::9]
    recon_fwhm = []
    for n in rnames:
        pr = np.load(os.path.join(M1_PRED_VAL, n + "_prof.npy")).astype(np.float32)
        cov = np.load(os.path.join(M1_PRED_VAL, n + "_cov.npy")).astype(np.float32)
        fw = fwhm(pr[cov >= 0.02])
        recon_fwhm.append(fw)
    recon_fwhm = np.concatenate(recon_fwhm)

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
        "recon_fwhm_med": float(np.median(recon_fwhm)),
        "gen_rays_kept_med": int(np.median([len(p) for p in pcs])),
        "gen_radius_p50_med": float(np.median(rs_p50)),
        "gen_radius_p90_med": float(np.median(rs_p90)),
        "gt_radius_p50_med": float(np.median(gt_rad_p50)),
        "gt_radius_p90_med": float(np.median(gt_rad_p90)),
        "gt_pairwise_chamfer_med": float(np.median(go)),
        "gt_pairwise_chamfer_min": float(go.min()),
    }
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    np.save(os.path.join(OUT, "z_sampled.npy"), z)

    # ---- figures ----
    # 1) 2D projections + FWHM distribution + pairwise chamfer (m1_gen style)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for i in (0, 3, 7, 11):
        pc = pcs[i]
        if len(pc) == 0:
            continue
        axs[0].scatter(pc[::4, 0], pc[::4, 2], s=1, alpha=0.4, label="gen %d" % i)
    axs[0].set_aspect("equal")
    axs[0].set_title("generated point clouds (x-z projection)")
    axs[0].legend(fontsize=7, markerscale=3)
    b = np.linspace(0, 40, 41)
    axs[1].hist(gt_fwhm, bins=b, density=True, alpha=0.6, color="#0072B2",
                label="GT (val sample)")
    axs[1].hist(recon_fwhm, bins=b, density=True, alpha=0.5, color="#009E73",
                label="M1 recon")
    axs[1].hist(gen_fwhm, bins=b, density=True, alpha=0.5, color="#D55E00",
                label="M2 gen")
    axs[1].set_xlabel("profile FWHM (bins)")
    axs[1].set_title("authenticity: width distribution")
    axs[1].legend(fontsize=8)
    im = axs[2].imshow(D, cmap="viridis")
    axs[2].set_title("pairwise Chamfer (diversity)")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig_gen.png"), dpi=150)
    plt.close(fig)

    # 2) latent on-manifold check: PCA of train mu vs sampled z
    mu_all = np.load(os.path.join(M2, "mu_all.npy")).astype(np.float32)
    Zc = np.concatenate([mu_all, z], axis=0).astype(np.float64)
    Zc -= Zc.mean(0, keepdims=True)
    _, _, V = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ V[:2].T
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(P[:len(mu_all), 0], P[:len(mu_all), 1], s=4, alpha=0.5,
               color="#0072B2", label="train mu (819)")
    ax.scatter(P[len(mu_all):, 0], P[len(mu_all):, 1], s=30, marker="*",
               color="#D55E00", label="M2 sampled z (16)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("latent on-manifold check: sampled z vs train mu")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig_latent_pca.png"), dpi=150)
    plt.close(fig)
    print("saved -> %s/" % OUT)


if __name__ == "__main__":
    main()
