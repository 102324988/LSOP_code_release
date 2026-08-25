#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 profile-shape diagnostics over the full val-90 set.

Answers the M1 success criterion that reconstruction does NOT collapse into
the training-mean profile (smearing): compares M1 reconstructed max-norm
profiles against GT max-norm per object, over all cov>=0.02 rays:
  - FWHM ratio (pred/GT) distribution, with the global-mean-profile FWHM
    ratio as the "collapse" reference
  - predicted peak-bin offset distribution
  - per-bin |pred - GT| (where does error sit; near-bin0 check)
  - representative-ray curves for 5 objects

Usage: python m1_diag.py  ->  m1_diag/{summary.json,figs/*.png}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
PRED = os.path.join(ROOT, "m1_vae_dg1024_b1e-4_val")
OUT = os.path.join(ROOT, "m1_diag_b1e4")
N_BINS = 96
REPS = ["Asus_Sabertooth_Z97", "Perricone_MD_Hypo", "Top_Paw_Dog_Bowl",
        "Nordic_Ware_Original", "My_Little_Pony_Princess"]


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


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    names = sorted({f[:-9] for f in os.listdir(PRED) if f.endswith("_prof.npy")})
    print("objects: %d" % len(names))

    # global-mean profile over the whole val set (proxy for collapse ref)
    all_gt = []
    gt_mean_prof = None
    rstats = {}
    bin_r = np.arange(N_BINS)
    all_absbin, all_peakoff, all_fwhm = [], [], []
    gt_absbin = []  # mean-profile abs error (collapse reference)
    n_rays = 0
    for n in names:
        pred = np.load(os.path.join(PRED, n + "_prof.npy")).astype(np.float32)
        cov = np.load(os.path.join(PRED, n + "_cov.npy")).astype(np.float32)
        gt = np.load(os.path.join(D4, n, "profiles", "profiles.npy")).astype(np.float32)
        gt = gt.reshape(-1, N_BINS)
        mx = gt.max(-1, keepdims=True)
        gtn = gt / np.maximum(mx, 1e-6)
        m = cov >= 0.02
        p, g = pred[m], gtn[m]
        absbin = np.abs(p - g).mean(0)
        peak_p, peak_g = p.argmax(-1), g.argmax(-1)
        fw_p, fw_g = fwhm(p), fwhm(g)
        good = fw_g >= 2
        all_absbin.append(absbin)
        all_peakoff.append(peak_p - peak_g)
        all_fwhm.append(fw_p[good] / np.maximum(fw_g[good], 1e-6))
        all_gt.append(g)
        n_rays += len(g)

    all_gt = np.concatenate(all_gt)
    mean_prof = all_gt.mean(0)                       # global-mean max-norm profile
    # collapse reference: |mean_prof - gtn| per object, and its FWHM ratio
    mean_ref_absbin, mean_ref_fwhm = [], []
    for n in names:
        pred = np.load(os.path.join(PRED, n + "_prof.npy")).astype(np.float32)
        cov = np.load(os.path.join(PRED, n + "_cov.npy")).astype(np.float32)
        gt = np.load(os.path.join(D4, n, "profiles", "profiles.npy")).astype(np.float32)
        gtn = (gt.reshape(-1, N_BINS) /
               np.maximum(gt.reshape(-1, N_BINS).max(-1, keepdims=True), 1e-6))
        m = cov >= 0.02
        mean_ref_absbin.append(np.abs(mean_prof[None, :] - gtn[m]).mean())
        fw_g = fwhm(gtn[m])
        fw_m = fwhm(np.broadcast_to(mean_prof, (gtn[m].shape[0], N_BINS)))
        good = fw_g >= 2
        mean_ref_fwhm.append(fw_m[good] / np.maximum(fw_g[good], 1e-6))

    absbin = np.array(all_absbin)
    fwhm_all = np.concatenate(all_fwhm)
    peakoff = np.concatenate(all_peakoff)
    med_absbin = absbin.mean(0)
    med_fwhm = np.median(fwhm_all)
    med_mean_fwhm = np.median(np.concatenate(mean_ref_fwhm))
    med_mean_abs = np.median(mean_ref_absbin)

    rstats = {
        "n_objects": len(names),
        "n_rays_cov02": int(n_rays),
        "median_fwhm_ratio_M1": float(med_fwhm),
        "median_fwhm_ratio_mean_profile": float(med_mean_fwhm),
        "fwhm_ratio_gt25": float(np.percentile(fwhm_all, 25)),
        "fwhm_ratio_gt75": float(np.percentile(fwhm_all, 75)),
        "peakoff_med": float(np.median(peakoff)),
        "peakoff_iqr": float(np.percentile(peakoff, 75) - np.percentile(peakoff, 25)),
        "perbin_abs_med_M1": float(np.median(absbin.mean(1))),
        "perbin_abs_med_mean_profile": float(med_mean_abs),
        "perbin_abs_bin0": float(med_absbin[0]),
        "perbin_abs_bin10": float(med_absbin[10]),
    }
    print(json.dumps(rstats, indent=1))
    json.dump(rstats, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    # ---- fig: FWHM ratio histogram (M1 vs mean-profile reference) ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(fwhm_all, bins=np.linspace(0, 6, 61), alpha=0.7,
               color="#0072B2", label="M1 recon")
    ax[0].axvline(1.0, color="gray", ls="--", lw=1)
    ax[0].axvline(1.5, color="#D55E00", ls="--", lw=1.4,
                  label="success bound 1.5")
    ax[0].axvline(med_mean_fwhm, color="#009E73", ls=":", lw=2,
                  label="mean-profile ref %.2f" % med_mean_fwhm)
    ax[0].set_xlabel("FWHM pred / GT")
    ax[0].set_title("profile width ratio (90 val objects)")
    ax[0].legend(fontsize=8)
    ax[0].set_xlim(0, 6)

    ax[1].hist(peakoff, bins=np.arange(-40, 41), alpha=0.7, color="#0072B2")
    ax[1].axvline(0, color="gray", ls="--", lw=1)
    ax[1].set_xlabel("peak-bin offset (pred - GT)")
    ax[1].set_title("predicted peak offset")

    # per-bin mean |pred - gt| vs mean-profile reference
    ax[2].plot(bin_r, med_absbin, lw=1.8, color="#0072B2", label="M1 recon")
    ax[2].plot(bin_r, np.repeat(med_mean_abs, N_BINS), lw=1.2, ls="--",
               color="#009E73", label="mean-profile ref %.3f" % med_mean_abs)
    ax[2].set_xlabel("radial bin")
    ax[2].set_ylabel("mean |pred - GT|")
    ax[2].set_title("per-bin profile error (cov>=0.02)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig_shape_stats.png"), dpi=150)
    plt.close(fig)

    # ---- fig: representative-ray curves ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    reps = [n for n in names for p in REPS if n.startswith(p)]
    for ax, n in zip(axes[:len(reps)], reps):
        pred = np.load(os.path.join(PRED, n + "_prof.npy")).astype(np.float32)
        cov = np.load(os.path.join(PRED, n + "_cov.npy")).astype(np.float32)
        gt = np.load(os.path.join(D4, n, "profiles", "profiles.npy")).astype(np.float32)
        gtn = (gt.reshape(-1, N_BINS) /
               np.maximum(gt.reshape(-1, N_BINS).max(-1, keepdims=True), 1e-6))
        m = cov >= 0.02
        soft = (pred * bin_r[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
        err = np.abs(soft[m] - np.median(soft[m]))  # typical (median-depth) ray
        r = int(np.abs(err - np.median(err)).argmin())
        rr = np.flatnonzero(m)[r]
        ax.plot(bin_r, pred[rr], lw=1.6, color="#0072B2", label="M1 pred")
        ax.plot(bin_r, gtn[rr], lw=1.6, ls="--", color="#D55E00",
                label="GT maxnorm")
        ax.set_title(n.replace("_", " ")[:34], fontsize=9)
        ax.set_xlim(0, 95)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[len(reps):]:
        ax.axis("off")
    fig.suptitle("M1 VAE reconstructed vs GT max-norm profile, representative ray",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig_repr_rays.png"), dpi=150)
    plt.close(fig)
    print("saved -> %s/" % OUT)


if __name__ == "__main__":
    main()
