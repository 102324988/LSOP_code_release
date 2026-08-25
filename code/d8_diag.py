#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D8 profile-shape diagnostics on representative val objects.

Compares D8 predicted per-ray profiles against GT max-norm profiles to answer,
for the M1 (profile VAE) design:
  1. Where along the radial bin axis is the systematic |pred - gt| error?
  2. How far off is the predicted peak bin from the GT peak bin?
  3. Are predicted profiles systematically wider / narrower (FWHM ratio)?
  4. Representative-ray curves (pred vs GT) for inspection.

Objects: 5 representatives spanning D8 behavior — Asus (mid), Perricone_Hypo
(most improved), Top_Paw (improved, hardest), Nordic (regressed), My_Little_Pony
(regressed).

Outputs: d8_diag/figs (PNG) + d8_diag/summary.json
Usage: python d8_diag.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
PRED = os.path.join(ROOT, "d8_mean_pool_val")
OUT = os.path.join(ROOT, "d8_diag")
N_BINS = 96
PREFIXES = ["Asus_Sabertooth_Z97", "Perricone_MD_Hypo", "Top_Paw_Dog_Bowl",
            "Nordic_Ware_Original", "My_Little_Pony_Princess"]


def resolve(prefix):
    names = sorted({f[:-4] for f in os.listdir(PRED) if f.endswith(".npy")
                    and not any(f.endswith(s) for s in
                                ("_cov.npy", "_soft.npy", "_prof.npy",
                                 "_predpeak.npy"))})
    hits = [n for n in names if n.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def load(n):
    pred = np.load(os.path.join(PRED, n + "_prof.npy")).astype(np.float32)
    cov = np.load(os.path.join(PRED, n + "_cov.npy")).astype(np.float32)
    gt = np.load(os.path.join(D4, n, "profiles", "profiles.npy")).astype(np.float32)
    gt = gt.reshape(-1, N_BINS)
    mx = gt.max(-1, keepdims=True)
    gtn = gt / np.maximum(mx, 1e-6)
    peak_gt = np.load(os.path.join(D4, n, "profiles", "depth_peak.npy")).astype(np.float32)
    rmax = float(json.load(open(os.path.join(D4, n, "profiles", "meta.json")))["rmax"])
    return pred, gtn, cov, peak_gt, rmax


def fwhm(prof):
    """FWHM in bins; profiles are sigmoid-ish (peak in middle)."""
    # half max crossing around the peak
    pk = prof.argmax(-1)
    hm = prof.max(-1) * 0.5
    w = np.zeros(len(prof), dtype=np.float32)
    for i in range(len(prof)):
        p = prof[i]
        # left crossing: last index below hm before peak, right crossing after
        lo, hi = pk[i], pk[i]
        while lo > 0 and p[lo - 1] >= hm[i]:
            lo -= 1
        while hi < N_BINS - 1 and p[hi + 1] >= hm[i]:
            hi += 1
        # interpolate edges
        w[i] = hi - lo
    return w


def representative_ray(soft_err, m):
    """Pick the ray whose |soft_err - median| is smallest (typical behavior)."""
    med = np.median(soft_err)
    return int(np.abs(soft_err - med).argmin())


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    names = [resolve(p) for p in PREFIXES]
    # ---------- fig 1: representative-ray curves ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    bin_r = np.arange(N_BINS)
    for ax, n in zip(axes[:5], names):
        pred, gtn, cov, peak_gt, rmax = load(n)
        m = cov >= 0.02
        soft = (pred * bin_r[None, :]).sum(-1) / (pred.sum(-1) + 1e-6)
        err = np.abs(soft / (N_BINS - 1) - (peak_gt / rmax))
        r = representative_ray(err[m], np.median(err[m]))
        rr = np.flatnonzero(m)[r]
        ax.plot(bin_r, pred[rr], lw=1.6, color="#0072B2", label="D8 pred")
        ax.plot(bin_r, gtn[rr], lw=1.6, ls="--", color="#D55E00", label="GT maxnorm")
        ax.set_title(n.replace("_", " ")[:34], fontsize=9)
        ax.set_xlim(0, 95)
        ax.legend(fontsize=7, loc="upper right")
    axes[5].axis("off")
    fig.suptitle("D8 predicted vs GT max-norm profile, representative ray",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig1_rays.png"), dpi=150)
    plt.close(fig)

    # ---------- fig 2: aggregate shape-error statistics ----------
    agg = {}
    all_absbin = {}   # per object: mean |pred-gtn| vs bin, cov>=0.02 rays
    all_peakoff = {}  # peak-bin offset (argmax pred - argmax gtn)
    all_fwhm = {}     # FWHM pred / FWHM gt
    for n in names:
        pred, gtn, cov, peak_gt, rmax = load(n)
        m = cov >= 0.02
        p, g = pred[m], gtn[m]
        absbin = np.abs(p - g).mean(0)              # (96,)
        peak_p = p.argmax(-1)
        peak_g = g.argmax(-1)
        off = peak_p - peak_g
        # FWHM ratio (guard: gtn peak can be bin0 for near-degenerate rays)
        fw_p = fwhm(p)
        fw_g = fwhm(g)
        good = (fw_g >= 2)
        ratio = fw_p[good] / np.maximum(fw_g[good], 1e-6)
        all_absbin[n] = absbin
        all_peakoff[n] = off
        all_fwhm[n] = ratio
        agg[n] = {
            "mean_absbin": float(absbin.mean()),
            "absbin_max_bin": int(absbin.argmax()),
            "absbin_max_val": float(absbin.max()),
            "peakoff_med": float(np.median(off)),
            "peakoff_iqr": float(np.percentile(off, 75) - np.percentile(off, 25)),
            "fwhm_ratio_med": float(np.median(ratio)),
            "fwhm_ratio_gt25": float(np.percentile(ratio, 25)),
            "fwhm_ratio_gt75": float(np.percentile(ratio, 75)),
            "cov_frac": float(m.mean()),
        }

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    short = {n: n.split("_")[0] + "_" + n.split("_")[1] for n in names}
    for n in names:
        axs[0].plot(bin_r, all_absbin[n], lw=1.4, label=short[n])
    axs[0].set_xlabel("radial bin")
    axs[0].set_ylabel("mean |pred - GT|")
    axs[0].set_title("per-bin profile error (cov>=0.02)")
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=0.3)

    for n in names:
        axs[1].hist(all_peakoff[n], bins=np.arange(-30, 31), alpha=0.5,
                    label=short[n], density=True)
    axs[1].set_xlabel("peak-bin offset (pred - GT)")
    axs[1].set_ylabel("density")
    axs[1].set_title("predicted peak offset")
    axs[1].legend(fontsize=7)

    bp = axs[2].boxplot([all_fwhm[n] for n in names], labels=[short[n] for n in names],
                        patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], ["#0072B2"] * len(names)):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    axs[2].axhline(1.0, color="gray", ls="--", lw=1)
    axs[2].set_ylabel("FWHM pred / GT")
    axs[2].set_title("profile width ratio")
    axs[2].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig2_shape_stats.png"), dpi=150)
    plt.close(fig)

    json.dump(agg, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("saved to %s/ (figs + summary.json)" % OUT)
    for n in names:
        a = agg[n]
        print("%-38s cov=%.2f per-bin|d|=%.4f max@bin%d=%.4f "
              "peakoff med=%+.1f IQR=%.1f fwhm_ratio med=%.2f [%.2f-%.2f]"
              % (n, a["cov_frac"], a["mean_absbin"], a["absbin_max_bin"],
                 a["absbin_max_val"], a["peakoff_med"], a["peakoff_iqr"],
                 a["fwhm_ratio_med"], a["fwhm_ratio_gt25"],
                 a["fwhm_ratio_gt75"]))


if __name__ == "__main__":
    main()
