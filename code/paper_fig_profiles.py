#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper figure: real profile curves for representative objects.
Compares GT profile, D8 (global cond), M3b v1 (per-ray L1), M3b v2 gamma=2
(per-ray + sharpened target) on selected covered rays.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/root/e0lab/e0"
sys.path.insert(0, ROOT)
from m1_train_vae import load_object  # noqa: E402

OUT = os.path.join(ROOT, "paper_fig_profiles.png")
D8 = os.path.join(ROOT, "d8_mean_pool_val")
V1 = os.path.join(ROOT, "m3b_preray_val", "profiles")
V2 = os.path.join(ROOT, "m3b_preray_sg2_val", "profiles")

names = json.load(open(os.path.join(ROOT, "d8_mean_pool", "meta.json")))["val_names"]
summ = json.load(open(os.path.join(ROOT, "m3b_preray_val", "summary.json")))
per = {p["name"]: p["dmed_s"] for p in summ["per_object"]}

# pick a mid-difficulty and a hard object (largest dmed_s)
dmeds = sorted(per.values())
mid = sorted(names, key=lambda n: abs(per[n] - dmeds[len(dmeds) // 2]))[0]
hard = max(names, key=lambda n: per[n])
print("objects:", mid, per[mid], "|", hard, per[hard], flush=True)


def pick_rays(sh, cov):
    covered = np.where(cov >= 0.02)[0]
    peak = sh.max(-1)
    ok = covered[peak[covered] > 0.2]
    bins = sh.argmax(-1)[ok]
    q = np.percentile(bins, [25, 50, 75]).astype(int)
    sel = []
    for qi in q:
        cand = ok[np.abs(bins - qi).argmin()]
        sel.append(cand)
    return sel


cmap = {"GT": "#222222", "D8 (global)": "#1f77b4",
        "per-ray L1": "#ff7f0e", "per-ray +$\\gamma$=2": "#2ca02c"}
ls = {"GT": "-", "D8 (global)": "-", "per-ray L1": "--",
      "per-ray +$\\gamma$=2": "-"}

fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.0), sharex=True)
for row, name in enumerate([mid, hard]):
    sh, cov, _, _ = load_object(name)
    sh = sh.numpy()
    cov = cov.numpy()
    profs = {
        "GT": sh,
        "D8 (global)": np.load(os.path.join(D8, name + "_prof.npy")),
        "per-ray L1": np.load(os.path.join(V1, name + "_prof.npy")),
        "per-ray +$\\gamma$=2": np.load(os.path.join(V2, name + "_prof.npy")),
    }
    sel = pick_rays(sh, cov)
    for col, ri in enumerate(sel):
        ax = axes[row, col]
        for k, P in profs.items():
            ax.plot(P[ri], ls[k], color=cmap[k], lw=1.3, label=k)
        ax.set_title("ray #%d (peak bin %d)" % (ri, sh.argmax(-1)[ri]),
                     fontsize=8)
        ax.set_xticks([0, 24, 48, 72, 95])
        if row == 1:
            ax.set_xlabel("radial bin (96 bins, $r_{\\max}$=2.25)")
        if col == 0:
            ax.set_ylabel("occupancy $P(r)$")
    axes[row, 0].annotate("$\\mathrm{dmed_s}$=%.3f" % per[name], xy=(0.0, 1.02),
                          xycoords="axes fraction", fontsize=9, fontweight="bold")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("saved ->", OUT, flush=True)
