#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D5 dataset pack: build normalized per-object assets for spherical-profile
prior training. For each object:
  surf_points.npy  - depth_peak surface point cloud (coverage>=0.02 filter)
  meta.json        - merged D4 profile meta + D2 normalization + D3 PSNR
  coverage/depth   - copied profile summaries
Plus dataset manifest CSV and stats.
Inputs: output/gso_d4/*/profiles/, /root/gso/renders/*/meta.json,
        d3_full_summary.log
Output: output/gso_d5/<name>/{surf_points.npy,meta.json} , d5_manifest.csv
"""
import csv
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
D5 = os.path.join(ROOT, "output", "gso_d5")
RENDERS = "/root/gso/renders"
D3LOG = os.path.join(ROOT, "d3_full_summary.log")
N_THETA, N_PHI = 64, 128
COV_FLOOR = 0.02

_dirs = None


def get_dirs():
    global _dirs
    if _dirs is None:
        th = np.linspace(1e-3, np.pi - 1e-3, N_THETA)
        ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
        PH, TH = np.meshgrid(ph, th, indexing="ij")
        _dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
                         axis=-1).reshape(-1, 3).astype(np.float32)
    return _dirs


def parse_psnr():
    psnr = {}
    if os.path.exists(D3LOG):
        for line in open(D3LOG):
            m = re.match(r"=== (?:SKIP )?(\S+) PSNR ([\d.]+)", line.strip())
            if m:
                psnr[m.group(1)] = float(m.group(2))
    return psnr


def pack_one(name, psnr):
    prof = os.path.join(D4, name, "profiles")
    if not os.path.isdir(prof):
        return None
    try:
        peak = np.load(os.path.join(prof, "depth_peak.npy")).astype(np.float32)
        cov = np.load(os.path.join(prof, "coverage.npy")).astype(np.float32)
        dmean = np.load(os.path.join(prof, "depth_mean.npy")).astype(np.float32)
        pmeta = json.load(open(os.path.join(prof, "meta.json")))
        rmax = float(pmeta["rmax"])
        dirs = get_dirs()
        r = peak * rmax
        keep = (cov >= COV_FLOOR) & (r >= 0.02)
        surf = (r[keep, None] * dirs[keep]).astype(np.float32)
        rmeta = json.load(open(os.path.join(RENDERS, name, "meta.json")))
        out_dir = os.path.join(D5, name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "surf_points.npy"), surf)
        meta = {
            "name": name,
            "rmax": rmax,
            "dr": pmeta.get("dr"),
            "n_rays": int(peak.size),
            "n_keep": int(keep.sum()),
            "n_surf_pts": int(surf.shape[0]),
            "coverage_mean": float(cov.mean()),
            "depth_peak_mean": float(peak.mean()),
            "depth_peak_med": float(np.median(peak)),
            "psnr": psnr.get(name),
            "scale": rmeta["scale"],
            "bbox_center": rmeta["bbox_center"],
            "grid_bbox": pmeta["grid"].get("bbox"),
            "grid_res": pmeta["grid"].get("res"),
            "views_used": pmeta["grid"].get("views_used"),
            "render_dir": os.path.join(RENDERS, name),
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        return meta
    except Exception as e:  # noqa: BLE001
        return {"name": name, "error": str(e)}


def main():
    os.makedirs(D5, exist_ok=True)
    psnr = parse_psnr()
    names = sorted(os.listdir(D4))
    rows = []
    from functools import partial
    worker = partial(pack_one, psnr=psnr)
    with Pool(8) as p:
        for r in p.imap_unordered(worker, names, chunksize=4):
            if r is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["name"])
    cols = ["name", "n_keep", "n_surf_pts", "coverage_mean", "depth_peak_med",
            "rmax", "psnr", "scale", "bbox_center", "render_dir"]
    with open(os.path.join(ROOT, "d5_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["bbox_center"] = str(r2.get("bbox_center"))
            w.writerow(r2)
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    n_pts = [r["n_surf_pts"] for r in ok]
    print("packed=%d errors=%d" % (len(ok), len(errs)))
    print("n_surf_pts: min=%d med=%d max=%d" % (min(n_pts), int(np.median(n_pts)), max(n_pts)))
    print("coverage_mean med=%.4f" % np.median([r["coverage_mean"] for r in ok]))
    if errs:
        print("errors:", [r["name"] for r in errs])
    print("manifest -> d5_manifest.csv ; assets -> output/gso_d5/")


if __name__ == "__main__":
    sys.exit(main())
