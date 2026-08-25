#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D4 full-batch aggregation: opacity grid -> spherical profile inversion stats.
Reads output/gso_d4/*/profiles/{metrics.json, meta.json, coverage.npy, depth_peak.npy}
and links D3 PSNR from d3_full_summary.log. Writes d4_summary.csv + d4_summary.json.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = "/root/e0lab/e0"
D4 = os.path.join(ROOT, "output", "gso_d4")
D3LOG = os.path.join(ROOT, "d3_full_summary.log")


def parse_d3():
    psnr = {}
    if os.path.exists(D3LOG):
        for line in open(D3LOG):
            m = re.match(r"=== (?:SKIP )?(\S+) PSNR ([\d.]+)", line.strip())
            if m:
                psnr[m.group(1)] = float(m.group(2))
    return psnr


def collect():
    psnr = parse_d3()
    rows = []
    degen, fail = [], []
    for name in sorted(os.listdir(D4)):
        prof = os.path.join(D4, name, "profiles")
        if not os.path.isdir(prof):
            continue
        meta_p = os.path.join(prof, "meta.json")
        met_p = os.path.join(prof, "metrics.json")
        row = {"name": name}
        if name in psnr:
            row["psnr"] = psnr[name]
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p))
            row["coverage"] = meta.get("coverage_mean", float("nan"))
            row["rmax"] = meta.get("rmax", float("nan"))
        cov = os.path.join(prof, "coverage.npy")
        dep = os.path.join(prof, "depth_peak.npy")
        if os.path.exists(cov):
            c = np.load(cov)
            row["coverage_min"] = float(c.min())
            row["coverage_max"] = float(c.max())
        if os.path.exists(dep):
            d = np.load(dep)
            row["depth_peak_mean"] = float(d.mean())
        if os.path.exists(met_p):
            m = json.load(open(met_p))
            row.update({"chamfer": m["chamfer"], "fscore": m["fscore"],
                        "f_a": m["f_a"], "f_b": m["f_b"]})
            rows.append(row)
        else:
            degen.append(row)  # degenerate inversion mesh
    return rows, degen


def stats(v):
    v = np.array([x for x in v if x == x], dtype=float)
    if v.size == 0:
        return None
    return dict(n=int(v.size), min=float(v.min()), q1=float(np.percentile(v, 25)),
                median=float(np.median(v)), q3=float(np.percentile(v, 75)),
                max=float(v.max()), mean=float(v.mean()))


def main():
    rows, degen = collect()
    if not rows:
        print("no metrics.json found yet; batch still running?")
        return 1
    keys = ["chamfer", "fscore", "f_a", "f_b", "coverage", "depth_peak_mean", "psnr"]
    summ = {"n_ok": len(rows), "n_degen": len(degen)}
    for k in keys:
        s = stats([r.get(k, float("nan")) for r in rows])
        if s:
            summ[k] = s
    # worst / best objects by fscore
    ok = [r for r in rows if r.get("fscore", -1) >= 0]
    ok.sort(key=lambda r: r["fscore"])
    summ["worst_fscore"] = [{"name": r["name"], "fscore": round(r["fscore"], 4),
                             "chamfer": round(r["chamfer"], 4),
                             "psnr": round(r.get("psnr", float("nan")), 2)} for r in ok[:10]]
    summ["best_fscore"] = [{"name": r["name"], "fscore": round(r["fscore"], 4),
                            "chamfer": round(r["chamfer"], 4)} for r in ok[-10:][::-1]]
    summ["degen_list"] = [r["name"] for r in degen]
    summ["degen_coverage"] = {r["name"]: round(r.get("coverage", float("nan")), 4) for r in degen}

    with open(os.path.join(ROOT, "d4_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)

    # CSV
    import csv
    fields = ["name", "psnr", "coverage", "chamfer", "fscore", "f_a", "f_b", "depth_peak_mean"]
    with open(os.path.join(ROOT, "d4_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(json.dumps({k: v for k, v in summ.items() if k not in ("degen_list",)}, indent=1, default=str))
    print("degen:", summ["degen_list"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
