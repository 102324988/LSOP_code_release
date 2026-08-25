"""E0/D3: collect mid-batch training results on the server.

Reads d3_mid_summary.log (per-object PSNR), the final PLY (gaussian count),
and each render's meta.json (diag_frac, watertight). Prints a summary table
plus failure/stat diagnostics for the 100-object mid-batch run.
Run on server: python collect_mid.py mid_batch_names.txt d3_mid_summary.log
"""
import json
import os
import re
import sys

import numpy as np
from plyfile import PlyData

ROOT = "/root/gso/renders"
OUT = "/root/e0lab/e0/output/gso_pilot"


def read_summary(path):
    """name -> psnr (float|None). '=== name PSNR x' or '=== SKIP name psnr'"""
    out = {}
    for line in open(path):
        m = re.match(r"=== (?:SKIP )?(\S+) PSNR ([\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def ply_stats(name):
    ply = os.path.join(OUT, name, "point_cloud", "iteration_6000", "point_cloud.ply")
    if not os.path.isfile(ply):
        return None, None
    d = PlyData.read(ply)
    v = d["vertex"]
    xyz = np.column_stack([v["x"], v["y"], v["z"]])
    r = float(np.linalg.norm(xyz - xyz.mean(0), axis=1).max())
    return len(v.data), r


def main():
    names = [x.strip() for x in open(sys.argv[1]) if x.strip()]
    psnr = read_summary(sys.argv[2])

    rows = []
    missing_meta = []
    for n in names:
        try:
            m = json.load(open(os.path.join(ROOT, n, "meta.json")))
            frac, wt = m["diag_frac"], m["watertight"]
        except Exception as e:
            missing_meta.append((n, str(e)))
            frac, wt = None, None
        ng, r = ply_stats(n)
        rows.append((n, psnr.get(n), ng, r, frac, wt))

    done = [r for r in rows if r[1] is not None]
    failed = [r for r in rows if r[1] is None]
    ok = [r for r in done if r[1] > 0]
    print(f"== MID-BATCH SUMMARY ==")
    print(f"total={len(rows)} done={len(done)} FAILED={len(failed)}")

    p = np.array([r[1] for r in ok]) if ok else np.array([])
    if len(p):
        print(f"PSNR  min={p.min():.2f} med={np.median(p):.2f} max={p.max():.2f} "
              f"mean={p.mean():.2f}")
        print(f"  q1={np.percentile(p,25):.2f} q3={np.percentile(p,75):.2f} "
              f"<25dB: {(p<25).sum()} | <28dB: {(p<28).sum()}")

    nwt_ok = [r for r in ok if r[5] is not None and not r[5]]
    nwt_p = np.array([r[1] for r in nwt_ok]) if nwt_ok else np.array([])
    nwt_total = sum(1 for r in rows if r[5] is False)
    print(f"non-watertight: done={len(nwt_ok)}/{nwt_total} "
          f"PSNR[min med max]=[{'--' if not len(nwt_p) else f'{nwt_p.min():.2f} {np.median(nwt_p):.2f} {nwt_p.max():.2f}'}]")

    ng_arr = np.array([r[2] for r in ok if r[2]])
    if len(ng_arr):
        print(f"gaussians min={ng_arr.min()} med={int(np.median(ng_arr))} max={ng_arr.max()}")

    if failed:
        print("\n-- FAILED (no PSNR in summary) --")
        for r in failed:
            print(f"  {r[0]}: psnr={r[1]} ng={r[2]} frac={r[4]} wt={r[5]}")
    if missing_meta:
        print("\n-- missing meta --")
        for n, e in missing_meta:
            print(f"  {n}: {e}")

    print("\n== per-object table ==")
    print(f"{'name':44s} {'PSNR':>6s} {'gauss':>7s} {'r':>6s} {'frac':>6s} {'wt':>5s}")
    for n, ps, ng, r, frac, wt in sorted(rows, key=lambda x: -(x[1] or 0)):
        p_ = f"{ps:.2f}" if ps is not None else "--"
        g_ = str(ng) if ng is not None else "--"
        r_ = f"{r:.2f}" if r is not None else "--"
        f_ = f"{frac:.3f}" if frac is not None else "--"
        w_ = str(wt) if wt is not None else "--"
        print(f"{n:44s} {p_:>6s} {g_:>7s} {r_:>6s} {f_:>6s} {w_:>5s}")


if __name__ == "__main__":
    main()
