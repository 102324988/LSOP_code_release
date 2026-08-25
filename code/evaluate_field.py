"""Offline evaluation of candidate opacity-field formulas on Av (point x view)
alpha_integrated matrix collected by probe_field_formulas.py.

alpha_integrated = accumulated front-to-back opacity A_v(x) along ray from view v
up to (clamped) depth of x.  Candidates:
  cur    alpha = 1 - min_v A_v      (official GOF formula)
  maxA   alpha = max_v A_v          (paper field 1 - min T == max(1-T))
  minA   alpha = min_v A_v
  oneMax alpha = 1 - max_v A_v
Scoring: known surface points should be HIGH, known empty points LOW.
GT torus: major 0.67 minor 0.28, in XY plane -> tube bbox [-0.95,0.95]^2 x [-0.28,0.28].
"""
import argparse
import json
import os

import numpy as np


def idx_for(coord, bbox, res):
    x, y, z = coord
    lo = np.array([bbox[0], bbox[2], bbox[4]])
    hi = np.array([bbox[1], bbox[3], bbox[5]])
    ii = np.round((np.array([x, y, z]) - lo) / (hi - lo) * (res - 1)).astype(int)
    i, j, k = ii
    return int(k * res * res + j * res + i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    Av = np.load(os.path.join(args.dir, "Av.npy")).astype(np.float32)
    with open(os.path.join(args.dir, "meta.json")) as f:
        meta = json.load(f)
    res = meta["res"]
    bbox = meta["bbox"]
    n, nv = Av.shape
    print(f"Av {n}x{nv}, res={res}, bbox={bbox}")

    fields = {
        "cur(1-min)": 1.0 - Av.min(1),
        "maxA": Av.max(1),
        "minA": Av.min(1),
        "1-maxA": 1.0 - Av.max(1),
        "meanA": Av.mean(1),
    }

    # GT torus: major R=0.55, minor r=0.23, tube around z-axis in XY plane
    probes = [
        ("outer_equator(0.78,0,0)", (0.78, 0.0, 0.0), 1),
        ("inner_equator(0.32,0,0)", (0.32, 0.0, 0.0), 1),
        ("top_of_tube(0.55,0,0.23)", (0.55, 0.0, 0.23), 1),
        ("tube_center(0.55,0,0)", (0.55, 0.0, 0.0), 1),
        ("hole_center(0,0,0)", (0.0, 0.0, 0.0), 0),   # GOF convex-hull limitation
        ("above(0,0,0.6)", (0.0, 0.0, 0.6), 0),
        ("far_empty(0,1.2,0)", (0.0, 1.2, 0.0), 0),
        ("beyond(0.9,0,0)", (0.9, 0.0, 0.0), 0),
    ]

    print(f"\n{'formula':>12s} | " + " | ".join(f"{name[:16]:>16s}" for name, _, _ in probes))
    for fname, alpha in fields.items():
        row = f"{fname:>12s} | "
        for name, coord, exp in probes:
            v = alpha[idx_for(coord, bbox, res)]
            mark = " *" if (exp == 1 and v < 0.5) or (exp == 0 and v > 0.5) else "  "
            row += f"{v:5.2f}{mark:>11s} | "
        print(row)

    gt = np.array([[-0.78, -0.78, -0.23], [0.78, 0.78, 0.23]])
    lo0 = np.array([bbox[0], bbox[2], bbox[4]])
    hi0 = np.array([bbox[1], bbox[3], bbox[5]])
    print(f"\nGT tube bbox: {gt.tolist()}")
    for fname, alpha in fields.items():
        inside = alpha > 0.5
        if inside.sum() == 0:
            print(f"  {fname:>12s}: NO voxels >0.5")
            continue
        iis = np.argwhere(inside)  # (M,3) in voxel indices
        lo = lo0 + iis.min(0) / (res - 1) * (hi0 - lo0)
        hi = lo0 + (iis.max(0) + 1) / (res - 1) * (hi0 - lo0)
        cnt = inside.sum()
        frac = cnt / n
        cent = (lo + hi) / 2
        print(f"  {fname:>12s}: voxels={cnt:6d} ({frac:5.1%}) "
              f"bbox[{lo[0]:+.2f},{hi[0]:+.2f}]x[{lo[1]:+.2f},{hi[1]:+.2f}]x[{lo[2]:+.2f},{hi[2]:+.2f}] "
              f"centroid=({cent[0]:+.3f},{cent[1]:+.3f},{cent[2]:+.3f})")


if __name__ == "__main__":
    main()
