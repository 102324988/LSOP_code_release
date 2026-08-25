"""E0: quick density-shell check for a PGSR-trained sphere.

Read a gaussian-splatting point_cloud.ply, report where the opacity-weighted
Gaussian density lives (radius distribution vs GT shell at r=0.95). This is the
core question: does PGSR's training concentrate density onto the shell, fixing
the 3DGS sphere degeneration (density smeared inward, p50 radius 0.72)?
"""
import json
import sys

import numpy as np
from plyfile import PlyData

ply_path = sys.argv[1]
v = PlyData.read(ply_path)["vertex"]
names = [p.name for p in v.properties]
xyz = np.stack([v["x"], v["y"], v["z"]], 1)
r = np.linalg.norm(xyz, axis=1)

if "opacity" in names:
    opv = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
else:
    opv = np.ones(len(xyz))
if "scale_0" in names:
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)
    scv = np.exp(np.asarray(sc))
    smax = scv.max(1)
else:
    smax = np.ones(len(xyz))

# opacity-weighted radius distribution (resample 200k by weight)
w = opv.astype(np.float64)
w = w / w.sum()
samp = np.random.default_rng(0).choice(len(r), size=200000, replace=True, p=w)

res = {
    "fields": names,
    "n": int(len(xyz)),
    "radius_p10": float(np.percentile(r, 10)),
    "radius_p50": float(np.percentile(r, 50)),
    "radius_p90": float(np.percentile(r, 90)),
    "frac_r_lt_07": float(np.mean(r < 0.7)),
    "frac_r_gt_09": float(np.mean(r > 0.9)),
    "op_median": float(np.median(opv)),
    "op_gt_05": float(np.mean(opv > 0.5)),
    # where the visible density actually lives:
    "opw_radius_p10": float(np.percentile(r[samp], 10)),
    "opw_radius_p50": float(np.percentile(r[samp], 50)),
    "opw_radius_p90": float(np.percentile(r[samp], 90)),
    "opw_frac_lt_07": float(np.mean(r[samp] < 0.7)),
    "opw_frac_gt_09": float(np.mean(r[samp] > 0.9)),
    "scale_p50": float(np.percentile(scv[:, 0], 50)),
    "scale_p90": float(np.percentile(scv[:, 0], 90)),
    "smax_p50": float(np.percentile(smax, 50)),
    "smax_p90": float(np.percentile(smax, 90)),
}
print(json.dumps(res, indent=1, ensure_ascii=False))
