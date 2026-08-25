import json, sys
import numpy as np
from plyfile import PlyData
ply = sys.argv[1]; out = sys.argv[2]
v = PlyData.read(ply)["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1)
r = np.linalg.norm(xyz, axis=1)
op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
w = op.astype(np.float64); w = w / w.sum()
samp = np.random.default_rng(0).choice(len(r), 300000, replace=True, p=w)
bins = np.linspace(0.0, 1.25, 51)
h, be = np.histogram(r[samp], bins=bins)
json.dump({"edges": be.tolist(), "hist": h.tolist(), "n": int(len(r)),
           "r_p50": float(np.percentile(r[samp], 50)),
           "r_p90": float(np.percentile(r[samp], 90))}, open(out, "w"))
print(out, len(r))
