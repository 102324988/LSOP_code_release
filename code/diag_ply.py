import sys, numpy as np
from plyfile import PlyData

def stat(p):
    d = PlyData.read(p)
    v = d["vertex"]
    n = len(v.data)
    xyz = np.column_stack([v["x"], v["y"], v["z"]])
    sc = np.column_stack([np.exp(v["scale_0"]), np.exp(v["scale_1"]), np.exp(v["scale_2"])])
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
    c = xyz.mean(0)
    r = float(np.linalg.norm(xyz - c, axis=1).max())
    print(f"{p}")
    print(f"  n={n}  xyz_center=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  bsphere_r={r:.3f}")
    print(f"  scale: med={np.median(sc):.4f}  p90={np.percentile(sc,90):.4f}  max={sc.max():.4f}")
    print(f"  opacity: med={np.median(op):.4f}  frac<0.05={np.mean(op<0.05):.3f}  frac<0.5={np.mean(op<0.5):.3f}")
    print(f"  xyz range: [{xyz.min(0)}, {xyz.max(0)}]")

for p in sys.argv[1:]:
    stat(p)
