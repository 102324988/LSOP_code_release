"""E0: quick scale/opacity diagnostics straight from the saved point cloud ply."""
import argparse
import numpy as np
from plyfile import PlyData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    args = ap.parse_args()

    pd = PlyData.read(args.ply)
    d = {v.name: np.asarray(v.data) for v in pd.elements}
    verts = d["vertex"]
    v = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
    scl = np.exp(np.stack([verts["scale_0"], verts["scale_1"], verts["scale_2"]], axis=1).astype(np.float32))
    op = 1.0 / (1.0 + np.exp(-verts["opacity"].astype(np.float32)[:, None]))

    print(f"N={len(v)}")
    print(f"scale: min={scl.min():.3f} p10={np.quantile(scl,0.1):.3f} "
          f"p50={np.quantile(scl,0.5):.3f} p90={np.quantile(scl,0.9):.3f} max={scl.max():.2f}")
    print(f"scale p50 per-axis: {[round(float(np.quantile(scl[:,i],0.5)),3) for i in range(3)]}")
    print(f"opacity: p10={np.quantile(op,0.1):.3f} p50={np.quantile(op,0.5):.3f} "
          f"p90={np.quantile(op,0.9):.3f} max={op.max():.3f}")
    print(f"frac max-scale >0.3: {(scl.max(1)>0.3).mean()*100:.1f}%")
    print(f"frac max-scale <0.15: {(scl.max(1)<0.15).mean()*100:.1f}%")
    print(f"frac max-scale <0.05: {(scl.max(1)<0.05).mean()*100:.1f}%")

    r = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    d = np.abs(np.sqrt((r - 0.55) ** 2 + v[:, 2] ** 2) - 0.23)
    for thr in [0.02, 0.05, 0.1, 0.2]:
        msk = d < thr
        print(f"  tube d<{thr}: {msk.mean()*100:5.1f}%  "
              f"scale p50 there={np.quantile(scl[msk],0.5):.3f}")


if __name__ == "__main__":
    main()
