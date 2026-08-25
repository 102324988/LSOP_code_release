"""Decisive convention check: qvec in images.txt vs loaded Camera.R vs R_gl@flip."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from scene.colmap_loader import qvec2rotmat


def read_qvec_images_txt(path):
    """Return {id: (qvec, tvec)} from COLMAP images.txt."""
    out = {}
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        toks = lines[i].split()
        qvec = np.array([float(x) for x in toks[1:5]])
        tvec = np.array([float(x) for x in toks[5:8]])
        out[int(toks[0])] = (qvec, tvec)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--view_id", type=int, default=1)
    args = ap.parse_args()

    with open(os.path.join(args.scene, "poses.json")) as f:
        data = json.load(f)
    views = {v["id"]: v for v in data["views"]}
    v = views[args.view_id]
    R_gl = np.array(v["R_gl"])
    t_gl = np.array(v["t"])
    flip = np.diag([1.0, -1.0, -1.0])
    R_gl_flip = R_gl @ flip

    qv = read_qvec_images_txt(os.path.join(args.scene, "sparse", "0", "images.txt"))
    qvec, tvec = qv[args.view_id]
    R_from_q = qvec2rotmat(qvec)
    R_from_q_T = R_from_q.T

    print(f"view {args.view_id}:")
    print(f"  R_gl_flip  = R_gl@diag(1,-1,-1):\n{np.round(R_gl_flip,4)}")
    print(f"  qvec2rotmat(qvec):\n{np.round(R_from_q,4)}")
    print(f"  qvec2rotmat(qvec).T:\n{np.round(R_from_q_T,4)}")
    print(f"  qvec2rotmat(qvec) == R_gl_flip?            {np.allclose(R_from_q, R_gl_flip, atol=1e-5)}")
    print(f"  qvec2rotmat(qvec).T == R_gl_flip?          {np.allclose(R_from_q_T, R_gl_flip, atol=1e-5)}")
    print(f"  tvec == t_gl?                              {np.allclose(tvec, t_gl, atol=1e-4)}")


if __name__ == "__main__":
    main()
