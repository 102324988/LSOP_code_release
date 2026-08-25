"""E0 dry-run: write COLMAP text-format sparse model from poses.json.

COLMAP images.txt stores camera-to-world: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME.
R_colmap = R_gl @ diag(1,-1,-1) (see render_turntable.py for the convention).
"""
import argparse
import json
import os

import numpy as np


def rot2quat(R):
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.scene, "poses.json")) as f:
        data = json.load(f)
    meta, views = data["meta"], data["views"]

    sparse = os.path.join(args.scene, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)

    W, H = meta["width"], meta["height"]
    fx, fy, cx, cy = meta["fx"], meta["fy"], meta["cx"], meta["cy"]
    with open(os.path.join(sparse, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    flip = np.diag([1.0, -1.0, -1.0])
    lines = []
    for v in views:
        R = np.asarray(v["R_gl"], dtype=float) @ flip
        t = np.asarray(v["t"], dtype=float)
        q = rot2quat(R)
        lines.append(
            f"{v['id']} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
            f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {v['name']}.png"
        )
        lines.append("")
    with open(os.path.join(sparse, "images.txt"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(sparse, "points3D.txt"), "w"):
        pass
    print(f"wrote COLMAP text sparse for {len(views)} images -> {sparse}")


if __name__ == "__main__":
    main()
