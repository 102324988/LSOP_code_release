"""E0 dry-run: round-trip verification of our COLMAP poses through GOF's reader.

Loads the scene with GOF's readColmapSceneInfo, then reprojects known world
points (origin, object-top, points on the object's bounding sphere) with the
loaded cameras and checks they land where they should.

Convention facts (verified against GOF source):
  - images.txt stores camera-to-world quaternion (w,x,y,z) + camera center t
  - readColmapCameras: Camera.R = transpose(qvec2rotmat(qvec)) = world->cam rot,
    Camera.T = tvec = camera center in world
  - so a world point projects as pc = R @ (pt - T), u = K @ pc

Note: the axis-aligned bounding-box corners are deliberately NOT used — the
procedural objects' AABB (half-diagonal ~1.6) legitimately pokes out of the
frame; only the object's bounding sphere must be fully in view.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/e0lab/gaussian-opacity-fields"))
from scene.dataset_readers import readColmapSceneInfo  # noqa: E402


def sphere_points(radius, n_lat=9, n_lon=16):
    """Samples on the sphere surface (avoiding the poles)."""
    pts = []
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            pts.append(radius * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]))
    return np.array(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--obj_radius", type=float, default=0.95)
    ap.add_argument("--margin_px", type=float, default=15.0)
    args = ap.parse_args()

    scene_dir = os.path.abspath(args.scene)
    info = readColmapSceneInfo(scene_dir, "images", False)
    cams = info.train_cameras
    print(f"GOF reader loaded {len(cams)} train cameras")

    with open(os.path.join(scene_dir, "poses.json")) as f:
        meta = json.load(f)["meta"]
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1.0]])
    W, H = meta["width"], meta["height"]

    sph = sphere_points(args.obj_radius)
    top = np.array([0.0, 0.0, 0.6])
    origin = np.zeros(3)

    # GOF Camera.R = world->camera rotation, T = camera center
    def project(R, t, pt):
        pc = (R @ (pt - t).T).T
        u = K @ pc.T
        return u[:2] / u[2]

    max_err_center = 0.0
    sph_out = 0
    top_below_center = 0
    for c in cams:
        R = np.asarray(c.R, dtype=float)
        t = np.asarray(c.T, dtype=float)
        c0 = project(R, t, origin)
        max_err_center = max(max_err_center, float(np.linalg.norm(c0 - np.array([meta["cx"], meta["cy"]]))))
        sp = project(R, t, sph)
        in_frame = ((sp[0] >= args.margin_px) & (sp[0] <= W - args.margin_px)
                    & (sp[1] >= args.margin_px) & (sp[1] <= H - args.margin_px))
        sph_out += int(np.sum(~in_frame))
        tp = project(R, t, top)
        if tp[1] > meta["cy"]:
            top_below_center += 1

    total_sph = len(cams) * sph.shape[0]
    print(f"max origin reprojection error : {max_err_center:.3f} px")
    print(f"sphere points outside frame   : {sph_out}/{total_sph}")
    print(f"top-of-object below center    : {top_below_center}/{len(cams)}")
    ok = max_err_center < 5.0 and sph_out == 0 and top_below_center == 0
    print("ROUND-TRIP:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
