"""E0 dry-run: turntable multi-view rendering with pyrender+EGL.

Writes PNGs + poses.json (pyrender/OpenGL camera-to-world R,t per view).
Camera conventions handled here:
  - pyrender : camera looks along local -Z, image y down (top-left origin)
  - COLMAP   : camera looks along local +Z toward scene, image y down
  Conversion for COLMAP is R_colmap = R_gl @ diag(1,-1,-1), same t
  (verified numerically in verify_colmap.py by projecting known points).
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import pyrender
import trimesh


def lookat_gl(eye, center, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation for a pyrender camera looking at `center`.

    Camera looks along local -Z (toward `center`), local +Y up.
    Returns R (3x3, columns = camera axes in world) and eye.
    """
    eye = np.asarray(eye, dtype=float)
    center = np.asarray(center, dtype=float)
    up = np.asarray(up, dtype=float)
    z = eye - center
    z = z / np.linalg.norm(z)
    y = up - np.dot(up, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    return R, eye


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_az", type=int, default=16)
    ap.add_argument("--elevations", type=str, default="20,45,70")
    ap.add_argument("--radius", type=float, default=3.2)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--yfov", type=float, default=40.0, help="vertical fov, degrees")
    ap.add_argument("--bg", type=str, default="0,0,0",
                    help="background RGB, e.g. 0.95,0.95,0.95 for a light background")
    ap.add_argument("--bg_mode", type=str, default="fixed", choices=["fixed", "random"],
                    help="fixed: one bg for all views; random: per-view random saturated color")
    args = ap.parse_args()

    bg = np.array([float(v) for v in args.bg.split(",")], dtype=np.float32)
    elevs = [float(e) for e in args.elevations.split(",")]
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    mesh = trimesh.load(args.mesh, force="mesh")
    pm = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    scene = pyrender.Scene(ambient_light=np.array([0.40, 0.40, 0.40]))
    scene.bg_color = bg  # composited in GOF with matching white_background flag
    scene.add(pm)

    for pos, intensity in [([2.0, 1.5, 3.0], 3.0), ([-2.0, -1.0, 1.5], 2.0)]:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        R, _ = lookat_gl(np.asarray(pos, float), np.zeros(3))
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = pos
        scene.add(light, pose=pose)

    yfov = np.deg2rad(args.yfov)
    aspect = args.width / args.height
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect)
    cam_node = scene.add(cam, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(args.width, args.height)

    fy = (args.height / 2.0) / np.tan(yfov / 2.0)
    fx = fy * aspect
    cx, cy = args.width / 2.0, args.height / 2.0

    views = []
    vid = 0
    rng = np.random.default_rng(0)
    for ei, el in enumerate(elevs):
        el_rad = np.deg2rad(el)
        for ai in range(args.n_az):
            az = ai / args.n_az * 2 * np.pi
            eye = np.array([
                args.radius * np.cos(el_rad) * np.cos(az),
                args.radius * np.cos(el_rad) * np.sin(az),
                args.radius * np.sin(el_rad),
            ])
            R, t = lookat_gl(eye, np.zeros(3))
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            if args.bg_mode == "random":
                # saturated, mid-to-high-value random color (H,S,V) so the
                # object + phantom gaussians can't hide in a uniform bg
                h = rng.uniform(0, 1)
                s = rng.uniform(0.5, 0.95)
                v = rng.uniform(0.4, 0.9)
                import colorsys
                r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
                bg = np.array([r_, g_, b_], dtype=np.float32)
                scene.bg_color = bg
            scene.set_pose(cam_node, pose)
            color, _ = renderer.render(scene)
            name = f"view_{vid:04d}"
            cv2.imwrite(os.path.join(img_dir, name + ".png"), color[:, :, ::-1])
            views.append({
                "name": name, "id": vid + 1,
                "R_gl": R.tolist(), "t": t.tolist(),
                "az": az, "el": el_rad,
                "bg": bg.tolist(),
            })
            vid += 1

    # numeric sanity diagnostics (can't view images on this host)
    last = cv2.imread(os.path.join(img_dir, f"view_{vid-1:04d}.png")).astype(np.float32) / 255.0
    fg = last[last.max(axis=-1) > 0.04]  # object = anything visibly brighter than black bg
    frac = fg.shape[0] / (last.shape[0] * last.shape[1])
    lum = 0.299 * fg[..., 0] + 0.587 * fg[..., 1] + 0.114 * fg[..., 2]
    print(f"[diag] last view: object pixel fraction={frac:.3f} "
          f"fg mean lum={lum.mean():.3f} std lum={lum.std():.3f} (std>0.03 implies texture/shading)")

    renderer.delete()
    meta = {"width": args.width, "height": args.height, "yfov_deg": args.yfov,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "n_views": vid}
    with open(os.path.join(args.out, "poses.json"), "w") as f:
        json.dump({"meta": meta, "views": views}, f, indent=1)
    # per-view background lookup for the trainer (image_name -> [r,g,b])
    with open(os.path.join(args.out, "bg.json"), "w") as f:
        json.dump({v["name"] + ".png": v["bg"] for v in views}, f, indent=1)
    print(f"rendered {vid} views -> {args.out}  (bg_mode={args.bg_mode})")


if __name__ == "__main__":
    main()
