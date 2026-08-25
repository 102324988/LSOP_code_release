#!/usr/bin/env python3
"""D2: GSO batch multi-view rendering (pyrender + EGL, GPU/CPU on server).

Per-object output (format matches e0/render_turntable.py, so D3/D4 consumers
work unchanged):
    <out>/<name>/
        images/view_XXXX.png
        poses.json   {"meta": {w,h,yfov_deg,fx,fy,cx,cy,n_views},
                      "views": [{name,id,R_gl,t,az,el,bg}]}
        bg.json      {view_XXXX.png: [r,g,b]}          (per-view bg for trainer)
        meta.json    {name, scale, bbox_center, n_verts, n_faces, watertight,
                      textured, render_ok, diag_frac, diag_lum_std, t_sec}

Normalization (per object): translate bbox center -> origin, then scale so the
bounding-sphere radius == 1.0. Camera radius stays 3.2 with yfov=40 (object
angular radius ~17deg < 20deg half-fov; object height ~85% of image height).

Resume: an object is skipped if poses.json exists, n_views == expected and
images/view_%04d.png all exist. Pass --force to re-render.

Workers: multiprocessing Pool; each worker lazily creates its own
OffscreenRenderer (parent never renders, so fork + EGL is safe).
"""
import argparse
import colorsys
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import cv2
import pyrender
import trimesh


def lookat_gl(eye, center, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation for a pyrender camera looking at `center`."""
    eye = np.asarray(eye, float)
    center = np.asarray(center, float)
    up = np.asarray(up, float)
    z = eye - center
    z = z / np.linalg.norm(z)
    y = up - np.dot(up, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    return R, eye


_worker_renderer = None  # per-process OffscreenRenderer
_GLOBAL_CFG = None  # config shared with pool workers (fork-inherited)


def _pool_render(task):
    return render_one(task, _GLOBAL_CFG)


def _worker_init(width, height):
    global _worker_renderer
    _worker_renderer = pyrender.OffscreenRenderer(width, height)


def render_one(task, cfg):
    """Render a single object. task = {name, glb}. Returns stats dict."""
    name = task["name"]
    glb = task["glb"]
    out_dir = os.path.join(cfg["out_root"], name)
    img_dir = os.path.join(out_dir, "images")
    t0 = time.time()
    try:
        return _render_impl(name, glb, out_dir, img_dir, cfg, t0)
    except Exception:
        print(f"[FAIL] {name}\n{traceback.format_exc()}", flush=True)
        return None


def _render_impl(name, glb, out_dir, img_dir, cfg, t0):
    m = trimesh.load(glb, force="mesh")

    # ---- center + normalize (bounding-sphere radius -> 1.0) ----
    center = m.bounds.mean(0)
    m.vertices -= center
    sph_r = float(np.linalg.norm(m.vertices, axis=1).max())
    scale = 1.0 / max(sph_r, 1e-9)
    if abs(scale - 1.0) > 1e-6:
        m.apply_scale(scale)

    pm = pyrender.Mesh.from_trimesh(m, smooth=False)
    scene = pyrender.Scene(ambient_light=np.array([0.40, 0.40, 0.40]))
    bg = np.array([float(v) for v in cfg["bg_fixed"].split(",")], dtype=np.float32)
    scene.bg_color = bg
    scene.add(pm)
    for pos, intensity in [([2.0, 1.5, 3.0], 3.0), ([-2.0, -1.0, 1.5], 2.0)]:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        R, _ = lookat_gl(np.asarray(pos, float), np.zeros(3))
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = pos
        scene.add(light, pose=pose)

    yfov = np.deg2rad(cfg["yfov_deg"])
    aspect = cfg["width"] / cfg["height"]
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect)
    cam_node = scene.add(cam, pose=np.eye(4))
    renderer = _worker_renderer

    fy = (cfg["height"] / 2.0) / np.tan(yfov / 2.0)
    fx = fy * aspect
    cx, cy = cfg["width"] / 2.0, cfg["height"] / 2.0

    os.makedirs(img_dir, exist_ok=True)
    views = []
    vid = 0
    rng = np.random.default_rng(cfg["seed"])
    elevs = cfg["elevations"]
    for ei, el in enumerate(elevs):
        el_rad = np.deg2rad(el)
        for ai in range(cfg["n_az"]):
            az = ai / cfg["n_az"] * 2 * np.pi
            eye = np.array([
                cfg["radius"] * np.cos(el_rad) * np.cos(az),
                cfg["radius"] * np.cos(el_rad) * np.sin(az),
                cfg["radius"] * np.sin(el_rad),
            ])
            R, t = lookat_gl(eye, np.zeros(3))
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            if cfg["bg_mode"] == "random":
                h = rng.uniform(0, 1)
                s = rng.uniform(0.5, 0.95)
                v = rng.uniform(0.4, 0.9)
                r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
                bg = np.array([r_, g_, b_], dtype=np.float32)
                scene.bg_color = bg
            scene.set_pose(cam_node, pose)
            color, _ = renderer.render(scene)
            fname = f"view_{vid:04d}"
            cv2.imwrite(os.path.join(img_dir, fname + ".png"), color[:, :, ::-1])
            views.append({
                "name": fname, "id": vid + 1,
                "R_gl": R.tolist(), "t": t.tolist(),
                "az": az, "el": el_rad,
                "bg": bg.tolist(),
            })
            vid += 1

    # numeric sanity diagnostics (can't view images here).
    # cv2.imread returns BGR; bg is stored RGB — convert image to RGB first,
    # else every pixel looks "different from bg". With random saturated bg the
    # naive brightness threshold also fails, so subtract the per-view bg.
    last = cv2.imread(os.path.join(img_dir, f"view_{vid-1:04d}.png"))[:, :, ::-1]
    last = last.astype(np.float32) / 255.0
    bg_last = np.asarray(views[-1]["bg"], np.float32)
    diff = np.abs(last - bg_last).max(axis=-1)
    fg = last[diff > 0.04]
    frac = fg.shape[0] / (last.shape[0] * last.shape[1])
    lum = 0.299 * fg[..., 0] + 0.587 * fg[..., 1] + 0.114 * fg[..., 2]
    lum_std = float(lum.std()) if len(lum) else 0.0
    lum_mean = float(lum.mean()) if len(lum) else 0.0
    meta = {"width": cfg["width"], "height": cfg["height"], "yfov_deg": cfg["yfov_deg"],
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "n_views": vid}
    with open(os.path.join(out_dir, "poses.json"), "w") as f:
        json.dump({"meta": meta, "views": views}, f, indent=1)
    with open(os.path.join(out_dir, "bg.json"), "w") as f:
        json.dump({v["name"] + ".png": v["bg"] for v in views}, f, indent=1)

    import trimesh as _tm
    wt = None
    try:
        merged = _tm.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        merged.merge_vertices()
        wt = bool(merged.is_watertight)
    except Exception:
        pass
    obj_meta = {
        "name": name, "scale": scale, "bbox_center": center.tolist(),
        "n_verts": int(len(m.vertices)), "n_faces": int(len(m.faces)),
        "watertight": wt, "textured": m.visual.kind == "texture",
        "render_ok": True, "diag_frac": float(frac), "diag_lum_std": lum_std,
        "diag_lum_mean": lum_mean, "t_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(obj_meta, f, indent=1)
    return obj_meta


def is_done(name, cfg):
    out_dir = os.path.join(cfg["out_root"], name)
    if not os.path.isfile(os.path.join(out_dir, "poses.json")):
        return False
    try:
        with open(os.path.join(out_dir, "poses.json")) as f:
            p = json.load(f)
        n = p["meta"]["n_views"]
        imgs = os.listdir(os.path.join(out_dir, "images"))
        return n == cfg["expected_views"] and len(imgs) >= n
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="gso_manifest.csv or a names file (one name/line)")
    ap.add_argument("--meshes", default="/root/gso/meshes")
    ap.add_argument("--out", default="/root/gso/renders")
    ap.add_argument("--n_az", type=int, default=16)
    ap.add_argument("--elevations", type=str, default="20,45,70")
    ap.add_argument("--radius", type=float, default=3.2)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--yfov", type=float, default=40.0)
    ap.add_argument("--bg", type=str, default="0.9,0.9,0.9")
    ap.add_argument("--bg_mode", type=str, default="random", choices=["fixed", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="render only this name (test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="render only first N (0=all)")
    args = ap.parse_args()

    if args.list.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(args.list)
        tasks = [{"name": n, "glb": os.path.join(args.meshes, n + ".glb")}
                 for n in df.loc[df.ok.fillna(False), "name"] if str(n).strip()]
    else:
        names = [ln.strip() for ln in open(args.list) if ln.strip()]
        tasks = [{"name": n, "glb": os.path.join(args.meshes, n + ".glb")} for n in names]

    if args.only:
        tasks = [t for t in tasks if t["name"] == args.only]
    if not tasks:
        print("empty task list")
        return
    # keep only meshes that exist locally
    tasks = [t for t in tasks if os.path.isfile(t["glb"])]

    os.makedirs(args.out, exist_ok=True)
    cfg = {
        "out_root": args.out, "bg_fixed": args.bg, "bg_mode": args.bg_mode,
        "seed": args.seed, "n_az": args.n_az,
        "elevations": [float(e) for e in args.elevations.split(",")],
        "radius": args.radius, "width": args.width, "height": args.height,
        "yfov_deg": args.yfov, "expected_views": args.n_az * len(args.elevations.split(",")),
    }
    global _GLOBAL_CFG
    _GLOBAL_CFG = cfg
    todo = [t for t in tasks if args.force or not is_done(t["name"], cfg)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"task list {len(tasks)}, already done {len(tasks)-len(todo)}, to render {len(todo)}")
    if not todo:
        print("nothing to do")
        return

    if args.only or args.workers <= 1:
        _worker_init(args.width, args.height)
        stats = []
        for i, t in enumerate(todo, 1):
            s = render_one(t, cfg)
            if s is None:
                continue
            stats.append(s)
            print(f"[{i}/{len(todo)}] {t['name']} frac={s['diag_frac']:.3f} "
                  f"lum_std={s['diag_lum_std']:.3f} {s['t_sec']}s", flush=True)
    else:
        with mp.get_context("fork").Pool(args.workers, initializer=_worker_init,
                                         initargs=(args.width, args.height)) as pool:
            results = []
            for i, s in enumerate(pool.imap_unordered(_pool_render, todo), 1):
                if s is None:
                    continue
                results.append(s)
                print(f"[{i}/{len(todo)}] {s['name']} frac={s['diag_frac']:.3f} "
                      f"lum_std={s['diag_lum_std']:.3f} {s['t_sec']}s", flush=True)
            stats = results

    import csv as _csv
    ok = [s for s in stats if s]
    with open(os.path.join(args.out, "render_summary.csv"), "w", newline="") as f:
        if ok:
            w = _csv.DictWriter(f, fieldnames=list(ok[0].keys()))
            w.writeheader()
            w.writerows(ok)
    bad = [s for s in stats if not s]
    print(f"\nDONE: {len(ok)} ok, {len(bad)} failed")

    if ok:
        fracs = np.array([s["diag_frac"] for s in ok])
        stds = np.array([s["diag_lum_std"] for s in ok])
        print(f"diag_frac  med={np.median(fracs):.3f} min={fracs.min():.3f}  "
              f"([0.25,0.95] expected)")
        print(f"lum_std    med={np.median(stds):.3f}  "
              f"(>0.03 implies texture/shading visible)")


if __name__ == "__main__":
    main()
