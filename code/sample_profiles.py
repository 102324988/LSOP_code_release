"""E0: radial occupancy profiles from the GOF opacity grid + inversion test.

For every ray direction (theta,phi) on an equirectangular grid, march r in
[0, Rmax] and compute:
    o(r)   : opacity at r (trilinear from the grid)
    T(r)   : transmittance, exp(-cumsum(o)*dr)
    P(r)   : stopping density, T(r)*(1-exp(-o(r)*dr))   <- the "occupancy profile"

Inversion test: backproject o(r) into a fresh occupancy grid (average over
rays), Marching Cubes at 0.5 -> mesh, compare to the GT mesh (Chamfer L1 +
F-score). Also meshes the GOF field directly to isolate inversion error.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree
import trimesh

RADIUS_MAX = None  # set from grid bbox half-diagonal


def trilinear_sample(grid_t, pts):
    """pts: (M,3) in grid coords [0, G-1]; returns opacity (M,)."""
    G = grid_t.shape[0]
    lo = torch.floor(pts).long()
    lo = lo.clamp(0, G - 2)
    hi = lo + 1
    w = pts - lo.float()
    # gather 8 corners
    def g(x, y, z):
        return grid_t[x, y, z]
    c000 = g(lo[:, 0], lo[:, 1], lo[:, 2]); c100 = g(hi[:, 0], lo[:, 1], lo[:, 2])
    c010 = g(lo[:, 0], hi[:, 1], lo[:, 2]); c110 = g(hi[:, 0], hi[:, 1], lo[:, 2])
    c001 = g(lo[:, 0], lo[:, 1], hi[:, 2]); c101 = g(hi[:, 0], lo[:, 1], hi[:, 2])
    c011 = g(lo[:, 0], hi[:, 1], hi[:, 2]); c111 = g(hi[:, 0], hi[:, 1], hi[:, 2])
    x0, x1, y0, y1, z0, z1 = w[:, 0], 1 - w[:, 0], w[:, 1], 1 - w[:, 1], w[:, 2], 1 - w[:, 2]
    return (x1 * (y1 * (z1 * c000 + z0 * c001) + y0 * (z1 * c010 + z0 * c011))
            + x0 * (y1 * (z1 * c100 + z0 * c101) + y0 * (z1 * c110 + z0 * c111)))


def world_to_grid(pts, bbox, res):
    """(M,3) world -> (M,3) grid coords (cuda-safe)."""
    b = torch.from_numpy(bbox[0::2]).to(pts.device).float()
    s = torch.from_numpy(bbox[1::2] - bbox[0::2]).to(pts.device).float()
    return (pts - b) / s * (res - 1)


def sample_metrics(mesh_a, mesh_b, n=50000, t=0.02):
    """Chamfer L1 + F-score between two meshes (sampled point clouds)."""
    pa = trimesh.sample.sample_surface(mesh_a, n)[0]
    pb = trimesh.sample.sample_surface(mesh_b, n)[0]
    kd_a, kd_b = cKDTree(pa), cKDTree(pb)
    dab, _ = kd_b.query(pa)
    dba, _ = kd_a.query(pb)
    chamfer = float((dab.mean() + dba.mean()) / 2)
    f_a = float((dab < t).mean())
    f_b = float((dba < t).mean())
    fscore = 2 * f_a * f_b / (f_a + f_b + 1e-12)
    return chamfer, f_a, f_b, fscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--mesh", required=True, help="GT mesh path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_theta", type=int, default=256)
    ap.add_argument("--n_phi", type=int, default=512)
    ap.add_argument("--n_bins", type=int, default=128)
    ap.add_argument("--chunk_rays", type=int, default=2048)
    ap.add_argument("--mc_level", type=float, default=0.5)
    args = ap.parse_args()

    grid = np.load(os.path.join(args.grid, "grid.npy"))
    with open(os.path.join(args.grid, "meta.json")) as f:
        meta = json.load(f)
    bbox = np.array(meta["bbox"], dtype=np.float32)
    res = meta["res"]
    G = torch.from_numpy(grid).cuda()

    span = bbox[1::2] - bbox[0::2]
    rmax = float(np.linalg.norm(span) / 2)  # half-diagonal covers all content
    dr = rmax / (args.n_bins - 1)
    r = np.linspace(0.0, rmax, args.n_bins, dtype=np.float32)

    # equirectangular ray directions (theta in (0,pi), phi in [0,2pi))
    th = np.linspace(1e-3, np.pi - 1e-3, args.n_theta)
    ph = np.linspace(0.0, 2 * np.pi, args.n_phi, endpoint=False)
    PH, TH = np.meshgrid(ph, th, indexing="ij")  # shape (n_phi, n_theta)
    dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
    dirs = dirs.reshape(-1, 3).astype(np.float32)  # (n_theta*n_phi, 3)
    n_rays = dirs.shape[0]

    profiles = np.zeros((n_rays, args.n_bins), dtype=np.float32)
    opacity_profiles = np.zeros_like(profiles)
    depth_mean = np.zeros(n_rays, dtype=np.float32)
    depth_peak = np.zeros(n_rays, dtype=np.float32)
    coverage = np.zeros(n_rays, dtype=np.float32)

    # backprojection accumulators
    occ_acc = np.zeros(res ** 3, dtype=np.float64)
    occ_cnt = np.zeros(res ** 3, dtype=np.float64)

    r_t = torch.from_numpy(r).cuda()
    for i0 in range(0, n_rays, args.chunk_rays):
        d = torch.from_numpy(dirs[i0:i0 + args.chunk_rays]).cuda()          # (R,3)
        R = d.shape[0]
        pts = (r_t[None, :, None] * d[:, None, :]).reshape(-1, 3)           # (R*B,3)
        o = trilinear_sample(G, world_to_grid(pts, bbox, res)).reshape(R, -1)
        o = o.clamp(0.0, 1.0)
        a = 1.0 - torch.exp(-o * dr)                                  # slab opacity
        T_enter = torch.cumprod(torch.cat([torch.ones_like(a[:, :1]),
                                          1.0 - a], dim=1), dim=1)[:, :-1]  # T before bin i
        P = T_enter * a                                               # stopping density

        oc = o.cpu().numpy(); pc = P.cpu().numpy()
        profiles[i0:i0 + R] = pc
        opacity_profiles[i0:i0 + R] = oc
        depth_mean[i0:i0 + R] = (pc * r_t.cpu().numpy()[None, :]).sum(axis=1)
        depth_peak[i0:i0 + R] = r[np.argmax(pc, axis=1)]
        coverage[i0:i0 + R] = pc.sum(axis=1)

        # backproject opacity into voxel grid
        gcoords = world_to_grid(pts, bbox, res)
        idx = (gcoords[:, 0].floor().long().clamp(0, res - 1) * res * res
               + gcoords[:, 1].floor().long().clamp(0, res - 1) * res
               + gcoords[:, 2].floor().long().clamp(0, res - 1))
        occ_acc += np.bincount(idx.cpu().numpy(), weights=oc.reshape(-1), minlength=res ** 3)
        occ_cnt += np.bincount(idx.cpu().numpy(), minlength=res ** 3)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "profiles.npy"), profiles.reshape(args.n_phi, args.n_theta, -1))
    np.save(os.path.join(args.out, "opacity_profiles.npy"),
            opacity_profiles.reshape(args.n_phi, args.n_theta, -1))
    np.save(os.path.join(args.out, "depth_mean.npy"), depth_mean)
    np.save(os.path.join(args.out, "depth_peak.npy"), depth_peak)
    np.save(os.path.join(args.out, "coverage.npy"), coverage)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"n_theta": args.n_theta, "n_phi": args.n_phi, "n_bins": args.n_bins,
                   "rmax": rmax, "dr": dr, "grid": meta,
                   "coverage_mean": float(coverage.mean())}, f, indent=1)

    print(f"[profiles] mean coverage={coverage.mean():.3f} "
          f"mean depth(peak)={depth_peak.mean():.3f} -> {args.out}")

    # ---- inversion: profiles -> occupancy grid -> mesh ----
    cnt = np.where(occ_cnt > 0, occ_cnt, 1)
    occ = (occ_acc / cnt).reshape(res, res, res)
    np.save(os.path.join(args.out, "inverted_occupancy.npy"), occ)

    from skimage.measure import marching_cubes
    spacing = span / (res - 1)
    occ_b = occ.copy()
    try:
        verts, faces, _, _ = marching_cubes(occ_b, level=args.mc_level, spacing=spacing)
        world_verts = bbox[0::2] + verts
        mesh_inv = trimesh.Trimesh(vertices=world_verts, faces=faces, process=True)
        mesh_inv.export(os.path.join(args.out, "mesh_inverted.ply"))
        mesh_inv.export(os.path.join(args.out, "mesh_inverted.obj"))
        print(f"[invert] MC mesh: V={len(mesh_inv.vertices)} F={len(mesh_inv.faces)}")
    except Exception as e:  # noqa: BLE001
        print(f"[invert] marching cubes failed: {e}")
        mesh_inv = None

    gt = trimesh.load(args.mesh, force="mesh")
    if mesh_inv is not None and len(mesh_inv.faces) >= 8:
        ch, fa, fb, fs = sample_metrics(gt, mesh_inv)
        print(f"[invert] vs GT: Chamfer={ch:.4f} F@0.02={fs:.4f} (f_a={fa:.3f} f_b={fb:.3f})")
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump({"chamfer": ch, "fscore": fs, "f_a": fa, "f_b": fb}, f, indent=1)
    else:
        print("[invert] mesh degenerate (<8 faces), metrics skipped")

    # direct MC on the GOF field grid (isolates profile-inversion loss)
    try:
        verts, faces, _, _ = marching_cubes(grid, level=args.mc_level, spacing=spacing)
        world_verts = bbox[0::2] + verts
        mesh_field = trimesh.Trimesh(vertices=world_verts, faces=faces, process=True)
        mesh_field.export(os.path.join(args.out, "mesh_field.ply"))
        if len(mesh_field.faces) >= 8:
            ch2, fa2, fb2, fs2 = sample_metrics(gt, mesh_field)
            print(f"[field ] vs GT: Chamfer={ch2:.4f} F@0.02={fs2:.4f} (f_a={fa2:.3f} f_b={fb2:.3f})")
            if mesh_inv is not None and len(mesh_inv.faces) >= 8:
                ch3, fa3, fb3, fs3 = sample_metrics(mesh_field, mesh_inv)
                print(f"[inv->field] mesh Chamfer={ch3:.4f} F@0.02={fs3:.4f}")
        else:
            print("[field ] mesh degenerate, metrics skipped")
    except Exception as e:  # noqa: BLE001
        print(f"[field ] marching cubes failed: {e}")


if __name__ == "__main__":
    main()
