"""E0: sample an initial point cloud from the GT mesh -> COLMAP points3D.txt.

GOF has no random-init: it requires a non-empty initial point cloud. Sampling
points on the GT surface gives the Gaussians a good start (and matches the
standard object-centric 3DGS practice).

CRITICAL: the mesh is normalized IDENTICALLY to the D2 renderer (bbox center ->
origin, bounding-sphere radius -> 1.0). The COLMAP cameras live in that
normalized frame; sampling in the raw GSO frame would leave the init cloud
misaligned with the cameras (a latent bug that made some objects' Gaussians
collapse at densification).
"""
import argparse
import os

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n_points", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mesh = trimesh.load(args.mesh, force="mesh")

    # ---- normalize identically to gso_render_batch.py D2 ----
    center = mesh.bounds.mean(0)
    mesh.vertices = mesh.vertices - center
    sph_r = float(np.linalg.norm(mesh.vertices, axis=1).max())
    if sph_r > 0:
        mesh.apply_scale(1.0 / max(sph_r, 1e-9))

    pts, _ = trimesh.sample.sample_surface(mesh, args.n_points)
    pts = pts + rng.normal(0, 0.0015, pts.shape)  # tiny jitter off the surface

    # colors: reuse the checker palette so SH-DC init sees texture
    g = np.floor(pts * 7.0).astype(int)
    parity = (g[:, 0] + g[:, 1] + g[:, 2]) % 2
    rgb = np.where(parity[:, None], np.array([205, 215, 230]), np.array([85, 95, 120]))

    sparse = os.path.join(args.scene, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    stale = os.path.join(sparse, "points3D.ply")
    if os.path.exists(stale):
        os.remove(stale)

    # GOF's read_points3D_text expects: POINT_ID X Y Z R G B ERROR
    lines = []
    for i, (p, c) in enumerate(zip(pts, rgb)):
        lines.append(f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 0.0")
    with open(os.path.join(sparse, "points3D.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {len(lines)} init points (normalized frame) -> {sparse}/points3D.txt")


if __name__ == "__main__":
    main()
