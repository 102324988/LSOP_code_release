"""E0 dry-run: procedural meshes with checker vertex colors.
Each mesh is centered at origin and scaled to max radius ~0.95.
"""
import os
import numpy as np
import trimesh


def checker_colors(verts, scale=7.0, c1=(205, 215, 230), c2=(85, 95, 120)):
    g = np.floor(verts * scale).astype(int)
    parity = (g[:, 0] + g[:, 1] + g[:, 2]) % 2
    cols = np.where(parity[:, None], np.array(c1), np.array(c2)).astype(np.uint8)
    return np.concatenate([cols, np.full((len(verts), 1), 255, np.uint8)], axis=1)


def normalize_and_save(mesh, name, outdir, radius_target=0.95, check=7.0):
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    r = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.apply_scale(radius_target / r)
    mesh.visual.vertex_colors = checker_colors(mesh.vertices, check)
    mesh.export(os.path.join(outdir, name + ".ply"))
    print(f"{name}: V={len(mesh.vertices)} F={len(mesh.faces)} maxr={radius_target:.2f}")


def rocky_sphere(subdiv=4, amp=0.16):
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    v = m.vertices
    n = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / n
    theta = np.arccos(u[:, 2])
    phi = np.arctan2(u[:, 1], u[:, 0])
    noise = (np.sin(6 * theta) * np.cos(5 * phi)
             + 0.5 * np.sin(13 * theta + 2 * phi) * np.cos(7 * phi))
    m.vertices = (1.0 + amp * noise)[:, None] * u
    return m


def vase():
    prof = np.array([[0.00, 0.42], [0.28, 0.52], [0.48, 0.42], [0.52, 0.05],
                     [0.40, -0.30], [0.22, -0.40], [0.00, -0.40]])
    return trimesh.creation.revolve(prof, resolution=64)


def mug():
    body = trimesh.creation.cylinder(radius=0.34, height=0.75, sections=64)
    body.apply_translation([0, 0, -0.05])
    handle = trimesh.creation.torus(major_radius=0.26, minor_radius=0.07)
    handle.apply_translation([0.55, 0.0, 0.15])
    return trimesh.util.concatenate([body, handle])


def torus():
    return trimesh.creation.torus(major_radius=0.55, minor_radius=0.23)


def bumpy_cube(s=0.7, amp=0.10):
    m = trimesh.creation.box(extents=[2 * s] * 3)
    v = m.vertices
    r = np.linalg.norm(v, axis=1, keepdims=True)
    r[..., 0][r[..., 0] == 0] = 1e-6
    bump = amp * np.sin(4 * r / (s * np.sqrt(3)) * np.pi)
    m.vertices = v * (1.0 + bump)
    return m


def main():
    outdir = os.path.join(os.path.dirname(__file__), "data", "meshes")
    os.makedirs(outdir, exist_ok=True)
    normalize_and_save(rocky_sphere(), "rocky", outdir)
    normalize_and_save(vase(), "vase", outdir)
    normalize_and_save(mug(), "mug", outdir)
    normalize_and_save(torus(), "torus", outdir)
    normalize_and_save(bumpy_cube(), "bumpy", outdir)
    normalize_and_save(trimesh.creation.icosphere(subdivisions=3, radius=1.0), "sphere", outdir)
    print("done")


if __name__ == "__main__":
    main()
