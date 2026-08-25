"""E0: Poisson mesh reconstruction from pixel-level inversion clouds.
For each object: filter far outliers, orient normals, Poisson (depth=8),
crop to bbox, density-filter, then compare the reconstructed mesh against
the GT mesh: symmetric sampled Chamfer, coverage@0.2, Hausdorff.
"""
import json
import os

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.expanduser("~/e0lab/e0")
OBJS = ["torus", "vase", "rocky", "mug", "sphere", "bumpy"]
N_SAMP = 30000
DEPTH = 8
FAR = 0.30            # drop inversion points farther than this from GT surface

os.makedirs(os.path.join(HERE, "poisson"), exist_ok=True)
for obj in OBJS:
    raw = np.load(os.path.join(HERE, "pixel_clouds", f"{obj}.npy"))   # Nx4
    keep = raw[:, 3] < FAR
    pts = raw[keep, :3]
    print(f"[poisson] {obj}: {len(raw)} -> {len(pts)} pts (far<{FAR})", flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

    mesh_o, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=DEPTH)
    mesh_o = mesh_o.crop(o3d.geometry.AxisAlignedBoundingBox(
        np.array([-1.2] * 3), np.array([1.2] * 3)))
    d = np.asarray(densities)
    keepv = d > np.quantile(d, 0.05)
    mesh_o.remove_vertices_by_mask(~keepv)
    mesh_o.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(HERE, "poisson", f"{obj}.ply"), mesh_o)
    print(f"[poisson] {obj}: mesh {len(mesh_o.vertices)}v {len(mesh_o.triangles)}f",
          flush=True)

    gt = trimesh.load(os.path.join(HERE, "data", "meshes", f"{obj}.ply"), force="mesh")
    rec = trimesh.load(os.path.join(HERE, "poisson", f"{obj}.ply"), force="mesh")
    s_gt = gt.sample(N_SAMP)
    s_rec = rec.sample(N_SAMP)
    d_rec2gt = cKDTree(s_gt).query(s_rec)[0]
    d_gt2rec = cKDTree(s_rec).query(s_gt)[0]
    chamfer = (d_rec2gt.mean() + d_gt2rec.mean()) / 2
    res = dict(
        obj=obj, n_in=len(raw), n_used=len(pts), n_vert=int(len(mesh_o.vertices)),
        n_face=int(len(mesh_o.triangles)),
        rec2gt_p50=float(np.median(d_rec2gt)),
        rec2gt_mean=float(d_rec2gt.mean()),
        gt2rec_mean=float(d_gt2rec.mean()),
        cov_02=float(np.mean(d_gt2rec < 0.2)),
        cov_01=float(np.mean(d_gt2rec < 0.1)),
        chamfer_mean=float(chamfer),
        hausdorff=float(max(d_rec2gt.max(), d_gt2rec.max())),
    )
    with open(os.path.join(HERE, "poisson", f"{obj}.res.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, ensure_ascii=False), flush=True)

print("POISSON_ALL_DONE")
