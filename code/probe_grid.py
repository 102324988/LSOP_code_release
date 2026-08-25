"""Probe a GOF opacity grid at known geometric points + compare field mesh bbox to GT."""
import argparse
import json
import os

import numpy as np
import trimesh


def trilinear(grid, bbox, pts):
    res = grid.shape[0]
    lo_b = np.array(bbox[0::2]); hi_b = np.array(bbox[1::2])
    g = (pts - lo_b) / (hi_b - lo_b) * (res - 1)
    lo = np.floor(g).astype(int).clip(0, res - 2)
    w = g - lo
    def s(x, y, z):
        return grid[x, y, z]
    c000 = s(lo[:,0],lo[:,1],lo[:,2]); c100 = s(lo[:,0]+1,lo[:,1],lo[:,2])
    c010 = s(lo[:,0],lo[:,1]+1,lo[:,2]); c110 = s(lo[:,0]+1,lo[:,1]+1,lo[:,2])
    c001 = s(lo[:,0],lo[:,1],lo[:,2]+1); c101 = s(lo[:,0]+1,lo[:,1],lo[:,2]+1)
    c011 = s(lo[:,0],lo[:,1]+1,lo[:,2]+1); c111 = s(lo[:,0]+1,lo[:,1]+1,lo[:,2]+1)
    x0,x1,y0,y1,z0,z1 = w[:,0],1-w[:,0],w[:,1],1-w[:,1],w[:,2],1-w[:,2]
    return (x1*(y1*(z1*c000+z0*c001)+y0*(z1*c010+z0*c011))
            + x0*(y1*(z1*c100+z0*c101)+y0*(z1*c110+z0*c111)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--mesh", required=True, help="GT mesh")
    args = ap.parse_args()

    grid = np.load(os.path.join(args.grid, "grid.npy"))
    with open(os.path.join(args.grid, "meta.json")) as f:
        meta = json.load(f)
    bbox = np.array(meta["bbox"], dtype=np.float32)
    res = meta["res"]
    print(f"grid res={res} bbox={bbox}")

    # GT torus (normalized): major 0.67, minor 0.28, in XY plane
    R, r = 0.67, 0.28
    pts = {
        "outer_equator":   np.array([R + r, 0.0, 0.0]),
        "top_of_tube":     np.array([R, 0.0, r]),
        "inner_equator":   np.array([R - r, 0.0, 0.0]),
        "hole_center":     np.array([0.0, 0.0, 0.0]),
        "far_empty":       np.array([0.0, 1.2, 0.0]),
        "beyond_surface":  np.array([1.05, 0.0, 0.0]),
    }
    P = np.stack(list(pts.values()))
    vals = trilinear(grid, bbox, P)
    for (name, _), v in zip(pts.items(), vals):
        print(f"  {name:>16s}: field={v:.3f}   (expect 1.0 if occupied, 0.0 if empty)")

    # field mesh bbox vs GT bbox
    from skimage.measure import marching_cubes
    spacing = (bbox[1::2] - bbox[0::2]) / (res - 1)
    verts, faces, _, _ = marching_cubes(grid, level=0.5, spacing=spacing)
    world = bbox[0::2] + verts
    mf = trimesh.Trimesh(vertices=world, faces=faces, process=True)
    print(f"field mesh: V={len(mf.vertices)} bbox_min={mf.bounds[0]} bbox_max={mf.bounds[1]}")
    gt = trimesh.load(args.mesh, force="mesh")
    print(f"GT     mesh: bbox_min={gt.bounds[0]} bbox_max={gt.bounds[1]}")
    print(f"field centroid={mf.centroid}  GT centroid={gt.centroid}")


if __name__ == "__main__":
    main()
