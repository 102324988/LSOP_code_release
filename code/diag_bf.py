import json, sys
import numpy as np
import torch
sys.path.insert(0, "/root/gof")
from scene.colmap_loader import read_points3D_text
from scene.dataset_readers import readColmapSceneInfo

for name in ["BEDROOM_NEO", "ALPHABET_AZ_GRADIENT", "45oz_RAMEKIN_ASST_DEEP_COLORS",
             "30_CONSTRUCTION_SET", "Diamond_Visions_Scissors_Red"]:
    m = json.load(open(f"/root/gso/renders/{name}/meta.json"))
    print(f"{name:38s} frac={m['diag_frac']:.3f} lum_std={m['diag_lum_std']:.3f} "
          f"nv={m['n_verts']:6d} nf={m['n_faces']:6d} wt={m['watertight']}")


def filter_stats(name):
    scene = f"/root/gso/renders/{name}"
    info = readColmapSceneInfo(scene, "images", False, llffhold=999999)
    pts = info.point_cloud.points
    xyz = torch.tensor(pts).float().cuda()
    cams = info.train_cameras
    dist = torch.ones(len(xyz)).cuda() * 100000.0
    focal = 0.0
    for c in cams:
        R = torch.tensor(c.R).cuda().float()
        T = torch.tensor(c.T).cuda().float()
        w = getattr(c, "image_width", getattr(c, "width", 800))
        h = getattr(c, "image_height", getattr(c, "height", 600))
        fx = getattr(c, "focal_x", 1099.0)
        fy = getattr(c, "focal_y", fx)
        xc = xyz @ R + T
        zd = xc[:, 2] > 0.2
        x, y, z = xc[:, 0], xc[:, 1], xc[:, 2]
        z = torch.clamp(z, min=0.001)
        x = x / z * fx + w / 2.0
        y = y / z * fy + h / 2.0
        ins = ((x >= -0.15 * w) & (x <= 1.15 * w) &
               (y >= -0.15 * h) & (y <= 1.15 * h))
        valid = zd & ins
        if valid.any():
            dist[valid] = torch.minimum(dist[valid], z[valid])
        focal = max(focal, fx)
    f3d = dist / focal * (0.2 ** 0.5)
    nvalid = int((dist < 100000.0).sum())
    nan_n = int(torch.isnan(dist).sum())
    print(f"{name:38s} npts={len(xyz):5d} valid_any_cam={nvalid:5d} "
          f"nan={nan_n:4d}  filter3D[min med max]="
          f"[{f3d.min():.5f} {f3d.median():.5f} {f3d.max():.5f}]  "
          f"z-dist[min med max]=[{dist.min():.3f} {dist.median():.3f} {dist.max():.3f}]")


for name in ["BEDROOM_NEO", "ALPHABET_AZ_GRADIENT", "30_CONSTRUCTION_SET"]:
    try:
        filter_stats(name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(name, "ERR", e)
