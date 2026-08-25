import trimesh, numpy as np
names = ["BEDROOM_NEO","ALPHABET_AZ_GRADIENT","45oz_RAMEKIN_ASST_DEEP_COLORS","30_CONSTRUCTION_SET","3D_Dollhouse_Swing","Diamond_Visions_Scissors_Red"]
for n in names:
    m = trimesh.load(f"/root/gso/meshes/{n}.glb", force="mesh")
    c = m.bounds.mean(0)
    r = float(np.linalg.norm(m.vertices - c, axis=1).max())
    ro = float(np.linalg.norm(m.vertices, axis=1).max())  # radius w.r.t. raw origin
    print(f"{n:36s} bbox_center=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  bsphere_r={r:.4f}  r_origin={ro:.4f}")
