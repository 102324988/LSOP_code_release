import numpy as np, json, re, os
from plyfile import PlyData

names = ["3D_Dollhouse_Lamp", "ALPHABET_AZ_GRADIENT", "30_CONSTRUCTION_SET", "50_BLOCKS",
         "45oz_RAMEKIN_ASST_DEEP_COLORS", "3D_Dollhouse_Refrigerator", "3D_Dollhouse_Swing",
         "BEDROOM_NEO", "5_HTP", "Balderdash_Game", "Diamond_Visions_Scissors_Red",
         "3D_Dollhouse_Sink"]
rows = []
for n in names:
    log = f"output/gso_pilot/{n}.log"
    psnr = None
    try:
        m = re.findall(r"PSNR ([\d.]+)", open(log).read())
        if m:
            psnr = float(m[-1])
    except Exception:
        pass
    ply = f"output/gso_pilot/{n}/point_cloud/iteration_6000/point_cloud.ply"
    npts = None
    r = None
    op = None
    if os.path.isfile(ply):
        d = PlyData.read(ply)
        v = d["vertex"]
        npts = len(v.data)
        xyz = np.column_stack([v["x"], v["y"], v["z"]])
        r = float(np.linalg.norm(xyz - xyz.mean(0), axis=1).max())
        if "opacity" in v.properties:
            o = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
            op = float(np.mean(o > 0.5))
    meta = json.load(open(f"/root/gso/renders/{n}/meta.json"))
    rows.append((n, psnr, npts, r, op, meta["diag_frac"], meta["watertight"]))

print(f"{'name':38s} {'PSNR':>6s} {'gauss':>7s} {'r':>6s} {'op>0.5':>7s} {'frac':>6s} {'wt':>5s}")
for n, psnr, npts, r, op, frac, wt in rows:
    ps = f"{psnr:.2f}" if psnr is not None else "--"
    np_ = str(npts) if npts is not None else "--"
    rr = f"{r:.3f}" if r is not None else "--"
    oo = f"{op:.3f}" if op is not None else "--"
    print(f"{n:38s} {ps:>6s} {np_:>7s} {rr:>6s} {oo:>7s} {frac:.3f} {str(wt):>5s}")
